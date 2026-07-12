"""Local CAA driver. Replaces ``EasyEdit/examples/steer_eval.py``.

CLI mirrors ``shell/steer_eval.sh`` flags 1:1. Path conventions match the EasyEdit
upstream so ``shell/convert.py``, ``shell/score.py`` and ``shell/tune.py`` keep
working unchanged:

    vector:    output/vectors/<model>/<dataset>/<run>/steer_eval_concept_<cid>/caa_vector/layer_<L>.pt
    tmp gen:   output/generation/<model>/<dataset>/<run>/caa/layer_<L>_multip_<M>/tmp_generation_results_<exp>.json
    final gen: output/generation/.../all_generation_results_<exp>.json
    log:       output/logs/<model>/<dataset>/<run>/caa/layer_<L>_multip_<M>.log

Each generated item clears the MPS allocator cache so the process stays fast
across 24 concepts (lost EasyEdit patch preserved here).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.caa.apply import apply_caa
from src.caa.generate import generate_caa_vectors
from src.caa.hparams import ApplyCAAHyperParams, CAAHyperParams
from src.model import make_qwen_wrapper
from src.util.templates import build_model_input


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def vector_dir(model: str, dataset: str, run: str) -> Path:
    return PROJECT_ROOT / "output" / "vectors" / model / dataset / run


def vector_concept_dir(model: str, dataset: str, run: str, cid: str) -> Path:
    return vector_dir(model, dataset, run) / f"steer_eval_concept_{cid}"


def _layers_tag(layers: List[int]) -> str:
    return "_".join(str(l) for l in layers)


def generation_dir(model: str, dataset: str, run: str, method: str, layers: List[int], multiplier: float) -> Path:
    return (
        PROJECT_ROOT / "output" / "generation" / model / dataset / run
        / method / f"layer_{_layers_tag(layers)}_multip_{multiplier}"
    )


def logs_dir(model: str, dataset: str, run: str, method: str, layers: List[int], multiplier: float) -> Path:
    return (
        PROJECT_ROOT / "output" / "logs" / model / dataset / run
        / method / f"layer_{_layers_tag(layers)}_multip_{multiplier}.log"
    )


def data_path(dataset: str, exp: str) -> Path:
    """Resolve the train/valid split file honouring $STEER_DATA_DIR for the test subset."""
    base = Path(os.environ.get("STEER_DATA_DIR", str(PROJECT_ROOT / "data")))
    return base / dataset / f"{exp}.json"


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_by_concept(items: list) -> dict:
    """Group items by concept_id, preserving first-seen order."""
    out = {}
    for it in items:
        cid = it.get("concept_id", "?")
        out.setdefault(cid, []).append(it)
    return out


# --------------------------------------------------------------------------- #
# Generation primitives
# --------------------------------------------------------------------------- #
def _empty_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_one(model, item: dict, generation_params: dict, use_chat_template: bool, logits_processor=None, from_pos=None):
    """Run the model on a single ``item`` and return the steered ``pred`` string.

    If ``logits_processor`` is given (e.g. a :class:`PhraseCountProcessor`), it is
    passed to ``model.generate`` so per-step hooks can be updated mid-generation.

    ``from_pos``: if ``"prompt_len"``, the activation-addition is restricted to
    generated tokens only (positions >= prompt length), so the prompt's KV cache
    is not poisoned by the steering vector. If an int, used directly. None = all
    positions (legacy CAA behavior).
    """
    question = item.get("question") or item.get("input") or ""
    if not question.strip():
        raise ValueError(f"empty prompt for item keys={sorted(item.keys())}")
    prompt = build_model_input(
        question, model.tokenizer,
        system_prompt="", use_chat_template=use_chat_template,
        enable_thinking=getattr(model.hparams, "enable_thinking", None),
    )
    inputs = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=not use_chat_template).to(model.device)
    # Restrict steering to generated positions (keeps the prompt KV cache clean).
    if from_pos == "prompt_len":
        model.set_from_position(int(inputs["input_ids"].shape[1]))
    elif isinstance(from_pos, int):
        model.set_from_position(from_pos)
    with torch.no_grad():
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            output = model.model.generate(**inputs, logits_processor=LogitsProcessorList([logits_processor]), **generation_params)
        else:
            output = model.model.generate(**inputs, **generation_params)
    model.set_from_position(None)  # restore default for the next item
    full_output = model.tokenizer.decode(output[0], skip_special_tokens=False)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Explicitly release the KV cache + output tensors before returning — otherwise
    # Python GC lags and torch.mps.empty_cache() can't reclaim the cache, so memory
    # fragments and each successive generate() gets slower (very visible at long
    # max_new_tokens).
    del output, new_tokens
    import gc as _gc
    _gc.collect()
    _empty_cache()
    return {"complete_output": [full_output], "pred": [text]}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Local CAA driver for Qwen3.")
    parser.add_argument("--model_name_or_path", default=str(PROJECT_ROOT / "qwen3-4b"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--method", default="caa")
    parser.add_argument("--dataset", default="SteerEval/personality")
    parser.add_argument("--generate_vector", default="true")
    parser.add_argument("--gen_out_path", default="baseline_v1")
    parser.add_argument("--generate_response", default="true")
    parser.add_argument("--generate_orig_output", default="false")
    parser.add_argument("--evaluate", default="false", help="Ignored; shell/score.py is used.")
    parser.add_argument("--layers", default="22,26,30",
                        help="comma-separated layer indices, e.g. 22,26,30")
    parser.add_argument("--multipliers", default="4")
    parser.add_argument("--mode", default="caa_diverge",
                        choices=["caa_mean", "caa_last", "caa_diverge", "token", "token_logit"],
                        help="caa_* = CAA extraction mode; token = L3 residual injection; token_logit = L3 direct logit bias (clean)")
    parser.add_argument("--concept_config", default=None,
                        help="optional JSON {cid: {mode, layers, multiplier}} overriding the globals per concept")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--exp", default="valid")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma-separated concept ids to run, e.g. L1_1 (others skipped). "
                             "Implies --merge: loads existing final, replaces only these concepts, writes back.")
    parser.add_argument("--merge", action="store_true",
                        help="merge results into existing all_generation_results_<exp>.json "
                             "(update run concepts in place) instead of starting fresh.")
    args = parser.parse_args()

    if args.method != "caa":
        print(f"[error] only 'caa' is supported in this driver, got {args.method!r}", file=sys.stderr)
        return 1

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    multiplier = float(args.multipliers)
    agg = args.mode.replace("caa_", "") if args.mode.startswith("caa_") else "mean"
    is_token_mode = args.mode == "token"
    concept_overrides = {}
    if args.concept_config:
        with open(args.concept_config, "r", encoding="utf-8") as f:
            concept_overrides = json.load(f)

    model_short = Path(args.model_name_or_path).name  # e.g. "qwen3-4b"

    # ---- Optionally wipe this run's generation outputs.
    if args.clean:
        gen_root = PROJECT_ROOT / "output" / "generation" / model_short / args.dataset / args.gen_out_path
        if gen_root.exists():
            import shutil

            print(f"[clean] removing {gen_root}")
            shutil.rmtree(gen_root)

    # ---- Load data.
    dataset_name = args.dataset  # e.g. "SteerEval/personality"
    train_path = data_path(dataset_name, "train")
    valid_path = data_path(dataset_name, args.exp)
    train_data = load_json(train_path)
    valid_data = load_json(valid_path)
    train_by_concept = split_by_concept(train_data)
    valid_by_concept = split_by_concept(valid_data)
    print(
        f"train concepts={len(train_by_concept)}, "
        f"{args.exp} concepts={len(valid_by_concept)}"
    )

    # ---- Hparams.
    dtype_str = args.dtype
    enable_thinking = False  # Qwen3-Instruct: skip the thinking block for all cases.

    def to_torch_dtype(s: str):
        return {"float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(s, torch.float16)

    # ---- Build model once.
    print(f"Loading model {args.model_name_or_path} on {args.device} ({dtype_str})")
    gen_hparams = CAAHyperParams(
        layers=layers,
        multiple_choice=False,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=dtype_str,
        use_cache=True,
        use_chat_template=True,
        system_prompt="",
        save_activations=True,
        enable_thinking=enable_thinking,
        agg=agg,
        normalize=True,
        diverge_win=8,
    )
    model, _tokenizer = make_qwen_wrapper(gen_hparams)
    model.hparams = gen_hparams
    model.eval()

    # Path conventions for this (run, layers, multiplier).
    vec_root = vector_dir(model_short, dataset_name, args.gen_out_path)
    # When per-concept overrides are in use, suffix the output dir so the
    # misleading global [layers]_[multip] tag (e.g. layer_22_26_30_multip_4.0)
    # is not mistaken for what was actually applied.
    run_tag = args.gen_out_path
    if concept_overrides:
        run_tag = args.gen_out_path + "_perconcept"
    gen_root = generation_dir(model_short, dataset_name, run_tag, args.method, layers, multiplier)
    gen_root.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "logs" / model_short / dataset_name / run_tag / args.method).mkdir(parents=True, exist_ok=True)

    # Per-concept iteration.
    concept_ids = list(valid_by_concept.keys())
    # --only: filter to a subset of concepts (for fast single-concept iteration).
    only_ids = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    if only_ids:
        concept_ids = [c for c in concept_ids if c in only_ids]
    merge_mode = bool(only_ids) or args.merge
    print(f"Will process {len(concept_ids)} concepts (mode={args.mode}, layers={layers}, m={multiplier}"
          f"{f', only={sorted(only_ids)}' if only_ids else ''}, merge={merge_mode})")

    # Merge mode: load existing final as base, drop the concepts we're re-running,
    # and write back to final (keeps the other concepts' results unchanged).
    final_path = gen_root / f"all_generation_results_{args.exp}.json"
    tmp_path = gen_root / f"tmp_generation_results_{args.exp}.json"
    if merge_mode:
        existing: List[dict] = load_json(final_path) if final_path.exists() else []
        existing = [c for c in existing if c.get("concept_id") not in set(concept_ids)]
        done_concepts = {c.get("concept_id") for c in existing}
    else:
        existing = []
        done_concepts = set()

    for cid in concept_ids:
        train_concept = train_by_concept.get(cid, [])
        if not train_concept:
            print(f"[skip] no train data for {cid}")
            continue

        # ---- Resolve effective config (global vs per-concept override).
        ov = concept_overrides.get(cid, {})
        eff_mode = ov.get("method", ov.get("mode", args.mode))  # spec uses "method"; accept "mode" alias
        eff_layers = [int(x) for x in ov.get("layers", layers)]
        eff_mult = float(ov.get("multiplier", multiplier))
        eff_agg = eff_mode.replace("caa_", "") if eff_mode.startswith("caa_") else "mean"
        eff_is_token = eff_mode in ("token", "token_logit")
        eff_is_logit = eff_mode == "token_logit"
        # Anti-repetition n-gram ban for CAA generation (kills emoji/keyword/list
        # loops that appear at both low and high m). 0 disables. Not applied to
        # token_logit: its done-branch n-gram ban handles post-phrase loops, and a
        # global ban here would block the 2nd forced occurrence.
        eff_ngram = int(ov.get("no_repeat_ngram", 6))
        # L3 phrase-insertion delay (token_logit only): let the model emit a
        # natural preamble before forcing the phrase, so opener-awkward phrases
        # ("variance is the point") embed mid-response instead of at the start.
        eff_delay = int(ov.get("delay_tokens", 0))

        concept_str = train_concept[0].get("concept", "")
        target = None
        if eff_is_token:
            from src.caa.token_steer import extract_target_phrase
            target = extract_target_phrase(concept_str, model.tokenizer)
            if target is None:
                print(f"[fallback] {cid}: no phrase in concept → caa_diverge")
                eff_mode, eff_agg, eff_is_token = "caa_diverge", "diverge", False
                eff_is_logit = False

        # ---- CAA: ensure vectors exist for eff_layers.
        if not eff_is_token and args.generate_vector.lower() == "true":
            gen_hparams.layers = eff_layers
            gen_hparams.agg = eff_agg
            gen_hparams.normalize = bool(ov.get("normalize", True))  # raw for behavioral, normalized otherwise
            vec_dir = vector_concept_dir(model_short, dataset_name, args.gen_out_path, cid)
            vec_files = [vec_dir / "caa_vector" / f"layer_{l}.pt" for l in eff_layers]
            if not all(p.exists() for p in vec_files):
                vec_dir.mkdir(parents=True, exist_ok=True)
                (vec_dir / "caa_vector").mkdir(parents=True, exist_ok=True)
                print(f"[vec] {cid}: train rows={len(train_concept)} agg={eff_agg} normalize={gen_hparams.normalize} layers={eff_layers}")
                gen_hparams.steer_vector_output_dir = str(vec_dir)
                generate_caa_vectors(gen_hparams, model, train_concept, dataset_name=dataset_name + "_concept_" + cid)

        if args.generate_response.lower() != "true":
            continue

        if cid in done_concepts:
            print(f"[resume] {cid} already done, skip")
            continue

        valid_concept = valid_by_concept[cid]
        generation_params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0,
            "do_sample": False,
            "pad_token_id": model.tokenizer.eos_token_id,
        }
        concept_result = {
            "concept_id": cid,
            "concept_name": concept_str,
            "concept": concept_str,
            "concept_description": train_concept[0].get("llm_description", "") or train_concept[0].get("concept_description", ""),
            "generated_results": [],
        }

        for item in tqdm(valid_concept, desc=f"concept {cid}"):
            try:
                model.reset_all()
                if eff_is_logit:
                    # Clean L3: direct logit bias via LogitsProcessor. No residual hooks.
                    from src.caa.token_steer import LogitBiasProcessor
                    processor = LogitBiasProcessor(
                        target.token_ids, target.count, bias=eff_mult,
                        tokenizer=model.tokenizer, phrase_str=target.phrase,
                        delay_tokens=eff_delay,
                    )
                    gen_res = generate_one(model, item, generation_params, use_chat_template=True, logits_processor=processor)
                elif eff_is_token:
                    from src.caa.token_steer import PhraseCountProcessor, apply_token_steer
                    installed = apply_token_steer(gen_hparams, model, target, eff_layers, eff_mult)
                    processor = PhraseCountProcessor(target, installed, model, model.tokenizer)
                    gen_res = generate_one(model, item, generation_params, use_chat_template=True, logits_processor=processor, from_pos="prompt_len")
                else:
                    ap_hparams = ApplyCAAHyperParams(
                        layers=eff_layers,
                        multipliers=[eff_mult] * len(eff_layers),
                        model_name_or_path=args.model_name_or_path,
                        device=args.device,
                        dtype=dtype_str,
                        use_cache=True,
                        use_chat_template=True,
                        system_prompt="",
                        enable_thinking=enable_thinking,
                        steer_vector_load_dir=str(vector_concept_dir(model_short, dataset_name, args.gen_out_path, cid) / "caa_vector"),
                    )
                    apply_caa(ap_hparams, model)
                    proc = None
                    if eff_ngram and eff_ngram > 1:
                        from transformers import NoRepeatNGramLogitsProcessor
                        proc = NoRepeatNGramLogitsProcessor(ngram_size=eff_ngram)
                    gen_res = generate_one(model, item, generation_params, use_chat_template=True, logits_processor=proc)
            except Exception as e:
                print(f"[err] {cid}: {e}")
                model.reset_all()
                continue
            concept_result["generated_results"].append({
                "input": item.get("question") or item.get("input", ""),
                "complete_output": gen_res["complete_output"],
                "pred": gen_res["pred"],
                "reference_response": None,
                "orig_pred": [],
            })
            _empty_cache()  # keep MPS allocator tidy; lost patch preserved here.

        existing.append(concept_result)
        if merge_mode:  # keep canonical concept order so single-concept edits don't reshuffle the file
            order_map = {c: i for i, c in enumerate(valid_by_concept.keys())}
            existing.sort(key=lambda c: order_map.get(c.get("concept_id"), 999))
        out_path = final_path if merge_mode else tmp_path
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[ok] {cid} ({eff_mode}) written to {out_path.name}")

        # Reset CAA hooks so the next concept starts clean.
        model.reset_all()
        # Aggressive reclaim between concepts: Python GC + MPS cache release.
        # Without this, the KV cache from the last generate() of the previous
        # concept stays referenced in the allocator's "used" pool, fragmenting
        # memory and slowing each successive concept's vector gen + generation.
        import gc as _gc
        _gc.collect()
        _empty_cache()

    # Final rename (only in full-run mode; merge mode writes final directly).
    if not merge_mode and tmp_path.exists():
        tmp_path.replace(final_path)
        print(f"[done] {final_path}")
    elif merge_mode:
        print(f"[done] merged into {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
