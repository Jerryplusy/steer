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


def generation_dir(model: str, dataset: str, run: str, method: str, layer: int, multiplier: float) -> Path:
    return (
        PROJECT_ROOT / "output" / "generation" / model / dataset / run
        / method / f"layer_{layer}_multip_{multiplier}"
    )


def logs_dir(model: str, dataset: str, run: str, method: str, layer: int, multiplier: float) -> Path:
    return (
        PROJECT_ROOT / "output" / "logs" / model / dataset / run
        / method / f"layer_{layer}_multip_{multiplier}.log"
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


def generate_one(model, item: dict, generation_params: dict, use_chat_template: bool):
    """Run the model on a single ``item`` and return the steered ``pred`` string."""
    question = item.get("question") or item.get("input") or ""
    if not question.strip():
        raise ValueError(f"empty prompt for item keys={sorted(item.keys())}")
    prompt = build_model_input(
        question, model.tokenizer,
        system_prompt="", use_chat_template=use_chat_template,
        enable_thinking=getattr(model.hparams, "enable_thinking", None),
    )
    inputs = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=not use_chat_template).to(model.device)
    with torch.no_grad():
        output = model.model.generate(**inputs, **generation_params)
    full_output = model.tokenizer.decode(output[0], skip_special_tokens=False)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"complete_output": [full_output], "pred": [text]}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Local CAA driver for Qwen3.")
    parser.add_argument("--model_name_or_path", default=str(PROJECT_ROOT / "qwen3-4b"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--method", default="caa")
    parser.add_argument("--dataset", default="SteerEval/personality")
    parser.add_argument("--generate_vector", default="true")
    parser.add_argument("--gen_out_path", default="baseline_v1")
    parser.add_argument("--generate_response", default="true")
    parser.add_argument("--generate_orig_output", default="false")
    parser.add_argument("--evaluate", default="false", help="Ignored; shell/score.py is used.")
    parser.add_argument("--layers", default="20")
    parser.add_argument("--multipliers", default="3")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--exp", default="valid")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.method != "caa":
        print(f"[error] only 'caa' is supported in this driver, got {args.method!r}", file=sys.stderr)
        return 1

    layer = int(args.layers)
    multiplier = float(args.multipliers)

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
        layers=[layer],
        multiple_choice=False,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=dtype_str,
        use_cache=True,
        use_chat_template=True,
        system_prompt="",
        save_activations=True,
        enable_thinking=enable_thinking,
    )
    model, _tokenizer = make_qwen_wrapper(gen_hparams)
    model.hparams = gen_hparams
    model.eval()

    # Path conventions for this (run, layer, multiplier).
    vec_root = vector_dir(model_short, dataset_name, args.gen_out_path)
    gen_root = generation_dir(model_short, dataset_name, args.gen_out_path, args.method, layer, multiplier)
    gen_root.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "logs" / model_short / dataset_name / args.gen_out_path / args.method).mkdir(parents=True, exist_ok=True)

    # Per-concept iteration.
    concept_ids = list(valid_by_concept.keys())
    print(f"Will process {len(concept_ids)} concepts")

    for cid in concept_ids:
        train_concept = train_by_concept.get(cid, [])
        if not train_concept:
            print(f"[skip] no train data for {cid}")
            continue

        # ---- Vector.
        vec_path = vector_concept_dir(model_short, dataset_name, args.gen_out_path, cid) / "caa_vector" / f"layer_{layer}.pt"
        if args.generate_vector.lower() == "true":
            vec_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[vec] {cid}: train rows={len(train_concept)} -> {vec_path}")
            gen_hparams.steer_vector_output_dir = str(vector_concept_dir(model_short, dataset_name, args.gen_out_path, cid))
            generate_caa_vectors(gen_hparams, model, train_concept, dataset_name=dataset_name + "_concept_" + cid)

        # ---- Apply.
        ap_hparams = ApplyCAAHyperParams(
            layers=[layer],
            multipliers=[multiplier],
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            dtype=dtype_str,
            use_cache=True,
            use_chat_template=True,
            system_prompt="",
            enable_thinking=enable_thinking,
            steer_vector_load_dir=str(vector_concept_dir(model_short, dataset_name, args.gen_out_path, cid) / "caa_vector"),
        )
        model = apply_caa(ap_hparams, model)

        # ---- Generate response.
        if args.generate_response.lower() != "true":
            continue

        results_path = gen_root / f"tmp_generation_results_{args.exp}.json"
        # Load partial file if resume.
        existing: List[dict] = []
        if results_path.exists():
            try:
                existing = load_json(results_path)
                done_concepts = {c.get("concept_id") for c in existing}
            except Exception:
                done_concepts = set()
                existing = []
        else:
            done_concepts = set()

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
        # We need both steered and possibly orig per item; here we follow the shell
        # default of --generate_orig_output=false, so only steered.
        concept_result = {
            "concept_id": cid,
            "concept": train_concept[0].get("concept", ""),
            "concept_description": train_concept[0].get("llm_description", "") or train_concept[0].get("concept_description", ""),
            "generated_results": [],
        }
        for item in tqdm(valid_concept, desc=f"concept {cid}"):
            try:
                gen_res = generate_one(model, item, generation_params, use_chat_template=True)
            except Exception as e:
                print(f"[err] {cid}: {e}")
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
        # Persist after each concept for crash resilience.
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[ok] {cid} written to {results_path.name}")

        # Reset CAA hooks so the next concept starts clean.
        model.reset_all()

    # Final rename.
    tmp_path = gen_root / f"tmp_generation_results_{args.exp}.json"
    final_path = gen_root / f"all_generation_results_{args.exp}.json"
    if tmp_path.exists():
        tmp_path.replace(final_path)
        print(f"[done] {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
