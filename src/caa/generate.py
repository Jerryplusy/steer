"""Generate CAA steering vectors (one per layer) from (pos - neg) activations.

Forked from ``EasyEdit/steer/vector_generators/caa/generate_caa_vectors.py`` with the
following changes:
- Imports cleaned up (no hparams_class import gymnastics, no dotenv, no main()).
- ``torch.mps.empty_cache()`` after pos/neg forward pairs to keep MPS memory from
  fragmenting within a single process (lost patch, now preserved).
- Stage 1: three aggregation modes (``agg``) — ``mean`` (legacy, over the full
  answer), ``last`` (mean of the last few answer tokens, un-diluted summary), and
  ``diverge`` (mean over the token window starting at the first pos/neg divergence —
  the only positions that actually differ in a minimal-edit pair, so the only ones
  contributing a non-zero ``(pos - neg)``). ``diverge`` is the default for this
  dataset because SteerEval matching/not_matching are minimal-edit pairs: identical
  answer tokens contribute zero to the contrast and dilute the ``mean`` vector by
  ~k/L.
- Stage 1: optional unit-L2 normalization before saving, so the applied multiplier
  is comparable across concepts (L3_1 had norm ≈8 vs L3_3 ≈43 under ``mean``).
"""
from __future__ import annotations

import os
from typing import Dict

import torch
from tqdm import tqdm

from src.data.caa_tokens import get_tokens_for_caa


def _empty_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _find_divergence(pos_ids: torch.Tensor, neg_ids: torch.Tensor, ques_len: int) -> int:
    """First index ``>= ques_len`` where pos and neg token ids differ.

    Pos and neg share the question prefix (and possibly some answer tokens); causal
    attention makes activations identical on the shared prefix, so the contrastive
    signal only starts at the first divergent token.
    """
    n = min(pos_ids.shape[1], neg_ids.shape[1])
    d = ques_len
    while d < n and pos_ids[0, d] == neg_ids[0, d]:
        d += 1
    return d


def _extract_vec(
    acts: torch.Tensor,
    ques_len: int,
    answer_len: int,
    agg: str,
    win: int,
    diverge_d: int,
) -> torch.Tensor:
    """Reduce ``acts`` (a layer's [1, seq, hidden] activation) to a single [hidden] vector.

    - ``mean``: mean over the full answer region (legacy CAA).
    - ``last``: mean over the last ``min(3, answer_len)`` answer tokens — an un-diluted
      summary of the answer's stance, which for minimal-edit pairs avoids the 1/L
      dilution of ``mean``.
    - ``diverge``: mean over ``[diverge_d, diverge_d + win)`` — only the divergent
      tokens contribute a non-zero ``(pos - neg)``.
    """
    seq = acts.shape[1]
    if agg == "last":
        k = min(3, max(answer_len, 1))
        start = max(ques_len + answer_len - k, ques_len)
        end = min(ques_len + answer_len, seq)
        if end <= start:
            return acts[0, -1, :]
        return acts[0, start:end].mean(0)
    if agg == "diverge":
        end = min(diverge_d + win, seq)
        if end <= diverge_d:
            return acts[0, -1, :]
        return acts[0, diverge_d:end].mean(0)
    # default: mean over answer
    end = min(ques_len + answer_len, seq)
    return acts[0, ques_len:end].mean(0)


def generate_caa_vectors(
    hparams,
    model,
    dataset,
    dataset_name: str = "steer_eval",
) -> Dict[str, torch.Tensor]:
    """Compute (pos - neg) activation vectors for each layer in ``hparams.layers``.

    Args:
        hparams: CAAHyperParams (must carry layers, multiple_choice, save_vectors,
                 steer_vector_output_dir, system_prompt, use_chat_template,
                 device, dtype, enable_thinking, agg, normalize, diverge_win).
        model: already-loaded Qwen wrapper (BaseModelWrapper subclass).
        dataset: list[dict] of train rows; each must have ``matching`` and
                 ``not_matching`` (and optionally ``question``).

    Writes one ``layer_<L>.pt`` per L to ``<steer_vector_output_dir>/caa_vector/``
    when ``save_vectors`` is True.
    """
    args = hparams
    tokenizer = model.tokenizer
    model.hparams = hparams  # so get_last_activations path matches training-time setup

    agg = getattr(args, "agg", "mean")
    normalize = getattr(args, "normalize", True)
    diverge_win = getattr(args, "diverge_win", 8)

    pos_activations = {layer: [] for layer in args.layers}
    neg_activations = {layer: [] for layer in args.layers}

    pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

    for p_dict, n_dict in tqdm(
        zip(pos_tokens_list, neg_tokens_list),
        total=len(pos_tokens_list),
        desc="Processing prompts",
    ):
        # Divergence point (shared between pos and neg for this row).
        diverge_d = _find_divergence(p_dict["pos_tokens"], n_dict["neg_tokens"], p_dict["ques_tokens_len"])

        # Pos
        model.reset_all()
        model.get_logits(p_dict["pos_tokens"])
        for layer in args.layers:
            acts = model.get_last_activations(layer)
            if args.multiple_choice:
                acts = acts[0, -2, :].detach().cpu()
            else:
                acts = _extract_vec(
                    acts.detach(),
                    p_dict["ques_tokens_len"], p_dict["pos_answer_len"],
                    agg, diverge_win, diverge_d,
                ).cpu()
            pos_activations[layer].append(acts)

        model.reset_all()
        _empty_cache()  # MPS doesn't release between pos and neg forwards without this.

        # Neg
        model.get_logits(n_dict["neg_tokens"])
        for layer in args.layers:
            acts = model.get_last_activations(layer)
            if args.multiple_choice:
                acts = acts[0, -2, :].detach().cpu()
            else:
                acts = _extract_vec(
                    acts.detach(),
                    n_dict["ques_tokens_len"], n_dict["neg_answer_len"],
                    agg, diverge_win, diverge_d,
                ).cpu()
            neg_activations[layer].append(acts)

        # Empty once per row keeps memory pool stable across the 70-row train set.
        _empty_cache()

    out_dir = os.path.join(args.steer_vector_output_dir, "caa_vector")
    if args.save_vectors:
        os.makedirs(out_dir, exist_ok=True)

    vectors: Dict[str, torch.Tensor] = {}
    for layer in args.layers:
        all_pos = torch.stack(pos_activations[layer])
        all_neg = torch.stack(neg_activations[layer])
        vec = (all_pos - all_neg).mean(dim=0)
        if normalize:
            nrm = vec.norm().clamp_min(1e-8)
            vec = vec / nrm
        if args.save_vectors:
            torch.save(vec, os.path.join(out_dir, f"layer_{layer}.pt"))
        vectors[f"layer_{layer}"] = vec

    return vectors
