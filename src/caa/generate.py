"""Generate CAA steering vectors (one per layer) by averaging (pos - neg) activations.

Forked from ``EasyEdit/steer/vector_generators/caa/generate_caa_vectors.py`` with the
following changes:
- Imports cleaned up (no hparams_class import gymnastics, no dotenv, no main()).
- ``torch.mps.empty_cache()`` after pos/neg forward pairs to keep MPS memory from
  fragmenting within a single process (lost patch, now preserved).
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
                 device, dtype, enable_thinking).
        model: already-loaded Qwen wrapper (BaseModelWrapper subclass).
        dataset: list[dict] of train rows; each must have ``matching`` and
                 ``not_matching`` (and optionally ``question``).

    Writes one ``layer_<L>.pt`` per L to ``<steer_vector_output_dir>/caa_vector/``
    when ``save_vectors`` is True.
    """
    args = hparams
    tokenizer = model.tokenizer
    model.hparams = hparams  # so get_last_activations path matches training-time setup

    pos_activations = {layer: [] for layer in args.layers}
    neg_activations = {layer: [] for layer in args.layers}

    pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

    for p_dict, n_dict in tqdm(
        zip(pos_tokens_list, neg_tokens_list),
        total=len(pos_tokens_list),
        desc="Processing prompts",
    ):
        # Pos
        model.reset_all()
        model.get_logits(p_dict["pos_tokens"])
        for layer in args.layers:
            acts = model.get_last_activations(layer)
            if args.multiple_choice:
                acts = acts[0, -2, :].detach().cpu()
            else:
                acts = acts[0, p_dict["ques_tokens_len"]:, :].mean(0).detach().cpu()
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
                acts = acts[0, n_dict["ques_tokens_len"]:, :].mean(0).detach().cpu()
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
        if args.save_vectors:
            torch.save(vec, os.path.join(out_dir, f"layer_{layer}.pt"))
        vectors[f"layer_{layer}"] = vec

    return vectors
