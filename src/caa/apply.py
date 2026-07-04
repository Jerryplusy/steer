"""Attach CAA vector interventions to model layers.

Forked from ``EasyEdit/steer/vector_appliers/caa/apply_caa.py``. The ``ori_generate``
save/clear/restore fix lives in ``src.model.wrapper.BlockOutputWrapper`` and
``BaseModelWrapper.ori_generate``; this module only configures the interventions.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from src.model.intervention import ActivationAddition


def reset_caa_layers(model, layers) -> None:
    """Clear any CAA activations and interventions on the given layer indices."""
    for layer_idx in layers:
        model._decoder_layers()[layer_idx].reset(method_name="caa")


def apply_caa(
    hparams,
    model,
    vector: Optional[dict] = None,
) -> object:
    """Install CAA intervention hooks on ``model`` for every layer in
    ``hparams.layers``, with the per-layer multipliers in ``hparams.multipliers``.

    Args:
        hparams: ApplyCAAHyperParams
        model: already-loaded Qwen wrapper
        vector: optional in-memory vector dict (matches generate.output shape).
                If None, vectors are loaded from ``hparams.steer_vector_load_dir``.
    """
    print(f"Apply CAA to model: {hparams.model_name_or_path}")
    reset_caa_layers(model, hparams.layers)

    layers = hparams.layers
    multipliers = hparams.multipliers
    for layer, multiplier in zip(layers, multipliers):
        if vector is not None:
            steering_vector = vector[f"layer_{layer}"].to(hparams.device)
            print(f"Steering vector: User input vector for layer_{layer}")
        else:
            vector_path = os.path.join(hparams.steer_vector_load_dir, f"layer_{layer}.pt")
            steering_vector = torch.load(vector_path, map_location=hparams.device)
            print(f"Steering vector path: {vector_path}")

        intervention = ActivationAddition(
            steering_vector=steering_vector,
            multiplier=multiplier,
        )
        intervention = intervention.to(hparams.device)
        model.set_intervention(layer, intervention, "caa")

    # Restrict activation-addition to positions >= from_pos (e.g. skip the prompt).
    from_pos = getattr(hparams, "from_pos", None)
    model.set_from_position(from_pos)
    return model
