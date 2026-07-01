"""Activation-addition intervention (CAA's vector hook)."""
from __future__ import annotations

from typing import Optional

import torch


class BaseIntervention(torch.nn.Module):
    """Base class for steering interventions (kept tiny — we only need Addition)."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.trainable = False
        self.is_source_constant = False
        self.keep_last_dim = kwargs.get("keep_last_dim", False)
        self.use_fast = kwargs.get("use_fast", False)
        self.subspace_partition = kwargs.get("subspace_partition", None)
        self.embed_dim = None
        self.interchange_dim = None
        self.source_representation = None


class ActivationAddition(BaseIntervention):
    """Add ``multiplier * steering_vector`` to the residual stream at every layer forward."""

    def __init__(self, steering_vector: torch.Tensor, multiplier: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.steering_vector = steering_vector
        self.multiplier = multiplier

    def forward(self, base, **kwargs):
        from .utils import add_vector_from_position

        steering_addition = self.multiplier * self.steering_vector
        position_ids = kwargs.get("position_ids", None)
        from_pos = kwargs.get("from_pos", None)
        if position_ids is not None or from_pos is not None:
            return add_vector_from_position(
                matrix=base,
                vector=steering_addition,
                position_ids=position_ids,
                from_pos=from_pos,
            )
        return base + steering_addition.unsqueeze(0).unsqueeze(0)

    def to(self, device):  # type: ignore[override]
        super().to(device)
        if isinstance(self.steering_vector, torch.Tensor):
            self.steering_vector = self.steering_vector.to(device)
        return self
