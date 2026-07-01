"""Model-side utilities (small enough to inline)."""
from __future__ import annotations

import torch


def add_vector_from_position(matrix, vector, position_ids, from_pos=None) -> torch.Tensor:
    """Add ``vector`` to ``matrix`` either everywhere or starting from a position.

    Used by CAA's ``ActivationAddition``. Kept identical to the upstream version in
    semantics; only docstring trimmed.
    """
    orig_dtype = matrix.dtype

    if position_ids is None:
        mask = torch.ones_like(matrix, dtype=orig_dtype)
    else:
        from_id = from_pos
        if from_id is None:
            from_id = position_ids.min().item() - 1

        mask = position_ids >= from_id  # [seq]
        mask = mask.unsqueeze(-1)  # [seq, 1]

    matrix = matrix.float() + mask.float() * vector.float()
    return matrix.to(orig_dtype)
