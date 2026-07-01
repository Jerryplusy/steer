"""CAA hyperparameters (combined base + train + apply dataclasses)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HyperParams:
    """Common options for both vector generation and application."""

    use_chat_template: bool = True
    system_prompt: str = ""
    dtype: str = "float16"
    seed: int = 42
    model_name_or_path: str = "your_own_path"
    device: str = "mps"
    use_cache: bool = True
    generate_orig_output: bool = False
    vllm_enable: bool = False
    save_activations: bool = True
    enable_thinking: Optional[bool] = None


@dataclass
class CAAHyperParams(HyperParams):
    """Vector-generation hyperparameters (one set per run)."""

    alg_name: str = "caa"
    layers: List[int] = field(default_factory=lambda: [26])
    steer_train_dataset: str = "steer_eval"
    multiple_choice: bool = False
    steer_vector_output_dir: str = "../"
    save_vectors: bool = True


@dataclass
class ApplyCAAHyperParams(HyperParams):
    """Vector-application hyperparameters (one set per (L, M) group)."""

    alg_name: str = "caa"
    layers: List[int] = field(default_factory=lambda: [26])
    multipliers: List[float] = field(default_factory=lambda: [2.5])
    steer_vector_load_dir: Optional[str] = None
