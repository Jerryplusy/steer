"""Model wrapper for Qwen3 + CAA vector hooks.

Forked from ``EasyEdit/steer/models/model_wrapper.py``. Only the Qwen branch and the
CAA-relevant state are kept. GPT, Gemma, Llama wrappers and the lm_steer pathway are
removed. The ``ori_generate`` save/clear/restore fix for ``intervention_dict`` (which
made CAA baselines genuinely unsteered) is kept here, since the corresponding EasyEdit
patch was lost on disk even though ``git diff EasyEdit/`` was empty (the directory is
untracked).
"""
from __future__ import annotations

import copy
import os
from typing import List, Optional

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from .intervention import ActivationAddition  # noqa: F401  (registered so set_intervention can store it)
from .utils import add_vector_from_position


def _qwen_layers(model) -> t.nn.ModuleList:
    """Return the decoder ModuleList for a Qwen (Qwen2 / Qwen3) HF causal LM."""
    base = model
    for attr in ("model", "language_model"):
        nxt = getattr(base, attr, None)
        if nxt is not None:
            base = nxt
            break
    if not hasattr(base, "layers"):
        raise AttributeError(
            f"Could not locate .layers under {type(model).__name__}; "
            "QwenWrapper assumes a standard CausalLM-style layout."
        )
    return base.layers


# vLLM is unused on macOS, but keep the env so accidental imports don't explode.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

try:
    from vllm import LLM  # noqa: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# ----------------------------------------------------------------------------
# Layer wrappers
# ----------------------------------------------------------------------------
class AttnWrapper(t.nn.Module):
    """Wrap attention so we can read its activations after the call."""

    def __init__(self, attn: t.nn.Module) -> None:
        super().__init__()
        self.attn = attn
        self.input_activations = None
        self.activations = None
        self.add_attn_activations = None
        self.attn_no_grad = False

    def set_no_grad(self) -> None:
        self.attn_no_grad = True

    def forward(self, *args, **kwargs):
        if args:
            self.input_activations = args[0]
        if self.attn_no_grad:
            with t.no_grad():
                output = self.attn(*args, **kwargs)
        else:
            output = self.attn(*args, **kwargs)
        self.activations = output[0] if isinstance(output, tuple) else output
        return output

    def add(self, activations) -> None:
        self.add_attn_activations = activations


class MLPWrapper(t.nn.Module):
    """Wrap MLP so we can read its mid-activation (between gate/up and down_proj)."""

    def __init__(self, mlp: t.nn.Module) -> None:
        super().__init__()
        self.mlp = mlp
        self.input_activations = None
        self.mid_activations = None
        self.activations = None
        self.add_mlp_activations = None
        self.mlp_no_grad = False

    def set_no_grad(self) -> None:
        self.mlp_no_grad = True

    def forward(self, *args, **kwargs):
        if args:
            self.input_activations = args[0]
        if self.mlp_no_grad:
            with t.no_grad():
                output = self.mlp(*args, **kwargs)
        else:
            output = self.mlp(*args, **kwargs)
        self.activations = output[0] if isinstance(output, tuple) else output

        hidden = args[0]
        if hasattr(self.mlp, "gate_proj") and hasattr(self.mlp, "up_proj"):
            self.mid_activations = self.mlp.act_fn(self.mlp.gate_proj(hidden)) * self.mlp.up_proj(hidden)
        elif hasattr(self.mlp, "gate_up_proj"):
            # For Qwen3 family, vllm combines gate and up into one matrix.
            gate_up = self.mlp.gate_up_proj(hidden)
            if isinstance(gate_up, tuple):
                gate_up = gate_up[0]
            self.mid_activations = self.mlp.act_fn(gate_up)
        else:
            self.mid_activations = None
        return output

    def add(self, activations) -> None:
        self.add_mlp_activations = activations


class BlockOutputWrapper(t.nn.Module):
    """Wrap one transformer block. Holds CAA intervention state.

    ``forward`` runs the block, captures activations into ``self.activations``, and
    applies ``add_activations_dict`` (legacy ``add_activations`` pathway) and
    ``intervention_dict`` (CAA) on top of the original output.
    """

    def __init__(self, block, layer_id: int, model_name_or_path: str) -> None:
        super().__init__()
        self.block = block
        self.layer_id = layer_id
        self.model_type = "qwen" if "qwen" in model_name_or_path.lower() else "llama"

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.block.mlp = MLPWrapper(self.block.mlp)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.save_activations = True
        self.activations = None
        self.add_activations_dict = {}  # legacy ``add`` activations pathway
        self.intervention_dict = {}  # CAA / RePS interventions

        self.from_position = None
        self.save_internal_decodings = False
        self.calc_dot_product_with = None
        self.dot_products = []

    def __getattr__(self, name):  # type: ignore[override]
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        original_is_tuple = isinstance(output, tuple)
        if not original_is_tuple:
            output = (output,)

        if self.save_activations:
            self.activations = output[0]
            if self.activations.dim() == 2:
                self.activations = self.activations.unsqueeze(0)

        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            top_token_id = t.topk(last_token_activations, 1)[1][0]
            self.dot_products.append(("?", 0.0))

        if self.add_activations_dict:
            augmented_output = output[0]
            for activations in self.add_activations_dict.values():
                if activations is not None:
                    position_ids = kwargs.get("position_ids", None)
                    augmented_output = add_vector_from_position(
                        matrix=augmented_output,
                        vector=activations,
                        position_ids=position_ids,
                        from_pos=self.from_position,
                    )
            output = (augmented_output,) + output[1:]

        if self.intervention_dict:
            augmented_output = output[0]
            for method_name, intervention in self.intervention_dict.items():
                if intervention is None:
                    continue
                intervention_result = intervention.forward(
                    augmented_output, from_pos=self.from_position, **kwargs
                )
                if hasattr(intervention_result, "output"):
                    augmented_output = intervention_result.output
                else:
                    augmented_output = intervention_result
            output = (augmented_output,) + output[1:]

        return output if original_is_tuple else output[0]

    def add(self, activations, method_name: str = "default") -> None:
        self.add_activations_dict[method_name] = activations

    def set_intervention(self, intervention, method_name: str) -> None:
        self.intervention_dict[method_name] = intervention

    def reset(self, method_name: str = "all") -> None:
        if method_name == "all":
            self.add_activations_dict.clear()
            self.intervention_dict.clear()
        else:
            self.add_activations_dict.pop(method_name, None)
            self.intervention_dict.pop(method_name, None)
        self.activations = None
        if self.model_type == "gpt":
            self.block.attn.activations = None
        else:
            self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []

    def set_save_activations(self, _value: bool) -> None:
        pass


# ----------------------------------------------------------------------------
# Base wrapper
# ----------------------------------------------------------------------------
class BaseModelWrapper:
    """Common wrapper for a HuggingFace causal LM with CAA hooks installed."""

    def __init__(
        self,
        dtype: t.dtype = t.float32,
        use_chat: bool = False,
        device: str = "mps" if t.backends.mps.is_available() else ("cuda" if t.cuda.is_available() else "cpu"),
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams: Optional["object"] = None,
    ) -> None:
        self.hparams = hparams
        self.use_chat = use_chat
        self.device = device
        self.dtype = self._resolve_dtype(dtype)
        self.use_cache = use_cache
        self.model_name_or_path = model_name_or_path
        self.processor = None
        self.VLLM_model = None  # always None in this port (no vLLM on macOS)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side="right" if "gemma" in self.model_name_or_path else "left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model
        self.model = self._load_hf_model()
        if override_model_weights_path is not None:
            self.model.load_state_dict(
                t.load(override_model_weights_path, map_location=self.device),
                strict=False,
            )

        self._adapt_model_layers()
        self.model.eval()

    @staticmethod
    def _resolve_dtype(dtype) -> t.dtype:
        if isinstance(dtype, t.dtype):
            return dtype
        mapping = {"fp16": t.float16, "float16": t.float16, "bf16": t.bfloat16, "bfloat16": t.bfloat16}
        return mapping.get(dtype, t.float32)

    def _load_hf_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=self.dtype,
            device_map=self.device,
            use_cache=self.use_cache,
        )

    def _decoder_layers(self) -> t.nn.ModuleList:
        return _qwen_layers(self.model)

    def _adapt_model_layers(self) -> None:
        """Install BlockOutputWrapper on every decoder layer (Qwen-style)."""
        layers = self._decoder_layers()
        for i, layer in enumerate(layers):
            layers[i] = BlockOutputWrapper(layer, i, self.model_name_or_path or "")
        self.set_save_activations(getattr(self.hparams, "save_activations", True))

    # ---- Activation helpers ----
    def get_last_activations(self, layer: int) -> t.Tensor:
        return self._decoder_layers()[layer].activations

    def set_add_activations(self, layer: int, activations, method_name: str = "default") -> None:
        self._decoder_layers()[layer].add(activations, method_name)

    def set_intervention(self, layer: int, intervention, method_name: str) -> None:
        self._decoder_layers()[layer].set_intervention(intervention, method_name)

    def set_save_activations(self, value: bool) -> None:
        for layer in self._decoder_layers():
            layer.save_activations = value

    def reset_all(self) -> None:
        for layer in self._decoder_layers():
            layer.reset(method_name="all")
        for attr in ("prompt", "generate_prompts", "steer"):
            if hasattr(self, attr):
                delattr(self, attr)

    def reset(self, method_name: str) -> None:
        method_name = method_name.lower()
        if method_name == "caa":
            for layer in self._decoder_layers():
                layer.reset(method_name="caa")
        elif method_name == "prompt":
            if hasattr(self, "prompt"):
                delattr(self, "prompt")
        else:
            raise ValueError(f"Method {method_name} not supported to reset")

    # ---- Forward helpers ----
    def get_logits(self, tokens) -> t.Tensor:
        return self.model(tokens).logits

    def eval(self) -> "BaseModelWrapper":
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "BaseModelWrapper":
        self.model.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def ori_generate(self, input_ids, **kwargs):
        """Generate without any steering hook active.

        CAA's ``intervention_dict`` is saved, cleared, and restored around the call so
        methods like RePS / CAA that set intervention state do not leak into the
        baseline. This is the upstream fix that was lost when the EasyEdit patch
        evaporated.
        """
        saved_interventions = {}
        model_layers = self._decoder_layers()
        for i, layer in enumerate(model_layers):
            if hasattr(layer, "intervention_dict") and layer.intervention_dict:
                saved_interventions[i] = dict(layer.intervention_dict)
                layer.intervention_dict = {}

        try:
            return self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            for i, intervention_dict in saved_interventions.items():
                model_layers[i].intervention_dict = intervention_dict


# ----------------------------------------------------------------------------
# Qwen wrapper
# ----------------------------------------------------------------------------
class QwenWrapper(BaseModelWrapper):
    """No Qwen-specific overrides — the model_type branch in BlockOutputWrapper
    covers it. Kept as a separate class so external callers can pattern-match on it."""

    pass
