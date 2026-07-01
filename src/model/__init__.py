"""model package — Qwen3 + CAA hooks."""
from .wrapper import BaseModelWrapper, BlockOutputWrapper, QwenWrapper
from .intervention import ActivationAddition
from .utils import add_vector_from_position

__all__ = [
    "BaseModelWrapper",
    "BlockOutputWrapper",
    "QwenWrapper",
    "ActivationAddition",
    "add_vector_from_position",
]


def make_qwen_wrapper(hparams):
    """Convenience factory mirroring the upstream ``get_model(...)`` 2-tuple."""
    wrapper = QwenWrapper(
        dtype=hparams.dtype,
        use_chat=hparams.use_chat_template,
        device=hparams.device,
        model_name_or_path=hparams.model_name_or_path,
        use_cache=hparams.use_cache,
        hparams=hparams,
    )
    if wrapper.tokenizer.pad_token is None:
        wrapper.tokenizer.pad_token = wrapper.tokenizer.eos_token
        wrapper.tokenizer.pad_token_id = wrapper.tokenizer.eos_token_id
    return wrapper, wrapper.tokenizer
