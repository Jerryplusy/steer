"""util package."""
from .seed import set_seed
from .templates import build_model_input, safe_apply_chat_template

__all__ = ["set_seed", "build_model_input", "safe_apply_chat_template"]
