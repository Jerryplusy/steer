"""Chat-template helpers. Only the text path is needed (no multimodal)."""
from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer

_WARNED_NO_TEMPLATE: set[str] = set()


def _has_chat_template(obj) -> bool:
    return getattr(obj, "chat_template", None) is not None


def _warn_no_template_once(obj) -> None:
    name = getattr(obj, "name_or_path", None) or str(type(obj))
    if name not in _WARNED_NO_TEMPLATE:
        _WARNED_NO_TEMPLATE.add(name)
        print(
            f"[WARNING] use_chat_template=True but '{name}' has no chat_template; "
            "falling back to raw text concatenation (base-model style)."
        )


def safe_apply_chat_template(tokenizer, messages, system_prompt: Optional[str] = None, **kw):
    """Apply chat template with optional system prompt, robustly (gemma has no system role)."""
    if system_prompt:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, *messages], **kw
            )
        except Exception:
            pass
    msgs = [dict(m) for m in messages]
    if system_prompt and msgs and msgs[0].get("role") == "user":
        msgs[0]["content"] = f"{system_prompt} {msgs[0]['content']}"
    return tokenizer.apply_chat_template(msgs, **kw)


def build_model_input(
    user_input: str,
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None,
    use_chat_template: Optional[bool] = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Build a prompt string for an instruct-tuned LM.

    - When ``use_chat_template`` is True and the tokenizer has a chat_template, we
      apply it. We forward ``enable_thinking`` to ``apply_chat_template`` so Qwen-style
      instruct models can be told to skip the thinking block.
    - Otherwise we fall back to a plain-text concatenation.
    """
    user_input = (user_input or "").strip()
    if model_output:
        model_output = model_output.strip()
    if suffix:
        suffix = suffix.strip()

    if not use_chat_template or not _has_chat_template(tokenizer):
        if use_chat_template and not _has_chat_template(tokenizer):
            _warn_no_template_once(tokenizer)
        user_content = ""
        if system_prompt:
            user_content = f"{system_prompt} "
        user_content += user_input
        if suffix:
            user_content += f" {suffix}"
        if model_output:
            user_content += f" {model_output}"
        return user_content

    input_content = user_input
    if suffix:
        input_content += f" {suffix}"

    messages = [{"role": "user", "content": input_content}]
    if model_output is not None:
        messages.append({"role": "assistant", "content": model_output})

    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking
    return safe_apply_chat_template(
        tokenizer, messages, system_prompt=system_prompt, **chat_kwargs
    )
