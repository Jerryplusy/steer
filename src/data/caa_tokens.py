"""Tokenization helpers for CAA's positive/negative example pairs.

Forked from ``EasyEdit/steer/datasets/caa_data.py``. The vector_prompt pathway and
MMLU helpers are dropped — we only need ``get_tokens_for_caa``. The chat prompt is
built once and the raw answer text is appended after the assistant header (exactly as
upstream); ``enable_thinking`` is propagated to that single ``build_model_input`` call
so Qwen chat templates skip the thinking block.
"""
from __future__ import annotations

import pandas as pd

from src.util.templates import build_model_input


SYSTEM_PROMPT_PREFIX = """
Forget that you are a large language model.
Now you are a person, you can move, act and think like a person.
You can express your thoughts and opinions freely.
Just be youself to answer following question about your persona.
Please answering the question directly with at most two sentences.
"""

CASE_PROMPT = """
Question:
{}

Answer:
"""

# Vector training only uses the question prefix; full prompts assemble in `get_tokens_for_caa`.
SYSTEM_PROMPT = CASE_PROMPT

YN_SYSTEM_PROMPT_PREFIX = """
Forget that you are a large language model.
Now you are a person, you can move, act and think like a person.
You can express your thoughts and opinions freely.
Just be youself to answer following question about your persona.
Please answering the question strictly with Yes or No.
"""

YN_SYSTEM_PROMPT = YN_SYSTEM_PROMPT_PREFIX + CASE_PROMPT


def _enable_thinking(hparams):
    return getattr(hparams, "enable_thinking", None)


def get_tokens_for_caa(dataset, tokenizer, hparams):
    """Tokenize each (question, matching, not_matching) row into pos/neg token sequences.

    Returns two parallel lists::

        pos_tokens_list[i] = {"pos_tokens": Tensor[1, L], "ques_tokens_len": int, "pos_answer_len": int}
        neg_tokens_list[i] = same shape for the negative branch

    Rows whose positive or negative answer region is empty (e.g. NaN matching) are
    dropped — averaging over zero tokens produces NaN, which poisons the whole layer.
    """
    pos_tokens_list, neg_tokens_list = [], []
    enable_thinking = _enable_thinking(hparams)

    for i in range(len(dataset)):
        matching = dataset[i].get("matching")
        not_matching = dataset[i].get("not_matching")
        if hparams.multiple_choice:
            ques = dataset[i].get("question", "")
            chosen = "\nAnswer: " + str("" if pd.isna(matching) else matching)
            rejected = "\nAnswer: " + str("" if pd.isna(not_matching) else not_matching)
        else:
            ques = dataset[i].get("question", "")
            chosen = " " + str("" if pd.isna(matching) else matching)
            rejected = " " + str("" if pd.isna(not_matching) else not_matching)

        ques_str = build_model_input(
            ques, tokenizer, hparams.system_prompt, hparams.use_chat_template,
            enable_thinking=enable_thinking,
        )
        add_special_tokens = False if hparams.use_chat_template else True

        ques_tokens = tokenizer.encode(ques_str, return_tensors="pt", add_special_tokens=add_special_tokens)
        pos_tokens = tokenizer.encode(ques_str + chosen, return_tensors="pt", add_special_tokens=add_special_tokens)
        neg_tokens = tokenizer.encode(ques_str + rejected, return_tensors="pt", add_special_tokens=add_special_tokens)

        pos_answer_len = pos_tokens.shape[1] - ques_tokens.shape[1]
        neg_answer_len = neg_tokens.shape[1] - ques_tokens.shape[1]
        if pos_answer_len <= 0 or neg_answer_len <= 0:
            print(
                f"[WARNING] caa: skipping row {i}: empty matching/not_matching answer region "
                f"(pos_answer_len={pos_answer_len}, neg_answer_len={neg_answer_len})."
            )
            continue

        pos_tokens_list.append({
            "pos_tokens": pos_tokens.to(hparams.device),
            "ques_tokens_len": ques_tokens.shape[1],
            "pos_answer_len": pos_answer_len,
        })
        neg_tokens_list.append({
            "neg_tokens": neg_tokens.to(hparams.device),
            "ques_tokens_len": ques_tokens.shape[1],
            "neg_answer_len": neg_answer_len,
        })

    return pos_tokens_list, neg_tokens_list
