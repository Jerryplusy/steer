"""L3 token-direction steering (Stage 2).

For L3 concepts (``Must include the phrase "X" ...``), a behavioral CAA vector can
shift tone but cannot force the model to emit a specific rare token ("sonder",
"mos maiorum"). The reliable lever is a **token-aligned direction** in the residual
stream: because Qwen3 has ``tie_word_embeddings=True``, the embedding row
``embed_tokens.weight[t]`` *is* the unembedding direction for token ``t``. Adding
``multiplier * normalize(embed[t])`` to the residual stream at late layers directly
raises ``logit[t]``.

We install an :class:`ActivationAddition` at the last few layers whose
``steering_vector`` is the (normalized) embedding of the **next phrase sub-token**.
A :class:`PhraseCountProcessor` (HF ``LogitsProcessor``) runs after each generation
step, advances a match pointer on the emitted token, and updates the installed
interventions' ``steering_vector`` to the next expected phrase sub-token — so the
full multi-sub-token phrase is forced sequentially. Once the phrase has appeared
the required number of times, the steering vector is zeroed so generation can
continue fluently.

This is a residual-stream activation intervention (no prompt) and reuses the
existing :class:`~src.model.intervention.ActivationAddition` hook — it only swaps
*which* direction is added at each step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import torch


# --------------------------------------------------------------------------- #
# Concept parsing
# --------------------------------------------------------------------------- #
_QUOTED_RE = re.compile(r'"([^"]+)"')


@dataclass
class PhraseTarget:
    phrase: str
    token_ids: List[int]
    count: int  # how many times the full phrase must appear
    kind: str   # exact_once | at_least_once | exactly_twice | symbol_once


def extract_target_phrase(concept: str, tokenizer) -> Optional[PhraseTarget]:
    """Parse an L3 concept string into a :class:`PhraseTarget`.

    Returns None for non-L3 (behavioral) concepts. Handles the phrasings seen in
    SteerEval personality: ``exact phrase "X" verbatim/once/twice``, ``word "X" at
    least once``, ``exact term "X" at least once``, ``exact Latin phrase "X" once``,
    ``symbol sequence "X" exactly once``.
    """
    if not concept:
        return None
    m = _QUOTED_RE.search(concept)
    if not m:
        return None
    phrase = m.group(1)
    if not phrase:
        return None

    cl = concept.lower()
    if "twice" in cl:
        count, kind = 2, "exactly_twice"
    elif "at least once" in cl:
        count, kind = 1, "at_least_once"
    elif "exactly once" in cl or "symbol sequence" in cl:
        count, kind = 1, "symbol_once"
    elif "verbatim" in cl or " once" in cl or "phrase" in cl or "term" in cl or "word" in cl:
        count, kind = 1, "exact_once"
    else:
        count, kind = 1, "exact_once"

    token_ids = tokenizer.encode(phrase, add_special_tokens=False)
    if not token_ids:
        return None
    return PhraseTarget(phrase=phrase, token_ids=token_ids, count=count, kind=kind)


# --------------------------------------------------------------------------- #
# Embedding access
# --------------------------------------------------------------------------- #
def get_embedding_matrix(model) -> torch.Tensor:
    """Return the tied embedding/unembedding matrix ``[vocab, hidden]`` of the HF model."""
    hf = model.model  # Qwen3ForCausalLM
    embeds = getattr(getattr(hf, "model", None), "embed_tokens", None)
    if embeds is None:
        # Fallback: lm_head (tied) or the base model's embedding.
        if getattr(hf, "lm_head", None) is not None:
            return hf.lm_head.weight
        raise AttributeError("Could not locate embed_tokens on the HF model.")
    return embeds.weight


def build_token_vector(model, token_id: int) -> torch.Tensor:
    """Raw embedding row for ``token_id`` — the optimal residual-space direction for
    boosting ``logit[token_id]``.

    We use the **raw** (not unit-normalized) embedding because the logit boost from
    adding ``α·v`` to the residual (post-RMSNorm, tied unembedding) is
    ``α·‖embed‖²/‖residual‖`` for ``v = embed`` vs ``α·‖embed‖/‖residual‖`` for the
    unit vector — i.e. raw is ~‖embed‖× (≈5×) more efficient per unit α. The
    multiplier then lives in a tractable range (~10-200) instead of ~400+.
    """
    W = get_embedding_matrix(model)  # [vocab, hidden]
    return W[token_id].detach().float()


# --------------------------------------------------------------------------- #
# Hook installation
# --------------------------------------------------------------------------- #
def apply_token_steer(hparams, model, target: PhraseTarget, layers: List[int], multiplier: float):
    """Install an :class:`ActivationAddition` at each late layer, primed with the
    first phrase sub-token's direction. Returns the list of installed interventions
    (so the LogitsProcessor can update their ``steering_vector`` per step).

    ``hparams`` is an :class:`ApplyCAAHyperParams`-like object; only ``device`` is used.
    """
    from src.caa.apply import reset_caa_layers
    from src.model.intervention import ActivationAddition

    reset_caa_layers(model, layers)
    device = hparams.device
    first_vec = build_token_vector(model, target.token_ids[0]).to(device)

    installed = []
    for layer in layers:
        intervention = ActivationAddition(steering_vector=first_vec, multiplier=multiplier)
        intervention = intervention.to(device)
        model.set_intervention(layer, intervention, "caa")
        installed.append((layer, intervention))

    # Steer all positions during generation (token injection must happen every step).
    model.set_from_position(None)
    return installed


def _set_intervention_vector(installed, vec: torch.Tensor) -> None:
    """Update every installed intervention's steering_vector to ``vec`` (same device/dtype)."""
    if not installed:
        return
    _, first = installed[0]
    vec = vec.to(first.steering_vector.device).to(first.steering_vector.dtype)
    for _, inst in installed:
        inst.steering_vector = vec


def zero_interventions(installed) -> None:
    """Stop steering without dismantling hooks (adds the zero vector → no-op)."""
    if not installed:
        return
    _, first = installed[0]
    zero = torch.zeros_like(first.steering_vector)
    for _, inst in installed:
        inst.steering_vector = zero


# --------------------------------------------------------------------------- #
# LogitsProcessor: sequential phrase-token steering
# --------------------------------------------------------------------------- #
class PhraseCountProcessor:
    """Advances a phrase-match pointer on each emitted token and updates the
    installed interventions' steering direction for the next forward pass.

    HF ``generate`` calls ``__call__`` after each forward, with ``input_ids`` ending
    in the just-sampled token. We use that token to advance (or reset) the pointer,
    then set the next-step steering vector to the embedding of the next expected
    phrase sub-token. When the phrase has appeared ``count`` times, we zero the
    steering so the rest of the response is fluent.
    """

    def __init__(self, target: PhraseTarget, installed, model, tokenizer):
        self.target = target
        self.installed = installed
        self.model = model
        self.tokenizer = tokenizer
        self.phrase_ids = target.token_ids
        self.required = target.count
        self.ptr = 0  # index into phrase_ids of the next expected token
        self.completed = 0  # full-phrase appearances so far
        self.done = False

    def __call__(self, input_ids, scores):
        # input_ids: [batch=1, seq]; the last column is the just-emitted token.
        last_id = int(input_ids[0, -1].item())
        expected = self.phrase_ids[self.ptr]
        if last_id == expected:
            self.ptr += 1
            if self.ptr >= len(self.phrase_ids):
                # Full phrase completed once.
                self.completed += 1
                self.ptr = 0
                if self.completed >= self.required:
                    self.done = True
                    zero_interventions(self.installed)
                    return scores
        else:
            # Mismatch: reset; if the emitted token happens to be phrase[0], start.
            self.ptr = 1 if last_id == self.phrase_ids[0] else 0

        if not self.done:
            next_id = self.phrase_ids[self.ptr]
            vec = build_token_vector(self.model, next_id)
            _set_intervention_vector(self.installed, vec)
        return scores


# --------------------------------------------------------------------------- #
# LogitsProcessor: direct logit bias (the clean L3 method)
# --------------------------------------------------------------------------- #
class LogitBiasProcessor:
    """Force the model to emit a target phrase by directly adding ``bias`` to the
    logit of the next expected phrase sub-token at each generation step.

    This is the **clean** alternative to residual-stream token injection: it does
    not touch the residual stream (no K/V cache pollution, no intermediate-layer
    corruption), so the rest of the response stays fluent. The bias is added at
    the logits — internal activations before sampling — which falls squarely in
    the "activation intervention" family (cf. ITI, constrained decoding).

    With a large enough ``bias`` (≈+20..+80), the targeted token dominates the
    argmax and the model emits it. The pointer advances on each match; once the
    full phrase has appeared ``count`` times, the processor stops biasing and
    the model continues normally.
    """

    def __init__(self, phrase_ids: List[int], count: int = 1, bias: float = 40.0,
                 anti_repeat: float = 5.0):
        self.phrase_ids = phrase_ids
        self.required = count
        self.bias = bias
        self.anti_repeat = anti_repeat  # small negative on phrase[0] after done to deter re-mention
        self.ptr = 0
        self.completed = 0
        self.done = False

    def __call__(self, input_ids, scores):
        if self.done:
            # Anti-repeat: gentle negative bias on every phrase sub-token so the
            # model doesn't keep treating the phrase as a topic after emission.
            scores[:, self.phrase_ids] -= self.anti_repeat
            return scores
        last_id = int(input_ids[0, -1].item())
        expected = self.phrase_ids[self.ptr]
        if last_id == expected:
            self.ptr += 1
            if self.ptr >= len(self.phrase_ids):
                self.completed += 1
                self.ptr = 0
                if self.completed >= self.required:
                    self.done = True
                    # Start anti-repeat immediately so the very next step won't
                    # re-emit a phrase sub-token via natural sampling.
                    scores[:, self.phrase_ids] -= self.anti_repeat
                    return scores
        else:
            self.ptr = 1 if last_id == self.phrase_ids[0] else 0
        # Bias the next expected phrase token. +bias on its logit dominates the argmax.
        scores[:, self.phrase_ids[self.ptr]] += self.bias
        return scores
