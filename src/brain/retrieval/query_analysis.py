"""Query analysis: route + entity/keyword extraction (LLM, with a deterministic fallback)."""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from ..llm.base import LLMClient, LLMError
from ..prompts import load as load_prompt
from ..text import content_tokens
from ..types import QueryPlan

log = logging.getLogger(__name__)

_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”|\'([^\']{3,})\'')
_CAPS = re.compile(r"\b[A-Z][\w'’-]*(?:\s+(?:[A-Z0-9][\w'’-]*|of|the|and|de|van|von))*")
_QUESTION_WORDS = {
    "which",
    "what",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "tell",
    "list",
    "show",
    "summarize",
    "explain",
    "describe",
    "give",
    "find",
    "the",
    "a",
    "an",
    "in",
    "on",
    "of",
    "for",
    "and",
    "or",
}
_GLOBAL_HINTS = re.compile(
    r"\b(summari[sz]e|overview|themes?|trends?|across (all|my)|big picture|main topics?)\b", re.I
)


class RouterOut(BaseModel):
    type: Literal["lookup", "multi_hop", "global", "no_retrieval"] = "lookup"
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


def _strip_leading_question_words(span: str) -> str:
    words = span.split()
    while words and words[0].casefold() in _QUESTION_WORDS:
        words.pop(0)
    return " ".join(words)


def heuristic_plan(text: str) -> QueryPlan:
    entities: list[str] = []
    for m in _QUOTED.finditer(text):
        entities.append(next(g for g in m.groups() if g))
    for m in _CAPS.finditer(text):
        span = _strip_leading_question_words(m.group(0).strip())
        if span and span.casefold() not in _QUESTION_WORDS:
            entities.append(span)
    seen: set[str] = set()
    ents = [e for e in entities if not (e.casefold() in seen or seen.add(e.casefold()))]
    kws: list[str] = []
    for t in content_tokens(text):
        if t not in kws:
            kws.append(t)
    qtype = "global" if _GLOBAL_HINTS.search(text) else "lookup"
    return QueryPlan(text=text, type=qtype, entities=ents[:8], keywords=kws[:10])


def analyze_query(text: str, llm: LLMClient | None = None, model: str | None = None) -> QueryPlan:
    base = heuristic_plan(text)
    if llm is None:
        return base
    try:
        out = llm.structured(load_prompt("router"), f"Question: {text}", RouterOut, model=model)
    except LLMError as e:
        log.warning("query analysis via LLM failed (%s); using heuristics", e)
        return base
    entities = [e.strip() for e in out.entities if e.strip()] or base.entities
    keywords = [k.strip() for k in out.keywords if k.strip()] or base.keywords
    # union: the LLM's entities plus any capitalised spans/quoted strings the heuristic found
    low = {e.casefold() for e in entities}
    entities += [e for e in base.entities if e.casefold() not in low]
    return QueryPlan(text=text, type=out.type, entities=entities[:8], keywords=keywords[:10])
