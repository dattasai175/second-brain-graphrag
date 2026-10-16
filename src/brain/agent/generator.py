"""Single-pass cited answer generation (Phase 1)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.base import LLMClient
from ..prompts import load as load_prompt
from ..types import Evidence

_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_CITE = re.compile(r"\bE\d+\b")
NO_EVIDENCE = "I couldn't find anything relevant to that in your notes."


@dataclass
class Answer:
    text: str
    cited: list[str] = field(default_factory=list)  # evidence ids actually cited, in order
    unknown_citations: list[str] = field(default_factory=list)  # cited ids not in the evidence
    uncited: bool = False  # model answered but cited nothing
    abstained: bool = False
    model: str = ""


def format_evidence(evidence: list[Evidence]) -> str:
    blocks = []
    for ev in evidence:
        header = f"[{ev.id}] ({ev.source})"
        blocks.append(f"{header}\n{ev.text.strip()}")
    return "\n\n".join(blocks)


def extract_citations(text: str, valid_ids: set[str]) -> tuple[list[str], list[str]]:
    cited: list[str] = []
    unknown: list[str] = []
    for m in _BRACKET.finditer(text):
        for cid in _CITE.findall(m.group(1)):
            bucket = cited if cid in valid_ids else unknown
            if cid not in bucket:
                bucket.append(cid)
    return cited, unknown


def generate_answer(
    question: str,
    evidence: list[Evidence],
    llm: LLMClient,
    *,
    model: str | None = None,
) -> Answer:
    if not evidence:
        return Answer(text=NO_EVIDENCE, abstained=True, model="")
    user = f"<evidence>\n{format_evidence(evidence)}\n</evidence>\n\nQuestion: {question}"
    text = llm.chat(load_prompt("generator"), user, model=model)
    cited, unknown = extract_citations(text, {e.id for e in evidence})
    return Answer(
        text=text,
        cited=cited,
        unknown_citations=unknown,
        uncited=not cited,
        model=model or llm.model,
    )
