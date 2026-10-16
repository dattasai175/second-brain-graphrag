"""Golden-set format.

One JSON object per line; blank lines and lines starting with ``#`` are ignored.

    {"id": "q1", "question": "...", "type": "lookup|multi_hop|global|unanswerable",
     "gold": [{"path": "notes/a.md", "quote": "exact text from the document"}],
     "reference_answer": "...", "answerable": true}

Gold evidence is anchored to *document + quote*, never to chunk ids: chunk ids change whenever
you re-chunk or re-ingest, quotes don't. ``path`` matches as a suffix of the document's path, and
a span without a ``quote`` matches any chunk of that document.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldSpan(BaseModel):
    path: str
    quote: str = ""


class GoldItem(BaseModel):
    id: str
    question: str
    type: str = "lookup"
    gold: list[GoldSpan] = Field(default_factory=list)
    reference_answer: str = ""
    answerable: bool = True


def load_golden(path: str | Path) -> list[GoldItem]:
    items: list[GoldItem] = []
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = GoldItem.model_validate(json.loads(line))
        except Exception as e:
            raise ValueError(f"{path}:{n}: invalid golden item ({e})") from e
        if item.answerable and not item.gold:
            raise ValueError(f"{path}:{n}: answerable item '{item.id}' has no gold spans")
        items.append(item)
    return items
