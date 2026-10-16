"""Persist a query trace so answers can be explained now and credited with feedback later."""

from __future__ import annotations

import json

from .agent.generator import Answer
from .store.db import Database
from .types import RetrievalResult


def save_trace(db: Database, question: str, result: RetrievalResult, answer: Answer | None) -> int:
    trace = {
        "plan": {
            "type": result.plan.type,
            "entities": result.plan.entities,
            "keywords": result.plan.keywords,
        },
        "preset": result.preset,
        "linked_entities": result.linked_entities,
        "timings_ms": result.timings,
        "notes": result.notes,
        "evidence": [e.to_dict() for e in result.evidence],
        "cited": answer.cited if answer else [],
    }
    with db.transaction() as conn:
        cur = conn.execute(
            "INSERT INTO queries(text, route, trace_json, answer, model_info) VALUES (?, ?, ?, ?, ?)",
            (
                question,
                result.plan.type,
                json.dumps(trace),
                answer.text if answer else None,
                answer.model if answer else None,
            ),
        )
        return int(cur.lastrowid)
