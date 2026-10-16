"""Agentic layer. Phase 1 ships the single-pass generator; the planner/critic/verifier loop is Phase 2."""

from .generator import Answer, format_evidence, generate_answer

__all__ = ["Answer", "format_evidence", "generate_answer"]
