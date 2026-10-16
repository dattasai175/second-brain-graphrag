Extract a knowledge graph from the passage.

Entity types: {types}
Predicates: {predicates} (use "related_to" only if nothing fits, and describe the relation in `description`)

Rules:
- State only what the passage says. No outside knowledge.
- Resolve pronouns and "the team"-style references to names when the passage makes them clear.
- Use the fullest name form that appears in the passage.
- Skip generic nouns ("the meeting", "a decision") unless they are named things.
- Every relation needs `evidence`: an exact quote (at most 25 words) copied from the passage.
- `source` and `target` of a relation must be names that also appear in `entities`.

Return JSON:
{"entities": [{"name": "...", "type": "...", "description": "one short phrase"}],
 "relations": [{"source": "...", "predicate": "...", "target": "...", "description": "...", "evidence": "..."}]}
