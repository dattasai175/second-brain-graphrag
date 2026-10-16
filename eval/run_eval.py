#!/usr/bin/env python
"""Retrieval evaluation: run the ablation ladder over a golden set and report metrics.

    uv run python eval/run_eval.py                       # default golden set + systems
    uv run python eval/run_eval.py --golden eval/datasets/golden.sample.jsonl
    uv run --  python eval/run_eval.py -c config/settings.smoke.yaml --systems dense,hybrid_graph

Systems (see brain.retrieval.PRESETS): dense -> dense_bm25 -> dense_bm25_rerank -> hybrid_graph.
Every system sees the same question analysis (computed once per question), so differences are
due to retrieval, not routing noise. A component earns its place in the pipeline only if it moves
the numbers for the question type it targets (graph -> multi_hop all_hops@k).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEMS = ["dense", "dense_bm25", "dense_bm25_rerank", "hybrid_graph"]
KS = (1, 3, 5, 10)

console = Console()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--golden", default=str(ROOT / "eval" / "datasets" / "golden.jsonl"))
    ap.add_argument("-c", "--config", default=None, help="settings.yaml (default: auto-discovered)")
    ap.add_argument(
        "--systems", default=",".join(DEFAULT_SYSTEMS), help="comma-separated retrieval presets"
    )
    ap.add_argument("-k", type=int, default=10, help="evidence depth retrieved per question")
    ap.add_argument(
        "--no-router",
        action="store_true",
        help="heuristic query analysis only (no LLM call per question)",
    )
    ap.add_argument(
        "--out", default=None, help="JSON report path (default: eval/reports/<ts>.json)"
    )
    args = ap.parse_args(argv)

    from brain.evalkit import aggregate, load_golden, score_question
    from brain.llm.base import LLMError
    from brain.retrieval import PRESETS, Retriever
    from brain.runtime import open_runtime

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in PRESETS]
    if unknown:
        console.print(f"[red]unknown systems:[/red] {unknown}; choose from {list(PRESETS)}")
        return 2

    items = load_golden(args.golden)
    if not items:
        console.print(
            f"[yellow]No golden questions in {args.golden}.[/yellow]\n"
            "Add JSONL lines (format in the file header, example: eval/datasets/golden.sample.jsonl)."
        )
        return 2

    try:
        rt = open_runtime(args.config)
    except (FileNotFoundError, RuntimeError, LLMError) as e:
        console.print(f"[red]error:[/red] {e}")
        return 2
    counts = rt.db.counts()
    if counts["chunks"] == 0:
        console.print("[red]The index is empty.[/red] Run `brain ingest <folder>` first.")
        return 2

    retriever = Retriever(
        rt.settings, rt.db, rt.embedder, None if args.no_router else rt.llm, rt.reranker
    )
    try:
        plans = {}
        t0 = time.perf_counter()
        for it in items:
            plans[it.id] = retriever.plan(it.question)
        plan_ms = (time.perf_counter() - t0) * 1000 / len(items)

        results: dict[str, dict] = {}
        per_question: dict[str, list[dict]] = {}
        notes: dict[str, list[str]] = {}
        for system in systems:
            scores = []
            for it in items:
                res = retriever.retrieve(it.question, system, k=args.k, plan=plans[it.id])
                scores.append(
                    score_question(
                        it, res.evidence, ks=KS, latency_ms=res.timings.get("total", 0.0)
                    )
                )
                for n in res.notes:
                    notes.setdefault(system, [])
                    if n not in notes[system]:
                        notes[system].append(n)
            results[system] = aggregate(scores, KS)
            per_question[system] = [
                {
                    "id": s.id,
                    "type": s.type,
                    "answerable": s.answerable,
                    "span_ranks": s.span_ranks,
                    "mrr": round(s.mrr, 3),
                    "latency_ms": s.latency_ms,
                }
                for s in scores
            ]
    except LLMError as e:
        console.print(f"[red]error:[/red] {e}")
        return 1

    # ----------------------------------------------------------------- report
    n_ans = results[systems[0]]["all"]["n"]
    n_unans = len(items) - n_ans
    console.print(
        f"\n[bold]Retrieval eval[/bold]  golden={Path(args.golden).name}  questions={len(items)} "
        f"({n_ans} answerable, {n_unans} unanswerable excluded)  k={args.k}\n"
        f"embedder={rt.embedder.name}  reranker={rt.reranker.name}  "
        f"router={'heuristic' if args.no_router else rt.llm.model}  "
        f"index: {counts['documents']} docs / {counts['chunks']} chunks / "
        f"{counts['entities']} entities / {counts['relations']} relations"
    )

    def table(title: str, group: str) -> None:
        t = Table(title=title, header_style="bold")
        t.add_column("system", no_wrap=True)
        # @10 columns equal @k when fewer than 10 items were retrieved (-k < 10)
        for col in ("n", "R@1", "R@5", "R@10", "Hit", "AllHop", "MRR", "nDCG", "ms"):
            t.add_column(col, justify="right", no_wrap=True)
        for system in systems:
            m = results[system].get(group)
            if not m:
                continue
            label = system + (
                " *" if system in notes and any("reranker" in n for n in notes[system]) else ""
            )
            t.add_row(
                label,
                str(m["n"]),
                f"{m['recall@1']:.2f}",
                f"{m['recall@5']:.2f}",
                f"{m['recall@10']:.2f}",
                f"{m['hit@10']:.2f}",
                f"{m['all_hops@10']:.2f}",
                f"{m['mrr']:.2f}",
                f"{m['ndcg@10']:.2f}",
                f"{m['latency_p50_ms']:.0f}",
            )
        console.print(t)

    table("All answerable questions", "all")
    for group in sorted({g for r in results.values() for g in r} - {"all"}):
        table(f"Question type: {group}", group)
    if any(any("reranker" in n for n in v) for v in notes.values()):
        console.print(
            "[yellow]* reranker inactive: rerank rows equal their non-rerank counterparts "
            "(uv sync --extra ml to enable).[/yellow]"
        )
    console.print(f"[dim]query analysis: {plan_ms:.0f} ms/question (shared across systems)[/dim]")

    out = (
        Path(args.out)
        if args.out
        else ROOT / "eval" / "reports" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "golden": str(args.golden),
                "k": args.k,
                "systems": systems,
                "embedder": rt.embedder.name,
                "reranker": rt.reranker.name,
                "router": "heuristic" if args.no_router else rt.llm.model,
                "index": counts,
                "results": results,
                "per_question": per_question,
                "notes": notes,
            },
            indent=2,
        )
    )
    console.print(f"[dim]report: {out}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
