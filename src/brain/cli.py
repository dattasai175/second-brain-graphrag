"""`brain` command line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .agent.generator import generate_answer
from .ingest import IngestPipeline
from .llm.base import LLMError
from .retrieval import PRESETS, Retriever
from .runtime import Runtime, open_runtime
from .trace import save_trace

app = typer.Typer(
    help="Adaptive personal knowledge-graph RAG: ingest your notes, ask questions with citations.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", envvar="BRAIN_CONFIG", help="Path to settings.yaml"),
]


@app.callback()
def main(
    ctx: typer.Context,
    config: ConfigOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ctx.obj = {"config": config}


def _runtime(ctx: typer.Context, **kwargs) -> Runtime:
    try:
        return open_runtime(ctx.obj["config"], **kwargs)
    except (FileNotFoundError, RuntimeError, LLMError) as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(2) from e


def _fail(e: Exception) -> typer.Exit:
    err.print(f"[red]error:[/red] {e}")
    return typer.Exit(1)


# ------------------------------------------------------------------------ doctor
@app.command()
def doctor(ctx: typer.Context) -> None:
    """Check Ollama, models, vector backend and optional components."""
    rt = _runtime(ctx)
    s = rt.settings
    rows: list[tuple[str, str, str]] = []
    try:
        models = rt.llm.list_models()
        rows.append(("ollama", "ok", s.llm.base_url))

        def has(m: str) -> bool:
            return m in models or f"{m}:latest" in models

        rows.append(
            (
                "chat model",
                "ok" if has(s.llm.model) else "MISSING",
                s.llm.model if has(s.llm.model) else f"ollama pull {s.llm.model}",
            )
        )
        if s.embedding.backend == "ollama":
            ok = has(s.embedding.model)
            rows.append(
                (
                    "embedding model",
                    "ok" if ok else "MISSING",
                    s.embedding.model if ok else f"ollama pull {s.embedding.model}",
                )
            )
        else:
            rows.append(("embedding model", "ok", f"{s.embedding.backend}:{s.embedding.model}"))
    except LLMError as e:
        rows.append(("ollama", "UNREACHABLE", str(e)))
    rows.append(
        (
            "vector backend",
            "ok",
            "sqlite-vec (vec0)"
            if rt.db.vec_ext
            else "numpy brute-force (this Python can't load sqlite extensions)",
        )
    )
    rows.append(("reranker", "ok" if rt.reranker.active else "inactive", rt.reranker.name))
    try:
        import docling  # noqa: F401

        rows.append(("pdf parser", "ok", "docling"))
    except ImportError:
        rows.append(("pdf parser", "fallback", "pypdf (uv sync --extra pdf for Docling)"))
    rows.append(("database", "ok", str(s.db_path)))
    table = Table(show_header=True, header_style="bold")
    for col in ("check", "status", "detail"):
        table.add_column(col)
    for r in rows:
        style = "green" if r[1] == "ok" else "yellow"
        table.add_row(r[0], f"[{style}]{r[1]}[/{style}]", r[2])
    console.print(table)


# ------------------------------------------------------------------------ ingest
@app.command()
def ingest(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Folder (or single file) to ingest")],
    force: Annotated[bool, typer.Option(help="Re-ingest even if unchanged")] = False,
    no_extract: Annotated[
        bool, typer.Option("--no-extract", help="Skip LLM entity extraction")
    ] = False,
    reset: Annotated[bool, typer.Option(help="Delete the database first")] = False,
) -> None:
    """Index documents: chunks + vectors + BM25 + knowledge graph (incremental)."""
    cfg_path = ctx.obj["config"]
    if reset:
        from .config import load_settings

        db_file = load_settings(cfg_path).db_path
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db_file}{suffix}").unlink(missing_ok=True)
        console.print(f"[yellow]reset:[/yellow] removed {db_file}")
    rt = _runtime(ctx, with_reranker=False)
    try:
        pipeline = IngestPipeline(rt.settings, rt.db, rt.embedder, rt.llm)
        rt.db.set_meta("embedder", rt.embedder.name)

        def progress(event: str, msg: str) -> None:
            color = {
                "ingested": "green",
                "unchanged": "dim",
                "failed": "red",
                "deleted": "yellow",
            }.get(event, "white")
            console.print(f"[{color}]{event:<9}[/{color}] {msg}")

        report = pipeline.run(path, force=force, extract=not no_extract, progress=progress)
    except (LLMError, FileNotFoundError, RuntimeError) as e:
        raise _fail(e) from e
    console.print(
        f"\n[bold]done[/bold] in {report.seconds:.1f}s: scanned {report.scanned} | "
        f"added {report.added}, updated {report.updated}, unchanged {report.unchanged}, "
        f"deleted {report.deleted}, failed {len(report.failed)}\n"
        f"chunks {report.chunks} | entities +{report.entities_created} "
        f"(merged {report.entities_merged}) | relations {report.relations}"
        + (
            f" | extraction failures {report.extraction_failures}"
            if report.extraction_failures
            else ""
        )
    )
    if report.extraction_failures:
        console.print(
            "[yellow]some chunks had no LLM extraction (Ollama unreachable or invalid output); "
            "re-run `brain ingest` to retry them.[/yellow]"
        )
    for rel, msg in report.failed:
        err.print(f"[red]failed[/red] {rel}: {msg}")
    if report.failed:
        raise typer.Exit(1)


# ------------------------------------------------------------------ search / ask
def _retriever(rt: Runtime, use_llm_router: bool) -> Retriever:
    stored = rt.db.get_meta("embedder")
    if stored and stored != rt.embedder.name:
        err.print(
            f"[yellow]warning:[/yellow] index was built with embedder '{stored}' but config uses "
            f"'{rt.embedder.name}'; results will be poor. Re-ingest with --reset."
        )
    return Retriever(
        rt.settings, rt.db, rt.embedder, rt.llm if use_llm_router else None, rt.reranker
    )


def _print_evidence(result, show_trace: bool) -> None:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    for col in ("id", "score", "source", "text"):
        table.add_column(col, overflow="fold")
    for ev in result.evidence:
        snippet = " ".join(ev.text.split())
        table.add_row(
            ev.id, f"{ev.score:.4f}", ev.source, snippet[:220] + ("…" if len(snippet) > 220 else "")
        )
    console.print(table)
    if show_trace:
        console.print(
            f"[dim]route={result.plan.type} entities={result.plan.entities} "
            f"linked={result.linked_entities}[/dim]"
        )
        console.print(f"[dim]timings_ms={result.timings}[/dim]")
        for ev in result.evidence:
            console.print(
                f"[dim]{ev.id}: channels={ {k: round(v, 4) for k, v in ev.channels.items()} } "
                f"ranks={ev.ranks} path={ev.graph_path}[/dim]"
            )
        for note in result.notes:
            console.print(f"[yellow]note:[/yellow] {note}")


PresetOpt = Annotated[
    str, typer.Option("--preset", "-p", help=f"Retrieval preset: {', '.join(PRESETS)}")
]


@app.command()
def search(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument()],
    preset: PresetOpt = "hybrid_graph",
    k: Annotated[int, typer.Option("--k", "-k")] = 8,
    trace: Annotated[bool, typer.Option(help="Show channel scores, ranks and graph paths")] = False,
    no_router: Annotated[
        bool, typer.Option("--no-router", help="Skip the LLM query-analysis call")
    ] = False,
) -> None:
    """Retrieve evidence only (no answer generation)."""
    if preset not in PRESETS:
        raise _fail(ValueError(f"unknown preset '{preset}'; choose from {', '.join(PRESETS)}"))
    rt = _runtime(ctx)
    try:
        result = _retriever(rt, not no_router).retrieve(question, preset, k=k)
    except LLMError as e:
        raise _fail(e) from e
    _print_evidence(result, trace)


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument()],
    preset: PresetOpt = "hybrid_graph",
    k: Annotated[int, typer.Option("--k", "-k")] = 8,
    trace: Annotated[bool, typer.Option(help="Show retrieval trace")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
    no_router: Annotated[
        bool, typer.Option("--no-router", help="Skip the LLM query-analysis call")
    ] = False,
) -> None:
    """Answer a question from your notes, with citations."""
    if preset not in PRESETS:
        raise _fail(ValueError(f"unknown preset '{preset}'; choose from {', '.join(PRESETS)}"))
    rt = _runtime(ctx)
    try:
        result = _retriever(rt, not no_router).retrieve(question, preset, k=k)
        answer = generate_answer(question, result.evidence, rt.llm)
    except LLMError as e:
        raise _fail(e) from e
    qid = save_trace(rt.db, question, result, answer)
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "query_id": qid,
                    "answer": answer.text,
                    "cited": answer.cited,
                    "unknown_citations": answer.unknown_citations,
                    "route": result.plan.type,
                    "linked_entities": result.linked_entities,
                    "timings_ms": result.timings,
                    "evidence": [e.to_dict() for e in result.evidence],
                },
                indent=2,
            )
        )
        return
    console.print(f"\n{answer.text}\n")
    if answer.unknown_citations:
        err.print(f"[yellow]warning:[/yellow] answer cites unknown ids: {answer.unknown_citations}")
    if answer.uncited and not answer.abstained:
        err.print("[yellow]warning:[/yellow] answer contains no citations; treat with caution")
    by_id = {e.id: e for e in result.evidence}
    shown = answer.cited or [e.id for e in result.evidence[:3]]
    console.print(
        "[bold]Sources[/bold]" + ("" if answer.cited else " [dim](top retrieved; none cited)[/dim]")
    )
    for eid in shown:
        ev = by_id.get(eid)
        if ev:
            console.print(f"  [cyan]{eid}[/cyan] {ev.source}")
    if trace:
        console.print()
        _print_evidence(result, True)


# ------------------------------------------------------------------ stats / list
@app.command()
def stats(ctx: typer.Context) -> None:
    """Counts for the current index."""
    rt = _runtime(ctx, with_reranker=False)
    table = Table(show_header=False)
    for k, v in rt.db.counts().items():
        table.add_row(k, str(v))
    table.add_row("embedder", rt.db.get_meta("embedder", "-") or "-")
    table.add_row("vector backend", rt.db.get_meta("vec_backend", "-") or "-")
    table.add_row("database", str(rt.settings.db_path))
    console.print(table)


@app.command()
def entities(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option()] = 30,
    type_: Annotated[str | None, typer.Option("--type")] = None,
    match: Annotated[str | None, typer.Option(help="Substring of the name")] = None,
) -> None:
    """List entities by number of chunks mentioning them."""
    rt = _runtime(ctx, with_reranker=False)
    sql = (
        "SELECT e.id, e.canonical_name, e.type, e.description, "
        "(SELECT count(*) FROM mentions m WHERE m.entity_id = e.id) AS n, "
        "(SELECT count(*) FROM relations r WHERE r.src_id = e.id OR r.dst_id = e.id) AS deg "
        "FROM entities e WHERE 1=1"
    )
    args: list = []
    if type_:
        sql += " AND e.type = ?"
        args.append(type_)
    if match:
        sql += " AND e.canonical_name LIKE ?"
        args.append(f"%{match}%")
    sql += " ORDER BY n DESC, deg DESC LIMIT ?"
    args.append(limit)
    table = Table(show_header=True, header_style="bold")
    for col in ("id", "name", "type", "chunks", "relations", "description"):
        table.add_column(col, overflow="fold")
    for r in rt.db.conn.execute(sql, args):
        table.add_row(
            str(r["id"]),
            r["canonical_name"],
            r["type"],
            str(r["n"]),
            str(r["deg"]),
            r["description"][:60],
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
