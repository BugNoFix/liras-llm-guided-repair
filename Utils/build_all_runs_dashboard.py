#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import build_model_runs_analysis as analysis

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


DEFAULT_RUN_DIRS = ("RunsNoCoT", "RunsCoT", "Runs2shotNoCoT", "Runs2ShotCoT")
DEFAULT_OUTPUT = ROOT / "Report" / "all_runs_dashboard.html"
OUTCOMES = ("success", "failed", "running", "unknown")
OUTCOME_COLORS = {
    "success": "#15803d",
    "failed": "#b91c1c",
    "running": "#b45309",
    "unknown": "#64748b",
}
MODEL_COLORS = {
    "Qwen/Qwen3.5-9B": "#2563eb",
    "zai-org/GLM-5.2": "#7c3aed",
    "openai/gpt-oss-20b": "#0891b2",
    "Qwen/Qwen3.6-35B-A3B": "#ea580c",
    "google/gemma-4-31B-it": "#16a34a",
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_number(value: Any, digits: int = 0) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    if digits:
        return f"{number:,.{digits}f}"
    return f"{number:,.0f}"


def _fmt_pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.1f}%"


def _fmt_duration(seconds: Any) -> str:
    number = _safe_float(seconds)
    if number is None:
        return "-"
    if number < 60:
        return f"{number:.1f}s"
    minutes = int(number // 60)
    return f"{minutes}m {round(number % 60)}s"


def _short_model(model: str) -> str:
    return analysis._short_model_label(model)


def _collect_all(run_dirs: list[Path], include_liras_code: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        group = run_dir.name
        for record in analysis._collect_records(run_dir, include_liras_code=include_liras_code):
            enriched = dict(record)
            enriched["run_group"] = group
            enriched["record_key"] = f"{group}/{record.get('metadata_path') or record.get('run_dir') or record.get('run_id')}"
            records.append(enriched)
    return records


def _group_by(records: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(record.get(key) or "unknown") for key in keys)].append(record)
    return dict(grouped)


def _completion_samples(records: list[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for record in records:
        for value in record.get("dsl_completion_token_samples") or []:
            number = _safe_float(value)
            if number is not None:
                samples.append(number)
    return samples


def _time_samples(records: list[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for record in records:
        for value in record.get("dsl_generation_time_samples") or []:
            number = _safe_float(value)
            if number is not None:
                samples.append(number)
    return samples


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    outcomes = Counter(str(r.get("outcome") or "unknown") for r in records)
    models = sorted({str(r.get("model") or "unknown") for r in records})
    groups = sorted({str(r.get("run_group") or "unknown") for r in records})
    scenarios = sorted({str(r.get("scenario") or "unknown") for r in records})

    group_rows = []
    for group, rows in sorted(_group_by(records, "run_group").items()):
        group_records = rows
        group_outcomes = Counter(str(r.get("outcome") or "unknown") for r in group_records)
        successes = group_outcomes["success"]
        token_total = sum(
            number
            for number in (_safe_float(r.get("llm_output_tokens") or r.get("dsl_completion_tokens")) for r in group_records)
            if number is not None
        )
        cycles = [number for number in (_safe_float(r.get("cycles")) for r in group_records) if number is not None]
        times = _time_samples(group_records)
        tokens_per_success = token_total / successes if successes and token_total else None
        group_rows.append(
            {
                "group": group[0],
                "total": len(group_records),
                "success": successes,
                "failed": group_outcomes["failed"],
                "running": group_outcomes["running"],
                "unknown": group_outcomes["unknown"],
                "success_rate": successes / len(group_records) if group_records else 0,
                "avg_cycles": sum(cycles) / len(cycles) if cycles else None,
                "avg_dsl_time": sum(times) / len(times) if times else None,
                "tokens_per_success": tokens_per_success,
            }
        )

    model_rows = []
    for (group, model), rows in sorted(_group_by(records, "run_group", "model").items()):
        successes = sum(1 for r in rows if r.get("outcome") == "success")
        token_total = sum(
            number
            for number in (_safe_float(r.get("llm_output_tokens") or r.get("dsl_completion_tokens")) for r in rows)
            if number is not None
        )
        cycles = [number for number in (_safe_float(r.get("cycles")) for r in rows) if number is not None]
        tokens = _completion_samples(rows)
        model_rows.append(
            {
                "group": group,
                "model": model,
                "total": len(rows),
                "success": successes,
                "failed": sum(1 for r in rows if r.get("outcome") == "failed"),
                "success_rate": successes / len(rows) if rows else 0,
                "avg_cycles": sum(cycles) / len(cycles) if cycles else None,
                "avg_output_tokens_per_dsl": sum(tokens) / len(tokens) if tokens else None,
                "tokens_per_success": token_total / successes if successes and token_total else None,
            }
        )

    return {
        "total": total,
        "outcomes": dict(outcomes),
        "models": models,
        "groups": groups,
        "scenarios": scenarios,
        "group_rows": group_rows,
        "model_rows": model_rows,
    }


def _save_fig(path: Path) -> str:
    assert plt is not None
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path.name


def _write_outcome_by_group(records: list[dict[str, Any]], asset_dir: Path) -> str:
    assert plt is not None
    groups = sorted({str(r.get("run_group") or "unknown") for r in records})
    counts = {group: Counter(str(r.get("outcome") or "unknown") for r in records if r.get("run_group") == group) for group in groups}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bottom = [0] * len(groups)
    for outcome in OUTCOMES:
        values = [counts[group][outcome] for group in groups]
        ax.bar(groups, values, bottom=bottom, label=outcome, color=OUTCOME_COLORS[outcome])
        bottom = [left + value for left, value in zip(bottom, values)]
    ax.set_title("Esiti per cartella run", fontweight="bold")
    ax.set_ylabel("Run")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save_fig(asset_dir / "outcomes_by_group.png")


def _write_success_heatmap(records: list[dict[str, Any]], asset_dir: Path) -> str:
    assert plt is not None
    groups = sorted({str(r.get("run_group") or "unknown") for r in records})
    models = sorted({str(r.get("model") or "unknown") for r in records}, key=_short_model)
    matrix: list[list[float]] = []
    for group in groups:
        row = []
        for model in models:
            rows = [r for r in records if r.get("run_group") == group and r.get("model") == model]
            row.append((sum(1 for r in rows if r.get("outcome") == "success") / len(rows)) if rows else 0)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(max(8, 1.35 * len(models)), 4.8))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Success rate per cartella e modello", fontweight="bold")
    ax.set_xticks(range(len(models)), [_short_model(model) for model in models], rotation=25, ha="right")
    ax.set_yticks(range(len(groups)), groups)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value * 100:.0f}%", ha="center", va="center", color="#172033", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    return _save_fig(asset_dir / "success_heatmap.png")


def _write_grouped_bars(
    records: list[dict[str, Any]],
    asset_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    metric: str,
) -> str:
    assert plt is not None
    groups = sorted({str(r.get("run_group") or "unknown") for r in records})
    models = sorted({str(r.get("model") or "unknown") for r in records}, key=_short_model)
    width = 0.78 / max(len(models), 1)
    x_positions = list(range(len(groups)))
    fig, ax = plt.subplots(figsize=(max(10, 1.35 * len(groups) + 0.9 * len(models)), 5.6))
    for index, model in enumerate(models):
        values = []
        for group in groups:
            rows = [r for r in records if r.get("run_group") == group and r.get("model") == model]
            if metric == "avg_cycles":
                samples = [number for number in (_safe_float(r.get("cycles")) for r in rows) if number is not None]
                values.append(sum(samples) / len(samples) if samples else 0)
            elif metric == "tokens_per_success":
                successes = sum(1 for r in rows if r.get("outcome") == "success")
                token_total = sum(
                    number
                    for number in (_safe_float(r.get("llm_output_tokens") or r.get("dsl_completion_tokens")) for r in rows)
                    if number is not None
                )
                values.append(token_total / successes if successes and token_total else 0)
            else:
                raise ValueError(metric)
        offsets = [x - 0.39 + width / 2 + index * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=_short_model(model), color=MODEL_COLORS.get(model))
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x_positions, groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    return _save_fig(asset_dir / filename)


def _write_boxplots(records: list[dict[str, Any]], asset_dir: Path, metric: str, title: str, xlabel: str, filename: str) -> str:
    assert plt is not None
    groups = sorted({str(r.get("run_group") or "unknown") for r in records})
    values = []
    labels = []
    for group in groups:
        rows = [r for r in records if r.get("run_group") == group]
        if metric == "cycles":
            samples = [number for number in (_safe_float(r.get("cycles")) for r in rows) if number is not None]
        elif metric == "tokens":
            samples = _completion_samples(rows)
        elif metric == "time":
            samples = _time_samples(rows)
        else:
            raise ValueError(metric)
        if samples:
            labels.append(group)
            values.append(samples)
    fig, ax = plt.subplots(figsize=(10, max(4.6, 0.55 * len(labels) + 1.8)))
    if not values:
        ax.axis("off")
        ax.text(0.5, 0.5, "Nessun dato disponibile", ha="center", va="center")
    else:
        ax.boxplot(values, vert=False, tick_labels=labels, patch_artist=True, showmeans=True)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_title(title, fontweight="bold")
    return _save_fig(asset_dir / filename)


def _write_figures(records: list[dict[str, Any]], output_html: Path) -> list[dict[str, str]]:
    if not HAS_MATPLOTLIB or plt is None or not records:
        return []
    asset_dir = output_html.with_suffix("")
    asset_dir = asset_dir.parent / f"{asset_dir.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    return [
        {"title": "Esiti per cartella", "file": _write_outcome_by_group(records, asset_dir)},
        {"title": "Success rate per modello", "file": _write_success_heatmap(records, asset_dir)},
        {
            "title": "Cicli medi per run",
            "file": _write_grouped_bars(
                records,
                asset_dir,
                "avg_cycles_by_group_model.png",
                "Cicli medi per run, cartella e modello",
                "Cicli medi",
                "avg_cycles",
            ),
        },
        {
            "title": "Token per successo",
            "file": _write_grouped_bars(
                records,
                asset_dir,
                "tokens_per_success_by_group_model.png",
                "Output token generate/repair per successo, cartella e modello",
                "Token per successo",
                "tokens_per_success",
            ),
        },
        {
            "title": "Distribuzione cicli",
            "file": _write_boxplots(
                records,
                asset_dir,
                "cycles",
                "Distribuzione dei cicli per cartella",
                "Cicli per run",
                "cycles_boxplot_by_group.png",
            ),
        },
        {
            "title": "Output token per DSL",
            "file": _write_boxplots(
                records,
                asset_dir,
                "tokens",
                "Output token per singolo DSL generato",
                "Output token",
                "tokens_boxplot_by_group.png",
            ),
        },
        {
            "title": "Tempo chiamate DSL",
            "file": _write_boxplots(
                records,
                asset_dir,
                "time",
                "Tempo delle chiamate generate/repair per cartella",
                "Secondi",
                "time_boxplot_by_group.png",
            ),
        },
    ]


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for key, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _label in columns:
            value = row.get(key)
            if key.endswith("rate"):
                text = _fmt_pct(value)
            elif key == "avg_dsl_time":
                text = _fmt_duration(value)
            elif key.startswith("avg"):
                text = _fmt_number(value, 2)
            elif key.endswith("success"):
                text = _fmt_number(value, 0)
            else:
                text = str(value if value is not None else "-")
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _write_html(records: list[dict[str, Any]], output_html: Path, figures: list[dict[str, str]]) -> None:
    summary = _summary(records)
    success = summary["outcomes"].get("success", 0)
    failed = summary["outcomes"].get("failed", 0)
    running = summary["outcomes"].get("running", 0)
    figure_html = "\n".join(
        '<article class="panel chart"><h2>{title}</h2><img src="{src}" alt="{title}" /></article>'.format(
            title=html.escape(item["title"]),
            src=html.escape(f"{output_html.with_suffix('').name}_assets/{item['file']}"),
        )
        for item in figures
    )
    group_table = _table(
        summary["group_rows"],
        [
            ("group", "Cartella"),
            ("total", "Run"),
            ("success", "Successi"),
            ("failed", "Fallite"),
            ("running", "In corso"),
            ("success_rate", "Success rate"),
            ("avg_cycles", "Cicli medi"),
            ("avg_dsl_time", "Tempo DSL medio"),
            ("tokens_per_success", "Token/successo"),
        ],
    )
    model_table = _table(
        summary["model_rows"],
        [
            ("group", "Cartella"),
            ("model", "Modello"),
            ("total", "Run"),
            ("success", "Successi"),
            ("failed", "Fallite"),
            ("success_rate", "Success rate"),
            ("avg_cycles", "Cicli medi"),
            ("avg_output_tokens_per_dsl", "Output token medi/DSL"),
            ("tokens_per_success", "Token/successo"),
        ],
    )
    recent = sorted(records, key=lambda r: str(r.get("started_at") or ""), reverse=True)[:80]
    recent_rows = []
    for record in recent:
        recent_rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('run_group') or '-'))}</td>"
            f"<td class=\"mono\">{html.escape(str(record.get('run_id') or '-'))}</td>"
            f"<td>{html.escape(str(record.get('model') or '-'))}</td>"
            f"<td>{html.escape(str(record.get('scenario') or '-'))}</td>"
            f"<td><span class=\"status {html.escape(str(record.get('outcome') or 'unknown'))}\">{html.escape(str(record.get('outcome') or 'unknown'))}</span></td>"
            f"<td>{html.escape(str(record.get('failure_category') or '-'))}</td>"
            f"<td>{_fmt_number(record.get('cycles'))}</td>"
            f"<td>{_fmt_duration(record.get('duration_seconds'))}</td>"
            "</tr>"
        )
    generated_at = datetime.now().isoformat(timespec="seconds")
    payload_json = html.escape(json.dumps({"summary": summary}, ensure_ascii=False))
    output_html.write_text(
        f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dashboard tutte le run</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #64748b;
      --line: #dbe3ef;
      --accent: #2563eb;
      --soft: #eef4ff;
      --good: #15803d;
      --bad: #b91c1c;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Avenir Next", "Segoe UI", sans-serif;
    }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 22px; display: grid; gap: 16px; }}
    .hero {{ background: #172033; color: white; border-radius: 8px; padding: 22px; }}
    .hero h1 {{ margin: 0 0 7px; font-size: 30px; }}
    .hero p {{ margin: 0; color: #cbd5e1; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05); }}
    .card {{ padding: 14px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    .value {{ font-size: 28px; font-weight: 800; margin-top: 5px; }}
    .charts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    @media (max-width: 1050px) {{ .charts {{ grid-template-columns: 1fr; }} }}
    .chart {{ padding: 14px; }}
    .chart h2, .panel h2 {{ margin: 0 0 10px; font-size: 17px; }}
    .chart img {{ width: 100%; display: block; border: 1px solid var(--line); border-radius: 8px; background: white; }}
    .table-wrap {{ overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ position: sticky; top: 0; z-index: 1; background: #f8fafc; text-align: left; padding: 9px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    td {{ padding: 8px 9px; border-bottom: 1px solid #edf2f7; vertical-align: top; }}
    td:not(:first-child), th:not(:first-child) {{ text-align: right; }}
    td:nth-child(2), th:nth-child(2), .recent td:nth-child(3), .recent th:nth-child(3), .recent td:nth-child(4), .recent th:nth-child(4), .recent td:nth-child(5), .recent th:nth-child(5), .recent td:nth-child(6), .recent th:nth-child(6) {{ text-align: left; }}
    .panel {{ padding: 14px; }}
    .mono {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; }}
    .muted {{ color: var(--muted); }}
    .status {{ display: inline-block; border-radius: 999px; padding: 3px 8px; font-weight: 700; font-size: 12px; }}
    .status.success {{ color: var(--good); background: #dcfce7; }}
    .status.failed {{ color: var(--bad); background: #fee2e2; }}
    .status.running {{ color: var(--warn); background: #fef3c7; }}
    .status.unknown {{ color: var(--muted); background: #e2e8f0; }}
    script[type="application/json"] {{ display: none; }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>Dashboard tutte le run</h1>
      <p>Analisi aggregata di {html.escape(", ".join(summary["groups"]))}. Generata il {html.escape(generated_at)}.</p>
    </section>
    <section class="cards">
      <article class="card"><div class="label">Run totali</div><div class="value">{_fmt_number(summary["total"])}</div></article>
      <article class="card"><div class="label">Riuscite</div><div class="value">{_fmt_number(success)}</div></article>
      <article class="card"><div class="label">Fallite</div><div class="value">{_fmt_number(failed)}</div></article>
      <article class="card"><div class="label">In corso</div><div class="value">{_fmt_number(running)}</div></article>
      <article class="card"><div class="label">Success rate</div><div class="value">{_fmt_pct(success / max(summary["total"], 1))}</div></article>
      <article class="card"><div class="label">Modelli</div><div class="value">{_fmt_number(len(summary["models"]))}</div></article>
    </section>
    <section class="charts">{figure_html}</section>
    <section class="panel"><h2>Riepilogo per cartella</h2><div class="table-wrap">{group_table}</div></section>
    <section class="panel"><h2>Riepilogo per cartella e modello</h2><div class="table-wrap">{model_table}</div></section>
    <section class="panel recent"><h2>Run recenti</h2><div class="table-wrap"><table><thead><tr><th>Cartella</th><th>Run</th><th>Modello</th><th>Scenario</th><th>Esito</th><th>Motivo</th><th>Cicli</th><th>Durata</th></tr></thead><tbody>{''.join(recent_rows)}</tbody></table></div></section>
  </main>
  <script id="summary-payload" type="application/json">{payload_json}</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_dashboard(run_dirs: list[Path], output_html: Path, include_liras_code: bool = False, print_summary: bool = False) -> None:
    resolved_run_dirs = []
    for run_dir in run_dirs:
        resolved = run_dir.expanduser()
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        if not resolved.exists():
            raise FileNotFoundError(f"Runs directory not found: {resolved}")
        resolved_run_dirs.append(resolved)

    output = output_html.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    records = _collect_all(resolved_run_dirs, include_liras_code=include_liras_code)
    figures = _write_figures(records, output)
    _write_html(records, output, figures)
    if print_summary:
        print(json.dumps(_summary(records), indent=2, ensure_ascii=False))
    print(f"[OK] All-runs dashboard written: {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one dashboard aggregating multiple LIRAS run folders.")
    parser.add_argument("--runs-dir", action="append", default=None, help="Runs directory to scan; repeat for multiple folders")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary to stdout")
    parser.add_argument("--include-liras-code", action="store_true", help="Collect .LIRAs artifacts when available")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dirs = [Path(item) for item in (args.runs_dir or DEFAULT_RUN_DIRS)]
    build_dashboard(
        run_dirs,
        Path(args.output),
        include_liras_code=args.include_liras_code,
        print_summary=args.summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
