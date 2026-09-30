#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import build_model_runs_analysis as analysis


DEFAULT_OUTPUT = ROOT / "Report" / "feedback_comparison_dashboard.html"
MATRIX_ROWS = [
    ("zero", "Zero-shot"),
    ("few", "Few-shot"),
]
MATRIX_COLUMNS = [
    ("nocot", "Senza CoT"),
    ("cot", "Con CoT"),
]
DEFAULT_CASES = [
    {
        "key": "zero_nocot",
        "shot": "zero",
        "cot": "nocot",
        "label": "Zero-shot senza CoT",
        "runs_dir": "RunsNoCoT",
    },
    {
        "key": "zero_cot",
        "shot": "zero",
        "cot": "cot",
        "label": "Zero-shot con CoT",
        "runs_dir": "RunsCoT",
    },
    {
        "key": "few_nocot",
        "shot": "few",
        "cot": "nocot",
        "label": "Few-shot senza CoT",
        "runs_dir": "Runs2shotNoCot",
    },
    {
        "key": "few_cot",
        "shot": "few",
        "cot": "cot",
        "label": "Few-shot con CoT",
        "runs_dir": "Runs2ShotCoT",
    },
]
MODES = [
    ("actual", "Feedback loop attivi", "Risultato reale delle run, con feedback loop semantico e sintattico attivi."),
    ("no_semantic", "Senza feedback loop semantico", "Simula lo stop dopo il primo ciclo semantico."),
    (
        "no_syntactic",
        "Senza feedback loop sintattico",
        "Usa solo ITER0 e interrompe la run se il candidato iniziale non compila; i cicli semantici successivi non vengono considerati.",
    ),
    (
        "no_semantic_no_syntactic",
        "Senza feedback loop semantico e sintattico",
        "Combina entrambe le simulazioni: considera solo ITER0 del primo ciclo semantico.",
    ),
]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.1f}%"


def _fmt_int(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number:,.0f}"


def _fmt_float(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number:,.{digits}f}"


def _resolve(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def _case_records(case: dict[str, str]) -> list[dict[str, Any]]:
    runs_dir = _resolve(case["runs_dir"])
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found for {case['label']}: {runs_dir}")

    records: list[dict[str, Any]] = []
    for record in analysis._collect_records(runs_dir):
        enriched = dict(record)
        enriched["comparison_case"] = case["key"]
        enriched["comparison_label"] = case["label"]
        enriched["shot_case"] = case["shot"]
        enriched["cot_case"] = case["cot"]
        enriched["source_runs_dir"] = case["runs_dir"]
        records.append(enriched)
    return records


def _mode_records(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "actual":
        return [dict(record) for record in records]
    if mode == "no_semantic":
        return analysis._simulate_no_feedback_records(records)
    if mode == "no_syntactic":
        return analysis._simulate_no_syntactic_feedback_records(records)
    if mode == "no_semantic_no_syntactic":
        return analysis._simulate_no_syntactic_feedback_records(
            analysis._simulate_no_feedback_records(records)
        )
    raise ValueError(f"Unknown mode: {mode}")


def _token_total(records: list[dict[str, Any]]) -> float | None:
    total = 0.0
    found = False
    for record in records:
        value = _safe_float(record.get("llm_output_tokens"))
        if value is None:
            value = _safe_float(record.get("dsl_completion_tokens"))
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    success = sum(1 for record in records if record.get("outcome") == "success")
    failed = sum(1 for record in records if record.get("outcome") == "failed")
    token_total = _token_total(records)
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "success_rate": success / total if total else None,
        "output_tokens": token_total,
        "tokens_per_success": token_total / success if token_total is not None and success else None,
    }


def _build_payload(cases: list[dict[str, str]]) -> dict[str, Any]:
    records_by_case = {case["key"]: _case_records(case) for case in cases}
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    flat_rows: list[dict[str, Any]] = []

    for mode_key, mode_label, mode_description in MODES:
        table: dict[str, dict[str, Any]] = {}
        for case in cases:
            rows = _mode_records(records_by_case[case["key"]], mode_key)
            metrics = _metrics(rows)
            cell_key = f"{case['shot']}_{case['cot']}"
            table[cell_key] = {
                **metrics,
                "case": case["key"],
                "label": case["label"],
                "shot": case["shot"],
                "cot": case["cot"],
                "source_runs_dir": case["runs_dir"],
            }
            flat_rows.append(
                {
                    "mode": mode_key,
                    "mode_label": mode_label,
                    "mode_description": mode_description,
                    **table[cell_key],
                }
            )
        tables[mode_key] = table

    baseline = tables["actual"]
    for mode_key, table in tables.items():
        for cell_key, cell in table.items():
            base_rate = baseline.get(cell_key, {}).get("success_rate")
            rate = cell.get("success_rate")
            if base_rate is None or rate is None:
                delta = None
            else:
                delta = rate - base_rate
            cell["baseline_success_rate"] = base_rate
            cell["success_rate_delta"] = delta
    for row in flat_rows:
        cell_key = f"{row['shot']}_{row['cot']}"
        base_rate = baseline.get(cell_key, {}).get("success_rate")
        rate = row.get("success_rate")
        row["baseline_success_rate"] = base_rate
        row["success_rate_delta"] = None if base_rate is None or rate is None else rate - base_rate

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": cases,
        "modes": [
            {"key": key, "label": label, "description": description}
            for key, label, description in MODES
        ],
        "tables": tables,
        "rows": flat_rows,
    }


def _success_cell(cell: dict[str, Any]) -> str:
    delta = _safe_float(cell.get("success_rate_delta"))
    delta_html = ""
    if delta is not None and abs(delta) > 0.0000001:
        sign = "+" if delta > 0 else ""
        delta_html = f'<em class="delta">({sign}{delta * 100:.1f} pp)</em>'
    return (
        f"<strong>{html.escape(_fmt_pct(cell.get('success_rate')))}</strong>"
        f"{delta_html}"
        f"<span>{html.escape(_fmt_int(cell.get('success')))} / {html.escape(_fmt_int(cell.get('total')))}</span>"
    )


def _tokens_cell(cell: dict[str, Any]) -> str:
    return (
        f"<strong>{html.escape(_fmt_int(cell.get('tokens_per_success')))}</strong>"
        f"<span>{html.escape(_fmt_int(cell.get('output_tokens')))} token / "
        f"{html.escape(_fmt_int(cell.get('success')))} successi</span>"
    )


def _matrix_table(
    table: dict[str, dict[str, Any]],
    *,
    title: str,
    description: str,
    value_fn: Any,
) -> str:
    body_rows = []
    for row_key, row_label in MATRIX_ROWS:
        cells = [f"<th>{html.escape(row_label)}</th>"]
        for col_key, _col_label in MATRIX_COLUMNS:
            cell = table.get(f"{row_key}_{col_key}", {})
            cells.append(f"<td>{value_fn(cell)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
      <section class="panel">
        <div class="panel-head">
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(description)}</p>
        </div>
        <table class="matrix">
          <thead>
            <tr>
              <th>Shot / CoT</th>
              <th>Senza CoT</th>
              <th>Con CoT</th>
            </tr>
          </thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
      </section>
    """


def _detail_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['mode_label']))}</td>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{html.escape(str(row['source_runs_dir']))}</td>"
            f"<td>{html.escape(_fmt_int(row.get('total')))}</td>"
            f"<td>{html.escape(_fmt_int(row.get('success')))}</td>"
            f"<td>{html.escape(_fmt_int(row.get('failed')))}</td>"
            f"<td>{html.escape(_fmt_pct(row.get('success_rate')))}</td>"
            f"<td>{html.escape(_fmt_float((_safe_float(row.get('success_rate_delta')) or 0) * 100, 1) if row.get('success_rate_delta') is not None else '-')}</td>"
            f"<td>{html.escape(_fmt_int(row.get('tokens_per_success')))}</td>"
            "</tr>"
        )
    return (
        '<section class="panel wide"><div class="panel-head"><h2>Dettaglio numerico</h2>'
        '<p>Tutte le celle usate nelle matrici, con conteggi e token/successo.</p></div>'
        '<div class="table-scroll"><table><thead><tr>'
        '<th>Modalita</th><th>Caso</th><th>Cartella</th><th>Run</th><th>Successi</th>'
        '<th>Fallite</th><th>Success rate</th><th>Delta baseline pp</th><th>Token/successo</th>'
        '</tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div></section>"
    )


def _write_html(payload: dict[str, Any], output: Path) -> None:
    active_mode = payload["tables"]["actual"]
    total_runs = sum(cell["total"] for cell in active_mode.values())
    total_success = sum(cell["success"] for cell in active_mode.values())
    total_tokens = sum(
        float(cell["output_tokens"] or 0)
        for cell in active_mode.values()
        if cell.get("output_tokens") is not None
    )
    cards = [
        ("Run reali", _fmt_int(total_runs)),
        ("Successi reali", _fmt_int(total_success)),
        ("Success rate globale", _fmt_pct(total_success / total_runs if total_runs else None)),
        ("Token/successo globale", _fmt_int(total_tokens / total_success if total_success else None)),
    ]

    matrices = []
    for mode in payload["modes"]:
        matrices.append(
            _matrix_table(
                payload["tables"][mode["key"]],
                title=f"Success rate - {mode['label']}",
                description=mode["description"],
                value_fn=_success_cell,
            )
        )
    matrices.append(
        _matrix_table(
            payload["tables"]["actual"],
            title="Token/successo medio - feedback loop attivi",
            description="Output token generate/repair totali divisi per numero di successi; query adaptation e prompt token esclusi.",
            value_fn=_tokens_cell,
        )
    )

    output.write_text(
        f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Comparison Feedback Loop</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #64748b;
      --line: #dbe3ef;
      --blue: #2563eb;
      --red: #dc2626;
      --green: #15803d;
      --soft-blue: #eff6ff;
      --soft-red: #fef2f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Avenir Next", "Segoe UI", sans-serif;
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 22px; display: grid; gap: 16px; }}
    .hero {{
      background: linear-gradient(100deg, #172033 0%, #7f1d1d 100%);
      color: white;
      border-radius: 8px;
      padding: 22px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 0; color: #e2e8f0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }}
    .card {{ padding: 14px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    .value {{ font-size: 28px; font-weight: 800; margin-top: 5px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    @media (max-width: 950px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .panel {{ padding: 14px; display: grid; gap: 12px; }}
    .panel.wide {{ grid-column: 1 / -1; }}
    .panel-head h2 {{ margin: 0 0 4px; font-size: 18px; }}
    .panel-head p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; vertical-align: top; }}
    th {{ text-align: left; background: #f8fafc; color: #334155; }}
    .matrix th:not(:first-child), .matrix td {{ text-align: center; }}
    .matrix td strong {{ display: block; font-size: 26px; color: var(--green); }}
    .matrix td .delta {{ display: block; margin-top: 2px; color: var(--red); font-style: normal; font-weight: 800; }}
    .matrix td span {{ display: block; margin-top: 4px; color: var(--muted); font-variant-numeric: tabular-nums; }}
    .matrix tbody tr:nth-child(1) th {{ border-left: 4px solid var(--blue); }}
    .matrix tbody tr:nth-child(2) th {{ border-left: 4px solid var(--red); }}
    .table-scroll {{ overflow: auto; }}
    .table-scroll table {{ min-width: 980px; }}
    .table-scroll td:nth-child(n+4), .table-scroll th:nth-child(n+4) {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    code {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>Comparison feedback loop</h1>
      <p>Confronto tabellare tra zero-shot/few-shot e CoT/NoCoT. Generata il {html.escape(payload["generated_at"])}.</p>
    </section>
    <section class="cards">
      {''.join(f'<article class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></article>' for label, value in cards)}
    </section>
    <section class="grid">
      {''.join(matrices)}
      {_detail_table(payload["rows"])}
    </section>
  </main>
  <script type="application/json" id="comparison-payload">{html.escape(json.dumps(payload, ensure_ascii=False))}</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_dashboard(output: Path) -> None:
    resolved_output = output.expanduser()
    if not resolved_output.is_absolute():
        resolved_output = ROOT / resolved_output
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(DEFAULT_CASES)
    _write_html(payload, resolved_output)
    print(f"[OK] Comparison dashboard written: {resolved_output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 2x2 comparison dashboard for feedback-loop experiments.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    build_dashboard(Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
