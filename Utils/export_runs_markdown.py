#!/usr/bin/env python3

"""Export run information to a single markdown file.

Reads the CSV outputs produced by Utils/summarize_runs_tables.py and writes one
consolidated markdown report with aggregated tables and run-level details.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLES_DIR = PROJECT_ROOT / "Report" / "Tables" / "RunSummaries"
DEFAULT_OUT = DEFAULT_TABLES_DIR / "run_report.md"


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _md_table(rows: List[dict], columns: Iterable[str]) -> str:
    cols = list(columns)
    if not rows:
        return "_No data available._\n"

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for r in rows:
        vals = []
        for c in cols:
            v = str(r.get(c, ""))
            v = v.replace("|", "\\|").replace("\n", " ").strip()
            vals.append(v)
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body) + "\n"


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one markdown file with all run info")
    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_TABLES_DIR),
        help=f"Directory containing summary CSV files (default: {DEFAULT_TABLES_DIR})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output markdown file path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=500,
        help="Max number of run-detail rows to include in markdown (default: 500)",
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    if not tables_dir.is_absolute():
        tables_dir = PROJECT_ROOT / tables_dir
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    runs = _read_csv(tables_dir / "runs_summary.csv")
    by_config = _read_csv(tables_dir / "table_by_config.csv")
    by_scenario = _read_csv(tables_dir / "table_by_scenario.csv")
    by_mps = _read_csv(tables_dir / "table_by_model_prompt_shot.csv")

    total_runs = len(runs)
    success_runs = sum(1 for r in runs if str(r.get("derived.success", "")).lower() == "true")
    success_rate = (success_runs / total_runs) if total_runs else 0.0
    total_iterations = sum(_to_int(r.get("derived.iteration_count", "0")) for r in runs)
    avg_iterations = (total_iterations / total_runs) if total_runs else 0.0

    total_prompt_tokens = sum(_to_int(r.get("summary.total_prompt_tokens_est", "0")) for r in runs)
    total_response_tokens = sum(_to_int(r.get("summary.total_response_tokens_est", "0")) for r in runs)
    total_duration = sum(_to_float(r.get("summary.run_duration_seconds", "0")) for r in runs)

    lines: List[str] = []
    lines.append("# Run Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("## Global Summary")
    lines.append("")
    lines.append(f"- Total runs: {total_runs}")
    lines.append(f"- Successful runs: {success_runs}")
    lines.append(f"- Success rate: {success_rate:.2%}")
    lines.append(f"- Total iterations: {total_iterations}")
    lines.append(f"- Average iterations per run: {avg_iterations:.3f}")
    lines.append(f"- Total prompt tokens (est.): {total_prompt_tokens}")
    lines.append(f"- Total response tokens (est.): {total_response_tokens}")
    lines.append(f"- Total run duration (s): {total_duration:.3f}")
    lines.append("")

    lines.append("## Table By Configuration")
    lines.append("")
    lines.append(
        _md_table(
            by_config,
            [
                "config_id",
                "runs",
                "success_count",
                "success_rate",
                "iterations_avg",
                "iterations_median",
                "iterations_max",
                "duration_seconds_avg",
                "prompt_tokens_total",
                "response_tokens_total",
            ],
        )
    )

    lines.append("## Table By Scenario")
    lines.append("")
    lines.append(
        _md_table(
            by_scenario,
            [
                "scenario",
                "runs",
                "success_count",
                "success_rate",
                "iterations_avg",
                "iterations_median",
                "iterations_max",
                "duration_seconds_avg",
            ],
        )
    )

    lines.append("## Table By Model Prompt Shot")
    lines.append("")
    lines.append(
        _md_table(
            by_mps,
            [
                "generation_model",
                "system_prompt",
                "shots",
                "runs",
                "success_count",
                "success_rate",
                "iterations_avg",
                "iterations_median",
                "iterations_max",
            ],
        )
    )

    lines.append("## Run Details")
    lines.append("")
    run_detail_rows = runs[: max(0, args.max_runs)]
    lines.append(
        _md_table(
            run_detail_rows,
            [
                "config_id",
                "run_id",
                "scenario",
                "system_prompt",
                "shots",
                "generation_model",
                "status",
                "derived.iteration_count",
                "derived.success",
                "run_started_at",
                "run_finished_at",
                "run_metadata_path",
            ],
        )
    )

    if len(runs) > len(run_detail_rows):
        lines.append(
            f"_Run details truncated to {len(run_detail_rows)} rows (total available: {len(runs)})._"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())