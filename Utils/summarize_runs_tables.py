#!/usr/bin/env python3

"""Summarize run metadata into tabular outputs.

Produces two CSV files:
1) runs_summary.csv      -> one row per run_metadata.json
2) iterations_summary.csv -> one row per iteration inside each run

Also prints a readable preview table in the terminal.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "Runs"
DEFAULT_OUTDIR = PROJECT_ROOT / "Report" / "Tables" / "RunSummaries"


def _iter_run_metadata_files(runs_root: Path) -> Iterable[Path]:
    return sorted(runs_root.glob("**/run_metadata.json"))


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(key, v, out)
        return

    if isinstance(value, list):
        # Keep nested arrays as JSON strings in run-level tables.
        out[prefix] = json.dumps(value, ensure_ascii=False)
        return

    out[prefix] = value


def _config_id_from_path(meta_path: Path, runs_root: Path) -> str:
    try:
        rel = meta_path.relative_to(runs_root)
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return "unknown"


def _to_bool_success(status: Any, iterations: List[dict]) -> bool:
    if str(status).lower() == "success":
        return True
    for it in iterations:
        if bool(it.get("is_valid")):
            return True
    return False


def _write_csv(path: Path, rows: List[Dict[str, Any]], preferred: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    keys = set()
    for row in rows:
        keys.update(row.keys())

    preferred_present = [k for k in preferred if k in keys]
    remaining = sorted(k for k in keys if k not in set(preferred_present))
    fieldnames = preferred_present + remaining

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_preview(rows: List[Dict[str, Any]], limit: int) -> None:
    if not rows:
        print("No runs found.")
        return

    columns = [
        "config_id",
        "scenario",
        "system_prompt",
        "shots",
        "generation_model",
        "status",
        "derived.iteration_count",
        "derived.success",
        "run_started_at",
        "run_finished_at",
    ]

    rows_view = rows[:limit]

    widths: Dict[str, int] = {}
    for col in columns:
        widths[col] = max(len(col), max(len(str(r.get(col, ""))) for r in rows_view))

    header = "  ".join(f"{c:<{widths[c]}}" for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)

    for row in rows_view:
        print("  ".join(f"{str(row.get(c, '')):<{widths[c]}}" for c in columns))

    if len(rows) > limit:
        print(f"\n... showing {limit} of {len(rows)} runs")


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _aggregate_table(run_rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in run_rows:
        key = tuple(str(row.get(k, "")) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        runs = len(rows)
        success_count = sum(1 for r in rows if str(r.get("derived.success")).lower() == "true")
        iterations = [_to_int(r.get("derived.iteration_count"), 0) for r in rows]
        durations = [_to_float(r.get("summary.run_duration_seconds"), 0.0) for r in rows]
        prompt_tokens = [_to_int(r.get("summary.total_prompt_tokens_est"), 0) for r in rows]
        response_tokens = [_to_int(r.get("summary.total_response_tokens_est"), 0) for r in rows]

        base: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        base.update(
            {
                "runs": runs,
                "success_count": success_count,
                "success_rate": round(success_count / runs, 4) if runs else 0.0,
                "iterations_avg": round(sum(iterations) / runs, 3) if runs else 0.0,
                "iterations_median": round(statistics.median(iterations), 3) if runs else 0.0,
                "iterations_max": max(iterations) if iterations else 0,
                "duration_seconds_avg": round(sum(durations) / runs, 3) if runs else 0.0,
                "prompt_tokens_total": sum(prompt_tokens),
                "response_tokens_total": sum(response_tokens),
            }
        )
        out.append(base)

    out.sort(key=lambda r: tuple(str(r.get(k, "")) for k in group_keys))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create tabular summaries from Runs/**/run_metadata.json"
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help=f"Directory containing run folders (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Output directory for CSV tables (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=30,
        help="How many rows to print in terminal preview (default: 30)",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir

    run_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []

    for meta_path in _iter_run_metadata_files(runs_root):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        iterations = meta.get("iterations") if isinstance(meta.get("iterations"), list) else []

        run_row: Dict[str, Any] = {}
        _flatten("", meta, run_row)
        run_row["config_id"] = _config_id_from_path(meta_path, runs_root)
        run_row["run_metadata_path"] = str(meta_path)
        run_row["derived.iteration_count"] = len(iterations)
        run_row["derived.success"] = _to_bool_success(meta.get("status"), iterations)
        run_rows.append(run_row)

        for idx, it in enumerate(iterations):
            row: Dict[str, Any] = {
                "config_id": run_row.get("config_id", "unknown"),
                "run_id": meta.get("run_id"),
                "scenario": meta.get("scenario"),
                "system_prompt": meta.get("system_prompt"),
                "shots": meta.get("shots"),
                "generation_model": meta.get("generation_model"),
                "repair_model": meta.get("repair_model"),
                "status": meta.get("status"),
                "iteration_index": idx,
                "run_metadata_path": str(meta_path),
            }
            _flatten("iteration", it, row)
            iter_rows.append(row)

    # Keep a deterministic order for preview readability.
    run_rows.sort(
        key=lambda r: (
            str(r.get("config_id", "")),
            str(r.get("scenario", "")),
            str(r.get("run_started_at", "")),
        )
    )

    runs_csv = outdir / "runs_summary.csv"
    iterations_csv = outdir / "iterations_summary.csv"
    by_config_csv = outdir / "table_by_config.csv"
    by_scenario_csv = outdir / "table_by_scenario.csv"
    by_model_prompt_shot_csv = outdir / "table_by_model_prompt_shot.csv"

    _write_csv(
        runs_csv,
        run_rows,
        preferred=[
            "config_id",
            "run_id",
            "scenario",
            "system_prompt",
            "shots",
            "generation_model",
            "repair_model",
            "status",
            "run_started_at",
            "run_finished_at",
            "derived.iteration_count",
            "derived.success",
            "run_metadata_path",
        ],
    )
    _write_csv(
        iterations_csv,
        iter_rows,
        preferred=[
            "config_id",
            "run_id",
            "scenario",
            "system_prompt",
            "shots",
            "generation_model",
            "repair_model",
            "status",
            "iteration_index",
            "iteration.iteration",
            "iteration.validated",
            "iteration.is_valid",
            "iteration.dsl_path",
            "iteration.compiler_output_path",
            "run_metadata_path",
        ],
    )

    table_by_config = _aggregate_table(run_rows, ["config_id"])
    table_by_scenario = _aggregate_table(run_rows, ["scenario"])
    table_by_mps = _aggregate_table(run_rows, ["generation_model", "system_prompt", "shots"])

    _write_csv(
        by_config_csv,
        table_by_config,
        preferred=[
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
    _write_csv(
        by_scenario_csv,
        table_by_scenario,
        preferred=[
            "scenario",
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
    _write_csv(
        by_model_prompt_shot_csv,
        table_by_mps,
        preferred=[
            "generation_model",
            "system_prompt",
            "shots",
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

    print(f"Runs scanned: {len(run_rows)}")
    print(f"Iteration rows: {len(iter_rows)}")
    print(f"Saved: {runs_csv}")
    print(f"Saved: {iterations_csv}\n")
    print(f"Saved: {by_config_csv}")
    print(f"Saved: {by_scenario_csv}")
    print(f"Saved: {by_model_prompt_shot_csv}\n")

    _print_preview(run_rows, limit=max(1, args.preview_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())