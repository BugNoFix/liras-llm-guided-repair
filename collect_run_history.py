#!/usr/bin/env python3

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "Results"
DEFAULT_EXPORT_PATH = PROJECT_ROOT / "Report" / "run_history.csv"

_ERROR_RE = re.compile(r"\berror\b", re.IGNORECASE)
_WARNING_RE = re.compile(r"\bwarning\b", re.IGNORECASE)


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _stat_safe(path: Optional[Path]) -> Tuple[Optional[float], Optional[int]]:
    if path is None:
        return None, None
    try:
        if not path.exists() or not path.is_file():
            return None, None
        st = path.stat()
        return float(st.st_mtime), int(st.st_size)
    except Exception:
        return None, None


def _normalize_run_key(metadata: Dict[str, Any], metadata_path: Path) -> str:
    run_dir = metadata.get("run_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        return run_dir.strip()
    return str(metadata_path.parent)


def _iter_run_metadata_files(results_root: Path) -> Iterable[Path]:
    if not results_root.exists():
        return []
    return sorted(results_root.glob("**/run_metadata.json"))


def _compute_compiler_metrics(compiler_output_path: Optional[str]) -> Dict[str, Optional[int]]:
    if not compiler_output_path:
        return {
            "error_lines": None,
            "warning_lines": None,
            "score": None,
        }

    path = Path(compiler_output_path)
    if not path.exists() or not path.is_file():
        return {
            "error_lines": None,
            "warning_lines": None,
            "score": None,
        }

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {
            "error_lines": None,
            "warning_lines": None,
            "score": None,
        }

    error_lines = 0
    warning_lines = 0
    for line in text.splitlines():
        if _ERROR_RE.search(line):
            error_lines += 1
        if _WARNING_RE.search(line):
            warning_lines += 1

    score = (error_lines * 10) + (warning_lines * 2) + min(len(text), 2000) // 200

    return {
        "error_lines": int(error_lines),
        "warning_lines": int(warning_lines),
        "score": int(score),
    }


def _compute_run_derived_metrics(
    *,
    metadata: Dict[str, Any],
    iteration_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    scores: List[Optional[int]] = []
    errors: List[Optional[int]] = []
    warnings: List[Optional[int]] = []
    is_valids: List[Optional[int]] = []

    for r in sorted(iteration_rows, key=lambda x: int(x.get("iteration") or 0)):
        scores.append(_safe_int(r.get("compiler_error_score")))
        errors.append(_safe_int(r.get("compiler_error_lines")))
        warnings.append(_safe_int(r.get("compiler_warning_lines")))
        is_valids.append(_safe_int(r.get("is_valid")))

    n = len(scores)
    derived_iteration_count = n

    first_success_iter = None
    for idx, v in enumerate(is_valids):
        if v == 1:
            first_success_iter = idx
            break
    derived_success = 1 if first_success_iter is not None else 0

    initial_score = scores[0] if n >= 1 else None
    final_score = scores[-1] if n >= 1 else None
    final_error_lines = errors[-1] if n >= 1 else None
    final_warning_lines = warnings[-1] if n >= 1 else None

    best_score = None
    best_iter = None
    best_error_lines = None
    best_warning_lines = None
    for idx, s in enumerate(scores):
        if s is None:
            continue
        if best_score is None or s < best_score:
            best_score = s
            best_iter = idx
            best_error_lines = errors[idx]
            best_warning_lines = warnings[idx]

    auc = None
    if n >= 2 and all(s is not None for s in scores):
        auc_val = 0.0
        for i in range(n - 1):
            auc_val += (float(scores[i]) + float(scores[i + 1])) / 2.0
        auc = auc_val

    normalized_auc = None
    if auc is not None and isinstance(initial_score, int) and initial_score > 0 and n > 1:
        normalized_auc = float(auc) / float(initial_score * (n - 1))

    rel_improve_best = None
    if isinstance(initial_score, int) and initial_score > 0 and isinstance(best_score, int):
        rel_improve_best = float(initial_score - best_score) / float(initial_score)

    rel_improve_final = None
    if isinstance(initial_score, int) and initial_score > 0 and isinstance(final_score, int):
        rel_improve_final = float(initial_score - final_score) / float(initial_score)

    improvement_steps = 0
    regression_steps = 0
    deltas: List[float] = []
    if n >= 2 and all(s is not None for s in scores):
        for i in range(1, n):
            d = float(scores[i - 1]) - float(scores[i])
            deltas.append(d)
            if d > 0:
                improvement_steps += 1
            elif d < 0:
                regression_steps += 1

    monotonicity_ratio = None
    avg_delta = None
    if n >= 2 and deltas:
        monotonicity_ratio = float(improvement_steps) / float(n - 1)
        avg_delta = float(sum(deltas)) / float(len(deltas))

    started = _parse_iso_datetime(_safe_str(metadata.get("run_started_at")))
    finished = _parse_iso_datetime(_safe_str(metadata.get("run_finished_at")))
    run_duration_seconds = None
    if started and finished:
        run_duration_seconds = (finished - started).total_seconds()

    llm_span_seconds = None
    calls = metadata.get("llm_call_history")
    if isinstance(calls, list):
        call_times: List[datetime] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            t = _parse_iso_datetime(_safe_str(call.get("timestamp")))
            if t:
                call_times.append(t)
        if len(call_times) >= 2:
            call_times.sort()
            llm_span_seconds = (call_times[-1] - call_times[0]).total_seconds()

    return {
        "derived_iteration_count": _safe_int(derived_iteration_count),
        "derived_success": _safe_int(derived_success),
        "derived_first_success_iteration": _safe_int(first_success_iter),
        "derived_iterations_to_best": _safe_int(best_iter),
        "derived_best_error_score": _safe_int(best_score),
        "derived_best_error_lines": _safe_int(best_error_lines),
        "derived_best_warning_lines": _safe_int(best_warning_lines),
        "derived_final_error_score": _safe_int(final_score),
        "derived_final_error_lines": _safe_int(final_error_lines),
        "derived_final_warning_lines": _safe_int(final_warning_lines),
        "derived_initial_error_score": _safe_int(initial_score),
        "derived_auc_error_score": _safe_float(auc),
        "derived_normalized_auc_error_score": _safe_float(normalized_auc),
        "derived_relative_improvement_best": _safe_float(rel_improve_best),
        "derived_relative_improvement_final": _safe_float(rel_improve_final),
        "derived_improvement_steps": _safe_int(improvement_steps),
        "derived_regression_steps": _safe_int(regression_steps),
        "derived_monotonicity_ratio": _safe_float(monotonicity_ratio),
        "derived_avg_delta_per_step": _safe_float(avg_delta),
        "derived_run_duration_seconds": _safe_float(run_duration_seconds),
        "derived_llm_span_seconds": _safe_float(llm_span_seconds),
    }


def _extract_run_row(metadata: Dict[str, Any], *, run_key: str, source_path: str) -> Dict[str, Any]:
    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    telemetry = metadata.get("telemetry") if isinstance(metadata.get("telemetry"), dict) else {}

    shots_value = metadata.get("shots")
    shots_json = json.dumps(shots_value, ensure_ascii=True) if shots_value is not None else None
    shots_count = None
    if isinstance(shots_value, int):
        shots_count = shots_value
    elif isinstance(shots_value, list):
        shots_count = len(shots_value)

    repair_shots_value = metadata.get("repair_shots")
    repair_shots_json = json.dumps(repair_shots_value, ensure_ascii=True) if repair_shots_value is not None else None

    return {
        "run_key": run_key,
        "run_id": _safe_str(metadata.get("run_id")),
        "run_started_at": _safe_str(metadata.get("run_started_at")),
        "run_finished_at": _safe_str(metadata.get("run_finished_at")),
        "status": _safe_str(metadata.get("status")),
        "project_id": _safe_str(metadata.get("project_id")),
        "location": _safe_str(metadata.get("location")),
        "system_prompt": _safe_str(metadata.get("system_prompt")),
        "scenario": _safe_str(metadata.get("scenario")),
        "shots_json": shots_json,
        "shots_count": _safe_int(shots_count),
        "repair_prompt": _safe_str(metadata.get("repair_prompt")),
        "generation_model": _safe_str(metadata.get("generation_model")),
        "generation_temperature": _safe_float(metadata.get("generation_temperature")),
        "repair_model": _safe_str(metadata.get("repair_model")),
        "repair_temperature": _safe_float(metadata.get("repair_temperature")),
        "repair_shots_json": repair_shots_json,
        "max_iterations": _safe_int(metadata.get("max_iterations")),
        "compiler_jar": _safe_str(metadata.get("compiler_jar")),
        "use_generated_dsl_cache": _safe_int(metadata.get("use_generated_dsl_cache")),
        "generated_dsl_source": _safe_str(metadata.get("generated_dsl_source")),
        "generated_dsl_cache_path": _safe_str(metadata.get("generated_dsl_cache_path")),
        "run_dir": _safe_str(metadata.get("run_dir")),
        "dsl_dir": _safe_str(metadata.get("dsl_dir")),
        "compiler_dir": _safe_str(metadata.get("compiler_dir")),
        "summary_iterations_recorded": _safe_int(summary.get("iterations_recorded")),
        "summary_validations_attempted": _safe_int(summary.get("validations_attempted")),
        "summary_compiler_failures": _safe_int(summary.get("compiler_failures")),
        "summary_compiler_successes": _safe_int(summary.get("compiler_successes")),
        "summary_setup_errors": _safe_int(summary.get("setup_errors")),
        "summary_first_success_iteration": _safe_int(summary.get("first_success_iteration")),
        "summary_final_success_dsl_path": _safe_str(summary.get("final_success_dsl_path")),
        "summary_total_compiler_feedback_chars": _safe_int(summary.get("total_compiler_feedback_chars")),
        "summary_llm_calls": _safe_int(summary.get("llm_calls")),
        "summary_prompt_tokens_est_total": _safe_int(summary.get("prompt_tokens_est_total")),
        "summary_response_tokens_est_total": _safe_int(summary.get("response_tokens_est_total")),
        "telemetry_llm_calls": _safe_int(telemetry.get("llm_calls")),
        "telemetry_prompt_chars_total": _safe_int(telemetry.get("prompt_chars_total")),
        "telemetry_response_chars_total": _safe_int(telemetry.get("response_chars_total")),
        "telemetry_prompt_tokens_est_total": _safe_int(telemetry.get("prompt_tokens_est_total")),
        "telemetry_response_tokens_est_total": _safe_int(telemetry.get("response_tokens_est_total")),
        "source_path": source_path,
        "ingested_at": _utcnow_iso(),
    }


def _extract_iteration_rows(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    iterations = metadata.get("iterations")
    if not isinstance(iterations, list):
        return rows

    for idx, it in enumerate(iterations):
        if not isinstance(it, dict):
            continue

        iteration_num = it.get("iteration")
        if iteration_num is None:
            iteration_num = idx

        metrics = _compute_compiler_metrics(_safe_str(it.get("compiler_output_path")))

        rows.append(
            {
                "iteration": _safe_int(iteration_num),
                "dsl_path": _safe_str(it.get("dsl_path")),
                "compiler_output_path": _safe_str(it.get("compiler_output_path")),
                "validated": _safe_int(it.get("validated")),
                "is_valid": _safe_int(it.get("is_valid")),
                "compiler_feedback_chars": _safe_int(it.get("compiler_feedback_chars")),
                "compiler_error_lines": metrics.get("error_lines"),
                "compiler_warning_lines": metrics.get("warning_lines"),
                "compiler_error_score": metrics.get("score"),
                "repair_prompt_included_previous_dsl": _safe_int(
                    it.get("repair_prompt_included_previous_dsl")
                ),
                "repair_prompt_mode": _safe_str(it.get("repair_prompt_mode")),
            }
        )

    return rows


@dataclass
class ExportStats:
    scanned: int = 0
    parsed: int = 0
    rows_written: int = 0
    errors: int = 0


def export_csv(results_root: Path, out_path: Path, *, verbose: bool = False) -> ExportStats:
    stats = ExportStats()
    rows: List[Dict[str, Any]] = []

    for meta_path in _iter_run_metadata_files(results_root):
        stats.scanned += 1
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                stats.errors += 1
                if verbose:
                    print(f"WARN: metadata not an object: {meta_path}")
                continue

            run_key = _normalize_run_key(metadata, meta_path)
            run_row = _extract_run_row(metadata, run_key=run_key, source_path=str(meta_path))
            iter_rows = _extract_iteration_rows(metadata)
            derived = _compute_run_derived_metrics(metadata=metadata, iteration_rows=iter_rows)
            run_row.update(derived)

            if not iter_rows:
                row = dict(run_row)
                row.update({"iteration": None})
                rows.append(row)
            else:
                for it_row in iter_rows:
                    row = dict(run_row)
                    row.update(it_row)
                    rows.append(row)

            stats.parsed += 1
            if verbose:
                print(f"OK: parsed {meta_path}")
        except Exception as exc:
            stats.errors += 1
            if verbose:
                print(f"ERROR: failed {meta_path}: {exc}")

    fieldnames = [
        "run_key",
        "run_id",
        "run_started_at",
        "run_finished_at",
        "status",
        "project_id",
        "location",
        "system_prompt",
        "scenario",
        "shots_json",
        "shots_count",
        "repair_prompt",
        "generation_model",
        "generation_temperature",
        "repair_model",
        "repair_temperature",
        "repair_shots_json",
        "max_iterations",
        "compiler_jar",
        "use_generated_dsl_cache",
        "generated_dsl_source",
        "generated_dsl_cache_path",
        "run_dir",
        "dsl_dir",
        "compiler_dir",
        "summary_iterations_recorded",
        "summary_validations_attempted",
        "summary_compiler_failures",
        "summary_compiler_successes",
        "summary_setup_errors",
        "summary_first_success_iteration",
        "summary_final_success_dsl_path",
        "summary_total_compiler_feedback_chars",
        "summary_llm_calls",
        "summary_prompt_tokens_est_total",
        "summary_response_tokens_est_total",
        "telemetry_llm_calls",
        "telemetry_prompt_chars_total",
        "telemetry_response_chars_total",
        "telemetry_prompt_tokens_est_total",
        "telemetry_response_tokens_est_total",
        "derived_iteration_count",
        "derived_success",
        "derived_first_success_iteration",
        "derived_iterations_to_best",
        "derived_best_error_score",
        "derived_best_error_lines",
        "derived_best_warning_lines",
        "derived_final_error_score",
        "derived_final_error_lines",
        "derived_final_warning_lines",
        "derived_initial_error_score",
        "derived_auc_error_score",
        "derived_normalized_auc_error_score",
        "derived_relative_improvement_best",
        "derived_relative_improvement_final",
        "derived_improvement_steps",
        "derived_regression_steps",
        "derived_monotonicity_ratio",
        "derived_avg_delta_per_step",
        "derived_run_duration_seconds",
        "derived_llm_span_seconds",
        "iteration",
        "dsl_path",
        "compiler_output_path",
        "validated",
        "is_valid",
        "compiler_feedback_chars",
        "compiler_error_lines",
        "compiler_warning_lines",
        "compiler_error_score",
        "repair_prompt_included_previous_dsl",
        "repair_prompt_mode",
        "source_path",
        "ingested_at",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        stats.rows_written = len(rows)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Results/Runs/**/run_metadata.json and write a single CSV with run + iteration data. "
            "This overwrites the CSV each run so you always have one canonical source."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root directory to scan (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help=f"CSV output path (default: {DEFAULT_EXPORT_PATH})",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-file ingest logs")

    args = parser.parse_args()

    stats = export_csv(args.results_root, args.out, verbose=args.verbose)
    print("\n=== Run history export complete ===")
    print(f"Scanned:     {stats.scanned}")
    print(f"Parsed:      {stats.parsed}")
    print(f"Rows written:{stats.rows_written}")
    print(f"Errors:      {stats.errors}")
    print(f"CSV: {args.out}")


if __name__ == "__main__":
    main()