#!/usr/bin/env python3

import argparse
import json
import sqlite3
import re
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "Results" / "Runs"
DEFAULT_DB_PATH = PROJECT_ROOT / "Report" / "run_history.sqlite"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "Report"


def _resolve_optional_path(value: Any, *, relative_to: Path) -> Optional[Path]:
    if not value:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        p = Path(text)
        if p.is_absolute():
            return p
        return relative_to / text
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


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    # Handle trailing Z
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


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value)
    except Exception:
        return None
    return text


def _get_nested(dct: Dict[str, Any], *keys: str) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _normalize_run_key(metadata: Dict[str, Any], metadata_path: Path) -> str:
    # Prefer the run_dir stored inside the metadata (absolute path, unique per run).
    run_dir = metadata.get("run_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        return run_dir.strip()
    return str(metadata_path.parent)


def _iter_run_metadata_files(results_root: Path) -> Iterable[Path]:
    if not results_root.exists():
        return []
    # Expected layout: Results/Runs/<Scenario>/<SystemPrompt>/RUN_<id>/run_metadata.json
    return sorted(results_root.glob("**/run_metadata.json"))


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_key TEXT PRIMARY KEY,
            run_id TEXT,
            run_started_at TEXT,
            run_finished_at TEXT,
            status TEXT,

            project_id TEXT,
            location TEXT,

            system_prompt TEXT,
            scenario TEXT,
            shots_json TEXT,
            shots_count INTEGER,

            repair_prompt TEXT,
            generation_model TEXT,
            generation_temperature REAL,
            repair_model TEXT,
            repair_temperature REAL,
            repair_shots_json TEXT,
            max_iterations INTEGER,
            compiler_jar TEXT,

            run_dir TEXT,
            dsl_dir TEXT,
            compiler_dir TEXT,

            updated_at TEXT,

            summary_iterations_recorded INTEGER,
            summary_validations_attempted INTEGER,
            summary_compiler_failures INTEGER,
            summary_compiler_successes INTEGER,
            summary_setup_errors INTEGER,
            summary_first_success_iteration INTEGER,
            summary_final_success_dsl_path TEXT,
            summary_total_compiler_feedback_chars INTEGER,
            summary_llm_calls INTEGER,
            summary_prompt_tokens_est_total INTEGER,
            summary_response_tokens_est_total INTEGER,

            telemetry_llm_calls INTEGER,
            telemetry_prompt_chars_total INTEGER,
            telemetry_response_chars_total INTEGER,
            telemetry_prompt_tokens_est_total INTEGER,
            telemetry_response_tokens_est_total INTEGER,

            -- Prompt/scenario identifiers (no prompt text stored)
            system_prompt_resolved_path TEXT,
            scenario_resolved_path TEXT,
            repair_prompt_resolved_path TEXT,
            system_prompt_mtime REAL,
            system_prompt_size INTEGER,
            scenario_mtime REAL,
            scenario_size INTEGER,
            repair_prompt_mtime REAL,
            repair_prompt_size INTEGER,

            -- Derived / analysis-friendly run metrics (computed by collector)
            derived_iteration_count INTEGER,
            derived_success INTEGER,
            derived_first_success_iteration INTEGER,
            derived_iterations_to_best INTEGER,
            derived_best_error_score INTEGER,
            derived_best_error_lines INTEGER,
            derived_best_warning_lines INTEGER,
            derived_final_error_score INTEGER,
            derived_final_error_lines INTEGER,
            derived_final_warning_lines INTEGER,
            derived_initial_error_score INTEGER,
            derived_auc_error_score REAL,
            derived_normalized_auc_error_score REAL,
            derived_relative_improvement_best REAL,
            derived_relative_improvement_final REAL,
            derived_improvement_steps INTEGER,
            derived_regression_steps INTEGER,
            derived_monotonicity_ratio REAL,
            derived_avg_delta_per_step REAL,
            derived_run_duration_seconds REAL,
            derived_llm_span_seconds REAL,

            -- Collector control flags
            analysis_skipped INTEGER,
            analysis_skip_reason TEXT,

            source_path TEXT,
            source_mtime REAL,
            source_size INTEGER,
            ingested_at TEXT
        );

        CREATE TABLE IF NOT EXISTS iterations (
            run_key TEXT NOT NULL,
            iteration INTEGER NOT NULL,

            dsl_path TEXT,
            compiler_output_path TEXT,
            validated INTEGER,
            is_valid INTEGER,

            validation_returncode INTEGER,
            validation_timed_out INTEGER,
            validation_java_missing INTEGER,
            validation_jar_missing INTEGER,
            validation_dsl_missing INTEGER,

            compiler_feedback_chars INTEGER,
            compiler_error_lines INTEGER,
            compiler_warning_lines INTEGER,
            compiler_error_score INTEGER,
            compiler_output_mtime REAL,
            compiler_output_size INTEGER,
            repair_prompt_included_previous_dsl INTEGER,
            repair_prompt_mode TEXT,

            ingested_at TEXT,

            PRIMARY KEY (run_key, iteration),
            FOREIGN KEY (run_key) REFERENCES runs(run_key) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS llm_calls (
            run_key TEXT NOT NULL,
            call_index INTEGER NOT NULL,
            kind TEXT,
            timestamp TEXT,
            prompt_chars INTEGER,
            response_chars INTEGER,
            prompt_tokens_est INTEGER,
            response_tokens_est INTEGER,
            ingested_at TEXT,
            PRIMARY KEY (run_key, call_index),
            FOREIGN KEY (run_key) REFERENCES runs(run_key) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_runs_run_id ON runs(run_id);
        CREATE INDEX IF NOT EXISTS idx_runs_scenario ON runs(scenario);
        CREATE INDEX IF NOT EXISTS idx_runs_system_prompt ON runs(system_prompt);
        CREATE INDEX IF NOT EXISTS idx_iter_valid ON iterations(is_valid);
        """
    )

    # Lightweight migrations for existing DBs.
    # SQLite doesn't support ALTER TABLE ... ADD COLUMN IF NOT EXISTS everywhere.
    for col_def in [
        ("compiler_error_lines", "INTEGER"),
        ("compiler_warning_lines", "INTEGER"),
        ("compiler_error_score", "INTEGER"),
        ("compiler_output_mtime", "REAL"),
        ("compiler_output_size", "INTEGER"),
    ]:
        col, typ = col_def
        try:
            conn.execute(f"ALTER TABLE iterations ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            # Column likely already exists
            pass

    for col_def in [
        ("generation_temperature", "REAL"),
        ("repair_temperature", "REAL"),
        ("derived_iteration_count", "INTEGER"),
        ("derived_success", "INTEGER"),
        ("derived_first_success_iteration", "INTEGER"),
        ("derived_iterations_to_best", "INTEGER"),
        ("derived_best_error_score", "INTEGER"),
        ("derived_best_error_lines", "INTEGER"),
        ("derived_best_warning_lines", "INTEGER"),
        ("derived_final_error_score", "INTEGER"),
        ("derived_final_error_lines", "INTEGER"),
        ("derived_final_warning_lines", "INTEGER"),
        ("derived_initial_error_score", "INTEGER"),
        ("derived_auc_error_score", "REAL"),
        ("derived_normalized_auc_error_score", "REAL"),
        ("derived_relative_improvement_best", "REAL"),
        ("derived_relative_improvement_final", "REAL"),
        ("derived_improvement_steps", "INTEGER"),
        ("derived_regression_steps", "INTEGER"),
        ("derived_monotonicity_ratio", "REAL"),
        ("derived_avg_delta_per_step", "REAL"),
        ("derived_run_duration_seconds", "REAL"),
        ("derived_llm_span_seconds", "REAL"),
        ("analysis_skipped", "INTEGER"),
        ("analysis_skip_reason", "TEXT"),
    ]:
        col, typ = col_def
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass

    for col_def in [
        ("system_prompt_resolved_path", "TEXT"),
        ("scenario_resolved_path", "TEXT"),
        ("repair_prompt_resolved_path", "TEXT"),
        ("system_prompt_mtime", "REAL"),
        ("system_prompt_size", "INTEGER"),
        ("scenario_mtime", "REAL"),
        ("scenario_size", "INTEGER"),
        ("repair_prompt_mtime", "REAL"),
        ("repair_prompt_size", "INTEGER"),
    ]:
        col, typ = col_def
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass

    # Convenience views for analysis.
    conn.executescript(
        """
        CREATE VIEW IF NOT EXISTS v_runs AS
        SELECT
            run_key,
            run_id,
            run_started_at,
            run_finished_at,
            status,
            project_id,
            location,
            scenario,
            system_prompt,
            scenario_resolved_path,
            system_prompt_resolved_path,
            generation_model,
            generation_temperature,
            repair_model,
            repair_temperature,
            shots_count,
            max_iterations,

            repair_prompt,
            repair_prompt_resolved_path,

            -- Provenance of prompt/scenario files (no prompt text stored)
            system_prompt_mtime,
            system_prompt_size,
            scenario_mtime,
            scenario_size,
            repair_prompt_mtime,
            repair_prompt_size,

            shots_json,
            repair_shots_json,

            derived_iteration_count,
            derived_success,
            derived_first_success_iteration,
            derived_iterations_to_best,
            derived_best_error_score,
            derived_final_error_score,
            derived_initial_error_score,
            derived_auc_error_score,
            derived_normalized_auc_error_score,
            derived_relative_improvement_best,
            derived_relative_improvement_final,
            derived_monotonicity_ratio,
            derived_regression_steps,
            derived_run_duration_seconds,
            derived_llm_span_seconds,
            analysis_skipped,
            analysis_skip_reason,
            summary_llm_calls,
            summary_prompt_tokens_est_total,
            summary_response_tokens_est_total,
            source_path,
            source_mtime,
            source_size,
            ingested_at
        FROM runs;

        CREATE VIEW IF NOT EXISTS v_iterations AS
        SELECT
            i.run_key,
            r.run_id,
            r.scenario,
            r.system_prompt,
            r.generation_model,
            r.repair_model,
            i.iteration,
            i.validated,
            i.is_valid,
            i.compiler_error_lines,
            i.compiler_warning_lines,
            i.compiler_error_score,
            i.compiler_feedback_chars,
            i.compiler_output_path,
            i.dsl_path
        FROM iterations i
        JOIN runs r ON r.run_key = i.run_key;
        """
    )

    conn.commit()


def _skip_reason_for_run(metadata: Dict[str, Any]) -> Optional[str]:
    """Return a reason to skip analysis for interrupted/broken runs."""
    status = metadata.get("status")
    if isinstance(status, str):
        s = status.strip().lower()
        if s in {"setup_error", "crashed", "error", "config_error"}:
            return f"status={s}"

    if metadata.get("interrupted") is True:
        return "interrupted=true"

    if isinstance(metadata.get("breaking_error"), dict):
        be_type = metadata.get("breaking_error", {}).get("type")
        if be_type:
            return f"breaking_error={be_type}"
        return "breaking_error"

    return None


def _should_skip(conn: sqlite3.Connection, *, source_path: str, mtime: float, size: int) -> bool:
    row = conn.execute(
        "SELECT source_mtime, source_size FROM runs WHERE source_path = ?", (source_path,)
    ).fetchone()
    if row is None:
        return False
    prev_mtime = row["source_mtime"]
    prev_size = row["source_size"]
    return (prev_mtime == mtime) and (prev_size == size)


def _upsert_run(conn: sqlite3.Connection, run_row: Dict[str, Any]) -> None:
    cols = sorted(run_row.keys())
    placeholders = ",".join(["?"] * len(cols))
    assignments = ",".join([f"{c}=excluded.{c}" for c in cols if c != "run_key"])
    sql = (
        f"INSERT INTO runs ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(run_key) DO UPDATE SET {assignments}"
    )
    conn.execute(sql, [run_row[c] for c in cols])


def _upsert_iteration(conn: sqlite3.Connection, it_row: Dict[str, Any]) -> None:
    cols = sorted(it_row.keys())
    placeholders = ",".join(["?"] * len(cols))
    assignments = ",".join([f"{c}=excluded.{c}" for c in cols if c not in ("run_key", "iteration")])
    sql = (
        f"INSERT INTO iterations ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(run_key, iteration) DO UPDATE SET {assignments}"
    )
    conn.execute(sql, [it_row[c] for c in cols])


def _upsert_llm_call(conn: sqlite3.Connection, call_row: Dict[str, Any]) -> None:
    cols = sorted(call_row.keys())
    placeholders = ",".join(["?"] * len(cols))
    assignments = ",".join([f"{c}=excluded.{c}" for c in cols if c not in ("run_key", "call_index")])
    sql = (
        f"INSERT INTO llm_calls ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(run_key, call_index) DO UPDATE SET {assignments}"
    )
    conn.execute(sql, [call_row[c] for c in cols])


def _extract_run_row(metadata: Dict[str, Any], *, run_key: str, source_path: str, mtime: float, size: int) -> Dict[str, Any]:
    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    telemetry = metadata.get("telemetry") if isinstance(metadata.get("telemetry"), dict) else {}

    shots_value = metadata.get("shots")
    shots_json = json.dumps(shots_value, ensure_ascii=False) if shots_value is not None else None
    shots_count = None
    if isinstance(shots_value, int):
        shots_count = shots_value
    elif isinstance(shots_value, list):
        shots_count = len(shots_value)

    repair_shots_value = metadata.get("repair_shots")
    repair_shots_json = json.dumps(repair_shots_value, ensure_ascii=False) if repair_shots_value is not None else None

    # Prompt/scenario identifiers (store names + resolved paths; never store prompt text)
    system_prompt_path = _resolve_optional_path(metadata.get("system_prompt"), relative_to=PROJECT_ROOT / "SPs")
    scenario_path = _resolve_optional_path(metadata.get("scenario"), relative_to=PROJECT_ROOT / "Scenarios")

    repair_prompt_path = None
    if isinstance(metadata.get("repair_prompt"), str) and str(metadata.get("repair_prompt")).strip():
        repair_prompt_path = Path(str(metadata.get("repair_prompt")).strip())
        if not repair_prompt_path.is_absolute():
            repair_prompt_path = PROJECT_ROOT / repair_prompt_path

    system_prompt_mtime, system_prompt_size = _stat_safe(system_prompt_path)
    scenario_mtime, scenario_size = _stat_safe(scenario_path)
    repair_prompt_mtime, repair_prompt_size = _stat_safe(repair_prompt_path)

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
        "generation_temperature": None if metadata.get("generation_temperature") is None else float(metadata.get("generation_temperature")),
        "repair_model": _safe_str(metadata.get("repair_model")),
        "repair_temperature": None if metadata.get("repair_temperature") is None else float(metadata.get("repair_temperature")),
        "repair_shots_json": repair_shots_json,
        "max_iterations": _safe_int(metadata.get("max_iterations")),
        "compiler_jar": _safe_str(metadata.get("compiler_jar")),

        "run_dir": _safe_str(metadata.get("run_dir")),
        "dsl_dir": _safe_str(metadata.get("dsl_dir")),
        "compiler_dir": _safe_str(metadata.get("compiler_dir")),

        "updated_at": _safe_str(metadata.get("updated_at")),

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

        "system_prompt_resolved_path": str(system_prompt_path) if system_prompt_path else None,
        "scenario_resolved_path": str(scenario_path) if scenario_path else None,
        "repair_prompt_resolved_path": str(repair_prompt_path) if repair_prompt_path else None,
        "system_prompt_mtime": system_prompt_mtime,
        "system_prompt_size": system_prompt_size,
        "scenario_mtime": scenario_mtime,
        "scenario_size": scenario_size,
        "repair_prompt_mtime": repair_prompt_mtime,
        "repair_prompt_size": repair_prompt_size,

        "source_path": source_path,
        "source_mtime": float(mtime),
        "source_size": int(size),
        "ingested_at": _utcnow_iso(),
    }


def _write_query_to_csv(conn: sqlite3.Connection, query: str, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.execute(query)
    columns = [d[0] for d in (cur.description or [])]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in cur:
            writer.writerow([row[c] for c in columns])


def _extract_iteration_rows(metadata: Dict[str, Any], *, run_key: str) -> List[Dict[str, Any]]:
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

        validation = it.get("validation") if isinstance(it.get("validation"), dict) else {}

        rows.append(
            {
                "run_key": run_key,
                "iteration": _safe_int(iteration_num) or 0,
                "dsl_path": _safe_str(it.get("dsl_path")),
                "compiler_output_path": _safe_str(it.get("compiler_output_path")),
                "validated": _safe_int(it.get("validated")),
                "is_valid": _safe_int(it.get("is_valid")),
                "validation_returncode": _safe_int(validation.get("returncode")),
                "validation_timed_out": _safe_int(validation.get("timed_out")),
                "validation_java_missing": _safe_int(validation.get("java_missing")),
                "validation_jar_missing": _safe_int(validation.get("jar_missing")),
                "validation_dsl_missing": _safe_int(validation.get("dsl_missing")),
                "compiler_feedback_chars": _safe_int(it.get("compiler_feedback_chars")),
                "compiler_error_lines": None,
                "compiler_warning_lines": None,
                "compiler_error_score": None,
                "compiler_output_mtime": None,
                "compiler_output_size": None,
                "repair_prompt_included_previous_dsl": _safe_int(it.get("repair_prompt_included_previous_dsl")),
                "repair_prompt_mode": _safe_str(it.get("repair_prompt_mode")),
                "ingested_at": _utcnow_iso(),
            }
        )

    return rows


def _extract_llm_call_rows(metadata: Dict[str, Any], *, run_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    calls = metadata.get("llm_call_history")
    if not isinstance(calls, list):
        return rows

    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        rows.append(
            {
                "run_key": run_key,
                "call_index": idx,
                "kind": _safe_str(call.get("kind")),
                "timestamp": _safe_str(call.get("timestamp")),
                "prompt_chars": _safe_int(call.get("prompt_chars")),
                "response_chars": _safe_int(call.get("response_chars")),
                "prompt_tokens_est": _safe_int(call.get("prompt_tokens_est")),
                "response_tokens_est": _safe_int(call.get("response_tokens_est")),
                "ingested_at": _utcnow_iso(),
            }
        )

    return rows


@dataclass
class IngestStats:
    scanned: int = 0
    skipped_unchanged: int = 0
    parsed: int = 0
    upserted_runs: int = 0
    upserted_iterations: int = 0
    upserted_llm_calls: int = 0
    computed_iteration_metrics: int = 0
    errors: int = 0


_ERROR_RE = re.compile(r"\berror\b", re.IGNORECASE)
_WARNING_RE = re.compile(r"\bwarning\b", re.IGNORECASE)


def _compute_compiler_metrics(compiler_output_path: Optional[str]) -> Dict[str, Optional[int]]:
    """Compute error/warning line counts from a compiler output file.

    Heuristic: counts lines containing whole-word 'error' and 'warning' (case-insensitive).
    """
    if not compiler_output_path:
        return {
            "error_lines": None,
            "warning_lines": None,
            "score": None,
            "mtime": None,
            "size": None,
        }

    path = Path(compiler_output_path)
    if not path.exists() or not path.is_file():
        return {
            "error_lines": None,
            "warning_lines": None,
            "score": None,
            "mtime": None,
            "size": None,
        }

    try:
        st = path.stat()
    except Exception:
        st = None

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {
            "error_lines": None,
            "warning_lines": None,
            "score": None,
            "mtime": float(getattr(st, "st_mtime", 0.0)) if st else None,
            "size": int(getattr(st, "st_size", 0)) if st else None,
        }

    error_lines = 0
    warning_lines = 0
    for line in text.splitlines():
        if _ERROR_RE.search(line):
            error_lines += 1
        if _WARNING_RE.search(line):
            warning_lines += 1

    # Same weighting idea as the generator script: errors dominate.
    score = (error_lines * 10) + (warning_lines * 2) + min(len(text), 2000) // 200

    return {
        "error_lines": int(error_lines),
        "warning_lines": int(warning_lines),
        "score": int(score),
        "mtime": float(getattr(st, "st_mtime", 0.0)) if st else None,
        "size": int(getattr(st, "st_size", 0)) if st else None,
    }


def _compute_run_derived_metrics(
    *,
    metadata: Dict[str, Any],
    iteration_rows: List[Dict[str, Any]],
    llm_call_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Iteration sequence (scores)
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

    # Success metrics
    first_success_iter = None
    for idx, v in enumerate(is_valids):
        if v == 1:
            first_success_iter = idx
            break
    derived_success = 1 if first_success_iter is not None else 0

    # Best/final/initial
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

    # AUC (trapezoidal, step=1). Only if we have at least 2 numeric points.
    numeric_scores = [s for s in scores if isinstance(s, int)]
    auc = None
    if n >= 2 and all(s is not None for s in scores):
        auc_val = 0.0
        for i in range(n - 1):
            auc_val += (float(scores[i]) + float(scores[i + 1])) / 2.0
        auc = auc_val

    normalized_auc = None
    if auc is not None and isinstance(initial_score, int) and initial_score > 0 and n > 1:
        # Normalize by a baseline rectangle: initial_score * (n-1)
        normalized_auc = float(auc) / float(initial_score * (n - 1))

    rel_improve_best = None
    if isinstance(initial_score, int) and initial_score > 0 and isinstance(best_score, int):
        rel_improve_best = float(initial_score - best_score) / float(initial_score)

    rel_improve_final = None
    if isinstance(initial_score, int) and initial_score > 0 and isinstance(final_score, int):
        rel_improve_final = float(initial_score - final_score) / float(initial_score)

    # Stability / monotonicity: stepwise deltas
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

    # Run duration (wall clock)
    started = _parse_iso_datetime(_safe_str(metadata.get("run_started_at")))
    finished = _parse_iso_datetime(_safe_str(metadata.get("run_finished_at")))
    run_duration_seconds = None
    if started and finished:
        run_duration_seconds = (finished - started).total_seconds()

    # LLM span: between first and last call timestamps (if present)
    llm_span_seconds = None
    call_times: List[datetime] = []
    for row in llm_call_rows:
        t = _parse_iso_datetime(_safe_str(row.get("timestamp")))
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
        "derived_auc_error_score": float(auc) if auc is not None else None,
        "derived_normalized_auc_error_score": float(normalized_auc) if normalized_auc is not None else None,
        "derived_relative_improvement_best": float(rel_improve_best) if rel_improve_best is not None else None,
        "derived_relative_improvement_final": float(rel_improve_final) if rel_improve_final is not None else None,
        "derived_improvement_steps": _safe_int(improvement_steps),
        "derived_regression_steps": _safe_int(regression_steps),
        "derived_monotonicity_ratio": float(monotonicity_ratio) if monotonicity_ratio is not None else None,
        "derived_avg_delta_per_step": float(avg_delta) if avg_delta is not None else None,
        "derived_run_duration_seconds": float(run_duration_seconds) if run_duration_seconds is not None else None,
        "derived_llm_span_seconds": float(llm_span_seconds) if llm_span_seconds is not None else None,
    }


def ingest(
    results_root: Path,
    db_path: Path,
    *,
    verbose: bool = False,
    recompute_metrics: bool = False,
) -> IngestStats:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        stats = IngestStats()

        for meta_path in _iter_run_metadata_files(results_root):
            stats.scanned += 1
            try:
                st = meta_path.stat()
                source_path = str(meta_path)
                if (not recompute_metrics) and _should_skip(
                    conn, source_path=source_path, mtime=st.st_mtime, size=st.st_size
                ):
                    stats.skipped_unchanged += 1
                    continue

                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if not isinstance(metadata, dict):
                    stats.errors += 1
                    if verbose:
                        print(f"WARN: metadata not an object: {meta_path}")
                    continue

                run_key = _normalize_run_key(metadata, meta_path)

                run_row = _extract_run_row(
                    metadata,
                    run_key=run_key,
                    source_path=source_path,
                    mtime=st.st_mtime,
                    size=st.st_size,
                )

                skip_reason = _skip_reason_for_run(metadata)
                if skip_reason:
                    # Record run but skip iteration metrics/derived metrics so analysis data stays clean.
                    run_row["analysis_skipped"] = 1
                    run_row["analysis_skip_reason"] = skip_reason
                    _upsert_run(conn, run_row)
                    stats.upserted_runs += 1
                    conn.commit()
                    stats.parsed += 1
                    if verbose:
                        print(f"SKIP: interrupted run ({skip_reason}): {meta_path}")
                    continue

                run_row["analysis_skipped"] = 0
                run_row["analysis_skip_reason"] = None

                it_rows = _extract_iteration_rows(metadata, run_key=run_key)
                for r in it_rows:
                    metrics = _compute_compiler_metrics(r.get("compiler_output_path"))
                    r["compiler_error_lines"] = metrics["error_lines"]
                    r["compiler_warning_lines"] = metrics["warning_lines"]
                    r["compiler_error_score"] = metrics["score"]
                    r["compiler_output_mtime"] = metrics["mtime"]
                    r["compiler_output_size"] = metrics["size"]
                    if metrics["error_lines"] is not None or metrics["warning_lines"] is not None:
                        stats.computed_iteration_metrics += 1
                    _upsert_iteration(conn, r)
                stats.upserted_iterations += len(it_rows)

                call_rows = _extract_llm_call_rows(metadata, run_key=run_key)
                for r in call_rows:
                    _upsert_llm_call(conn, r)
                stats.upserted_llm_calls += len(call_rows)

                # Derived metrics (run-level), computed from iteration + call rows
                derived = _compute_run_derived_metrics(
                    metadata=metadata,
                    iteration_rows=it_rows,
                    llm_call_rows=call_rows,
                )
                run_row.update(derived)

                _upsert_run(conn, run_row)
                stats.upserted_runs += 1

                conn.commit()
                stats.parsed += 1

                if verbose:
                    print(f"OK: ingested {meta_path}")

            except Exception as e:
                conn.rollback()
                stats.errors += 1
                if verbose:
                    print(f"ERROR: failed {meta_path}: {e}")

        return stats
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Results/Runs/**/run_metadata.json and incrementally upsert into a single SQLite file. "
            "Re-running only ingests new/updated metadata files."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root directory to scan (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite output path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-file ingest logs")
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help=(
            "Recompute per-iteration compiler metrics (error/warning counts) even if the "
            "run_metadata.json file is unchanged. Useful for backfilling existing DBs after "
            "upgrading the collector."
        ),
    )

    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help=f"Directory to write CSV exports (default: {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--export-runs-csv",
        action="store_true",
        help="Write Report/run_history_runs.csv with one row per run",
    )
    parser.add_argument(
        "--export-iterations-csv",
        action="store_true",
        help="Write Report/run_history_iterations.csv with one row per iteration",
    )

    args = parser.parse_args()

    stats = ingest(
        args.results_root,
        args.db,
        verbose=args.verbose,
        recompute_metrics=args.recompute_metrics,
    )
    print("\n=== Run history ingest complete ===")
    print(f"Scanned:           {stats.scanned}")
    print(f"Skipped unchanged: {stats.skipped_unchanged}")
    print(f"Parsed:            {stats.parsed}")
    print(f"Upserted runs:     {stats.upserted_runs}")
    print(f"Upserted iterations:{stats.upserted_iterations}")
    print(f"Upserted llm_calls:{stats.upserted_llm_calls}")
    print(f"Iteration metrics computed: {stats.computed_iteration_metrics}")
    print(f"Errors:            {stats.errors}")
    print(f"DB: {args.db}")

    if args.export_runs_csv or args.export_iterations_csv:
        conn = _connect(args.db)
        try:
            if args.export_runs_csv:
                out = args.export_dir / "run_history_runs.csv"
                _write_query_to_csv(
                    conn,
                    """
                    SELECT
                        *
                    FROM v_runs
                    ORDER BY
                        COALESCE(run_started_at, ''),
                        COALESCE(run_id, ''),
                        run_key
                    """.strip(),
                    out,
                )
                print(f"Runs CSV: {out}")

            if args.export_iterations_csv:
                out = args.export_dir / "run_history_iterations.csv"
                _write_query_to_csv(
                    conn,
                    """
                    SELECT
                        *
                    FROM v_iterations
                    ORDER BY
                        COALESCE(run_id, ''),
                        run_key,
                        iteration
                    """.strip(),
                    out,
                )
                print(f"Iterations CSV: {out}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
