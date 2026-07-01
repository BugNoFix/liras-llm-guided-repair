#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_path(raw: str, *, must_exist: bool = True) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if must_exist and not candidate.exists():
        raise FileNotFoundError(str(candidate))
    return candidate


def _coerce_positive_int(raw, *, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _coerce_probability(raw) -> Optional[float]:
    if raw is None:
        return None
    value = str(raw).strip().rstrip(".;,")
    if not value:
        return None
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1].strip()
    try:
        parsed = float(value)
    except ValueError:
        return None
    if is_percent:
        parsed /= 100.0
    if parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _probability_threshold(config: dict) -> float:
    raw = config.get("verifyta_probability_delta_threshold", 0.05)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.05
    return max(0.0, min(1.0, value))


def _results_base_from_config(config: dict) -> Path:
    cfg_results_dir = config.get("results_dir")
    if cfg_results_dir and str(cfg_results_dir).strip():
        results_base = Path(str(cfg_results_dir).strip()).expanduser()
        if not results_base.is_absolute():
            results_base = PROJECT_ROOT / results_base
        return results_base
    return PROJECT_ROOT / "Results"


def _default_runs_root(config: dict) -> Path:
    scenario_name = str(config["scenario"]).replace(".txt", "")
    sp_name = str(config["system_prompt"]).replace(".txt", "")
    return _results_base_from_config(config) / scenario_name / sp_name


def _create_outer_run_dir(config: dict) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outer_run_dir = _default_runs_root(config) / f"RUN_{run_id}"
    suffix = 1
    while outer_run_dir.exists():
        outer_run_dir = _default_runs_root(config) / f"RUN_{run_id}_{suffix}"
        suffix += 1
    outer_run_dir.mkdir(parents=True, exist_ok=True)
    return outer_run_dir


def _init_global_pipeline_metadata(config: dict, outer_run_dir: Path, max_cycles: int) -> Path:
    metadata_path = outer_run_dir / "run_metadata.json"
    metadata = {
        "run_id": outer_run_dir.name.replace("RUN_", "", 1),
        "run_started_at": datetime.now().isoformat(),
        "pipeline_runner_version": "v3_structured_stage_metadata",
        "overall_result": "running",
        "failed_stage": None,
        "failure_type": None,
        "failure_reason": None,
        "failure_details": None,
        "generation_provider": config.get("generation_provider"),
        "repair_provider": config.get("repair_provider"),
        "query_provider": config.get("query_provider"),
        "system_prompt": config.get("system_prompt"),
        "repair_prompt": config.get("repair_prompt"),
        "scenario": config.get("scenario"),
        "generation_model": config.get("generation_model"),
        "repair_model": config.get("repair_model"),
        "shots": config.get("shots"),
        "repair_shots": config.get("repair_shots"),
        "max_iterations": config.get("max_iterations"),
        "max_uppaal_feedback_cycles": max_cycles,
        "run_dir": str(outer_run_dir),
        "cycles": [],
        "run_finished_at": None,
    }
    _save_json(metadata_path, metadata)
    return metadata_path


def _append_global_cycle_metadata(metadata_path: Path, cycle_record: dict) -> None:
    metadata = _load_json(metadata_path)
    cycles = metadata.setdefault("cycles", [])
    cycles.append(cycle_record)
    metadata["updated_at"] = datetime.now().isoformat()
    _save_json(metadata_path, metadata)


def _finalize_global_pipeline_metadata(metadata_path: Path, updates: dict) -> None:
    metadata = _load_json(metadata_path)
    metadata.update(updates)
    metadata["run_finished_at"] = datetime.now().isoformat()
    metadata["updated_at"] = datetime.now().isoformat()
    _save_json(metadata_path, metadata)


def _resolve_executable(raw: str) -> str:
    """Resolve executable from absolute/relative path or PATH command name."""
    value = (raw or "").strip()
    if not value:
        raise ValueError("Empty executable name")

    # Absolute paths or values containing path separators are treated as file paths.
    if "/" in value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if not candidate.exists():
            raise FileNotFoundError(str(candidate))
        return str(candidate)

    # Plain command names are resolved through PATH.
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(
            f"Executable '{value}' not found on PATH"
        )
    return resolved


def _resolve_verifyta_command(config: dict) -> str:
    """Resolve verifyta executable with explicit config and sensible fallbacks."""
    configured = config.get("verifyta_bin")
    if isinstance(configured, str) and configured.strip():
        return _resolve_executable(configured.strip())

    # Preferred fallback when a local verifyta binary/script is shipped in the repo.
    local_verifyta = PROJECT_ROOT / "verifyta"
    if local_verifyta.exists():
        return str(local_verifyta)

    resolved = shutil.which("verifyta")
    if resolved:
        return resolved

    raise FileNotFoundError(
        "verifyta executable not found. Set 'verifyta_bin' in config.json "
        "(absolute path, relative path, or command name on PATH)."
    )


def _validate_pipeline_config(config: dict) -> None:
    required_liras = ("generation_provider", "generation_model", "shots", "system_prompt", "scenario")
    missing_liras = [k for k in required_liras if k not in config]
    if missing_liras:
        raise ValueError(f"config.json missing required LIRAS keys: {missing_liras}")

    if not bool(config.get("generation_only", False)) and "repair_provider" not in config:
        raise ValueError("'repair_provider' is required when generation_only=false")

    supported_providers = ("gemini", "groq", "mistral", "openrouter", "huggingface")
    for key in ("generation_provider", "repair_provider", "query_provider"):
        if key in config:
            value = str(config.get(key) or "").strip().lower()
            if value not in supported_providers:
                raise ValueError(f"'{key}' must be one of: {', '.join(supported_providers)}")

    enable_xml_export = bool(config.get("enable_xml_export", True))
    if enable_xml_export:
        jar = config.get("lira_cli_jar")
        if not isinstance(jar, str) or not jar.strip():
            raise ValueError("'lira_cli_jar' is required when enable_xml_export=true")

    if "verifyta_bin" in config and config.get("verifyta_bin") is not None:
        val = config.get("verifyta_bin")
        if not isinstance(val, str):
            raise ValueError("'verifyta_bin' must be a string when provided")

    if bool(config.get("enable_query_adaptation", False)):
        from query_adapter import validate_query_config

        if "query_provider" not in config:
            raise ValueError("'query_provider' is required when enable_query_adaptation=true")
        validate_query_config(config, PROJECT_ROOT)

    if "max_uppaal_feedback_cycles" in config:
        try:
            max_cycles = int(config.get("max_uppaal_feedback_cycles"))
        except (TypeError, ValueError):
            raise ValueError("'max_uppaal_feedback_cycles' must be an integer")
        if max_cycles < 1:
            raise ValueError("'max_uppaal_feedback_cycles' must be >= 1")

    if "verifyta_probability_delta_threshold" in config:
        try:
            threshold = float(config.get("verifyta_probability_delta_threshold"))
        except (TypeError, ValueError):
            raise ValueError("'verifyta_probability_delta_threshold' must be a number")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("'verifyta_probability_delta_threshold' must be between 0 and 1")


def _read_run_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    return _load_json(metadata_path)


def _write_run_metadata(metadata_path: Path, metadata: dict) -> None:
    if not metadata_path.exists():
        return
    _save_json(metadata_path, metadata)


def _update_pipeline_metadata(metadata_path: Path, updates: dict) -> None:
    metadata = _read_run_metadata(metadata_path)
    metadata.update(updates)
    _write_run_metadata(metadata_path, metadata)


def _compact_dict(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if value is not None}


def _stage_record(
    stage: str,
    result: str,
    *,
    execution_result: Optional[str] = None,
    semantic_result: Optional[str] = None,
    failure_type: Optional[str] = None,
    failure_reason: Optional[str] = None,
    failure_details: Optional[dict] = None,
    details: Optional[dict] = None,
    artifacts: Optional[dict] = None,
) -> dict:
    return _compact_dict(
        {
            "stage": stage,
            "result": result,
            "execution_result": execution_result,
            "semantic_result": semantic_result,
            "failure_type": failure_type,
            "failure_reason": failure_reason,
            "failure_details": failure_details,
            "details": details,
            "artifacts": artifacts,
        }
    )


def _upsert_stage(stages: list[dict], stage_record: dict) -> list[dict]:
    stage_name = stage_record.get("stage")
    updated: list[dict] = []
    replaced = False
    for existing in stages:
        if isinstance(existing, dict) and existing.get("stage") == stage_name:
            updated.append(stage_record)
            replaced = True
        else:
            updated.append(existing)
    if not replaced:
        updated.append(stage_record)
    return updated


def _update_stage_metadata(metadata_path: Path, stage_record: dict, *, cycle_result: Optional[str] = None) -> None:
    metadata = _read_run_metadata(metadata_path)
    stages = metadata.get("stages")
    if not isinstance(stages, list):
        stages = []
    metadata["stages"] = _upsert_stage(stages, stage_record)
    if cycle_result is not None:
        metadata["cycle_result"] = cycle_result
    if stage_record.get("result") == "failed":
        metadata.update(
            {
                "cycle_result": "failed",
                "failed_stage": stage_record.get("stage"),
                "failure_type": stage_record.get("failure_type"),
                "failure_reason": stage_record.get("failure_reason"),
                "failure_details": stage_record.get("failure_details"),
            }
        )
    metadata["updated_at"] = datetime.now().isoformat()
    _write_run_metadata(metadata_path, metadata)


def _failure_payload(stage_record: dict) -> dict:
    return {
        "failed_stage": stage_record.get("stage"),
        "failure_type": stage_record.get("failure_type"),
        "failure_reason": stage_record.get("failure_reason"),
        "failure_details": stage_record.get("failure_details"),
    }


def _stage_artifacts(**paths: Optional[object]) -> dict:
    return {key: str(value) for key, value in paths.items() if value is not None}


def _dsl_generation_stage_from_metadata(run_metadata: dict, success_liras_path: Optional[Path]) -> dict:
    artifacts = _stage_artifacts(
        selected_liras_path=success_liras_path,
        prompt_log_path=run_metadata.get("prompt_log"),
        response_log_path=run_metadata.get("response_log"),
    )
    if success_liras_path is not None:
        return _stage_record(
            "dsl_generation",
            "ok",
            details=run_metadata.get("summary") if isinstance(run_metadata.get("summary"), dict) else None,
            artifacts=artifacts,
        )

    status = str(run_metadata.get("status") or "failed")
    summary = run_metadata.get("summary") if isinstance(run_metadata.get("summary"), dict) else {}
    breaking_error = run_metadata.get("breaking_error") if isinstance(run_metadata.get("breaking_error"), dict) else {}
    if status == "max_iterations_reached":
        failure_type = "max_iterations_reached"
        reason = "DSL generation reached the maximum number of repair iterations without producing a valid LIRAs model."
    elif status == "setup_error":
        failure_type = "configuration"
        reason = str(breaking_error.get("message") or "DSL generation failed because validation could not be configured.")
    elif status == "crashed":
        failure_type = "execution"
        reason = str(breaking_error.get("message") or "DSL generation crashed before producing a valid LIRAs model.")
    else:
        failure_type = "missing_artifact"
        reason = "DSL generation finished without producing a valid LIRAs artifact."

    return _stage_record(
        "dsl_generation",
        "failed",
        failure_type=failure_type,
        failure_reason=reason,
        failure_details=_compact_dict(
            {
                "generator_status": status,
                "max_iterations": run_metadata.get("max_iterations"),
                "compiler_failures": summary.get("compiler_failures"),
                "compiler_successes": summary.get("compiler_successes"),
                "error_type": breaking_error.get("type"),
                "error_message": breaking_error.get("message"),
            }
        ),
        details=summary or None,
        artifacts=artifacts,
    )


def _find_success_liras_path(run_metadata: dict, run_dir: Path) -> Optional[Path]:
    summary = run_metadata.get("summary") or {}
    final_success = summary.get("final_success_dsl_path")
    if isinstance(final_success, str) and final_success.strip():
        candidate = Path(final_success)
        if candidate.exists():
            return candidate

    dsl_dir = run_dir / "dsl"
    if not dsl_dir.exists():
        return None
    success_files = sorted(dsl_dir.glob("SUCCESS_*.LIRAs"))
    if success_files:
        return success_files[-1]
    return None


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _read_text_if_exists(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_adapted_query_file(query_path: Optional[Path]) -> dict[int, dict]:
    if query_path is None or not query_path.exists():
        return {}

    entries: dict[int, dict] = {}
    pending_comments: list[str] = []
    pending_expected_probability: Optional[float] = None
    query_index = 0
    for line_no, raw_line in enumerate(query_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            comment = stripped[2:].strip()
            expected = _extract_expected_probability(comment)
            if expected is not None:
                pending_expected_probability = expected
            else:
                pending_comments.append(comment)
            continue

        query_index += 1
        entries[query_index] = {
            "index": query_index,
            "line": line_no,
            "description": " ".join(c for c in pending_comments if c).strip(),
            "adapted_formula": stripped,
            "expected_probability": pending_expected_probability,
        }
        pending_comments = []
        pending_expected_probability = None

    return entries


def _extract_expected_probability(text: str) -> Optional[float]:
    match = re.search(
        r"(?:expected|ground\s*truth)\s*(?:probability|prob)\s*:?\s*\**\s*([0-9]+(?:\.[0-9]+)?%?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _coerce_probability(match.group(1))


def _parse_source_query_file(source_query_path: Optional[Path]) -> dict[int, dict]:
    if source_query_path is None or not source_query_path.exists():
        return {}

    text = source_query_path.read_text(encoding="utf-8", errors="replace")
    result: dict[int, dict] = {}
    matches = list(re.finditer(r"^###\s+Query\s+(\d+)\s*$", text, flags=re.MULTILINE | re.IGNORECASE))
    for pos, match in enumerate(matches):
        index = int(match.group(1))
        start = match.end()
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
        block = text[start:end]

        desc_match = re.search(r"\*\*Description:\*\*\s*(.+)", block)
        formula_match = re.search(r"\*\*Original formula:\*\*\s*(.+)", block)
        expected_match = re.search(
            r"\*\*(?:Expected|Ground truth)\s*(?:probability|prob):\*\*\s*(.+)",
            block,
            flags=re.IGNORECASE,
        )
        if expected_match is None:
            expected_match = re.search(
                r"(?:Expected|Ground truth)\s*(?:probability|prob)\s*:?\s*([0-9]+(?:\.[0-9]+)?%?)",
                block,
                flags=re.IGNORECASE,
            )
        result[index] = {
            "description": desc_match.group(1).strip() if desc_match else "",
            "source_formula": formula_match.group(1).strip() if formula_match else "",
            "expected_probability": _coerce_probability(expected_match.group(1)) if expected_match else None,
        }
    return result


def _has_verifyta_error_stderr(stderr: str) -> bool:
    for line in stderr.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "[warning]" in stripped.lower():
            continue
        return True
    return False


def _filter_verifyta_warning_lines(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if "[warning]" in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _first_error_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _classify_verifyta_error(stderr: str, exit_code: Optional[int] = None) -> str:
    text = (stderr or "").lower()
    if exit_code == 124 or "timed out" in text:
        return "timeout"
    if "has no member named" in text:
        return "invalid_model_reference"
    if "boolean expected" in text or "type error" in text:
        return "invalid_query"
    if "syntax error" in text or "parse error" in text:
        return "invalid_query"
    if "no such file" in text or "not found" in text:
        return "missing_artifact"
    return "execution"


def _verifyta_execution_reason(stage: str, failure_type: str, stderr: str) -> str:
    phase = "internal verification" if stage == "verifyta_internal" else "adapted-query verification"
    first_error = _first_error_line(stderr)
    if failure_type == "timeout":
        return f"UPPAAL verifyta timed out during {phase}."
    if failure_type == "invalid_model_reference":
        return f"UPPAAL verifyta failed during {phase} because a query references a model member that does not exist."
    if failure_type == "invalid_query":
        return f"UPPAAL verifyta failed during {phase} because a query is not valid for the generated model."
    if failure_type == "missing_artifact":
        return f"UPPAAL verifyta failed during {phase} because a required file or executable is missing."
    if first_error:
        return f"UPPAAL verifyta failed during {phase}: {first_error}"
    return f"UPPAAL verifyta failed before completing {phase}."


def _semantic_failure_reason(stage: str, analysis: dict) -> str:
    phase = "internal UPPAAL verification" if stage == "verifyta_internal" else "adapted-query UPPAAL verification"
    count = int(analysis.get("failed_query_count") or 0)
    if count == 1:
        return f"{phase} completed, but 1 property/query was not satisfied."
    return f"{phase} completed, but {count} properties/queries were not satisfied."


def _query_failure_reason(failure_kind: str) -> str:
    if failure_kind == "probability_delta":
        return "The obtained probability differs from the expected probability by more than the configured threshold."
    if failure_kind == "not_satisfied":
        return "The property/query was evaluated by UPPAAL but was not satisfied."
    if failure_kind == "aborted":
        return "UPPAAL aborted while evaluating this property/query."
    if failure_kind == "verifyta_error":
        return "UPPAAL reported an error while evaluating this property/query."
    return "The property/query failed verification."


def _parse_verifyta_probability(block: str) -> dict:
    interval_match = re.search(
        r"Pr\([^)]*\)\s+in\s+\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]",
        block,
    )
    if interval_match:
        lower = float(interval_match.group(1))
        upper = float(interval_match.group(2))
        return {
            "obtained_probability": (lower + upper) / 2.0,
            "probability_interval_lower": lower,
            "probability_interval_upper": upper,
        }

    value_match = re.search(r"Pr\([^)]*\)\s*=\s*([0-9.eE+-]+)", block)
    if value_match:
        value = float(value_match.group(1))
        return {
            "obtained_probability": value,
            "probability_interval_lower": value,
            "probability_interval_upper": value,
        }

    return {}


def _query_context(index: int, adapted_queries: dict[int, dict], source_queries: dict[int, dict]) -> dict:
    adapted = adapted_queries.get(index, {})
    source = source_queries.get(index, {})
    expected_probability = source.get("expected_probability")
    if expected_probability is None:
        expected_probability = adapted.get("expected_probability")
    return {
        "index": index,
        "query_line": adapted.get("line"),
        "description": source.get("description") or adapted.get("description") or "",
        "source_formula": source.get("source_formula") or "",
        "adapted_formula": adapted.get("adapted_formula") or "",
        "expected_probability": expected_probability,
    }


def _parse_verifyta_failures(
    *,
    stdout_path: Optional[Path],
    stderr_path: Optional[Path] = None,
    adapted_query_path: Optional[Path],
    source_query_path: Optional[Path],
    probability_threshold: float = 0.05,
) -> dict:
    stdout = _strip_ansi(_read_text_if_exists(stdout_path))
    stderr = _strip_ansi(_read_text_if_exists(stderr_path))
    stderr_without_warnings = _filter_verifyta_warning_lines(stderr)
    adapted_queries = _parse_adapted_query_file(adapted_query_path)
    source_queries = _parse_source_query_file(source_query_path)

    failed_queries: list[dict] = []
    probability_checked_queries: list[dict] = []
    verified_formula_count = 0
    last_formula_index = None
    blocks = list(
        re.finditer(
            r"Verifying formula\s+(\d+).*?(?=^Verifying formula\s+\d+|\Z)",
            stdout,
            flags=re.DOTALL | re.MULTILINE,
        )
    )

    for match in blocks:
        verified_formula_count += 1
        index = int(match.group(1))
        last_formula_index = index
        block = match.group(0).strip()
        lowered = block.lower()
        is_not_satisfied = "not satisfied" in lowered or "isn't satisfied" in lowered
        is_aborted = "-- aborted" in lowered or "aborted." in lowered
        query = _query_context(index, adapted_queries, source_queries)
        probability = _parse_verifyta_probability(block)
        expected_probability = query.get("expected_probability")
        probability_delta = None
        probability_failure = False
        failure_kind = None

        if expected_probability is not None and probability.get("obtained_probability") is not None:
            probability_delta = abs(float(probability["obtained_probability"]) - float(expected_probability))
            probability_failure = probability_delta > probability_threshold
            probability_checked_queries.append(
                {
                    **query,
                    **probability,
                    "probability_delta": probability_delta,
                    "probability_threshold": probability_threshold,
                    "probability_status": "failed" if probability_failure else "ok",
                }
            )

        if is_aborted:
            failure_kind = "aborted"
            probability_delta = 1.0 if expected_probability is not None else probability_delta
        elif is_not_satisfied:
            failure_kind = "not_satisfied"
            probability_delta = 1.0 if expected_probability is not None else probability_delta
        elif probability_failure:
            failure_kind = "probability_delta"
        else:
            continue

        failed_queries.append(
            {
                **query,
                **probability,
                "failure_kind": failure_kind,
                "failure_reason": _query_failure_reason(failure_kind or ""),
                "probability_delta": probability_delta,
                "probability_threshold": probability_threshold if expected_probability is not None else None,
                "verifyta_output": "\n".join(
                    part for part in (block[-4000:], stderr_without_warnings[-4000:]) if part
                ).strip(),
            }
        )

    if not failed_queries and _has_verifyta_error_stderr(stderr):
        query = (
            _query_context(last_formula_index, adapted_queries, source_queries)
            if last_formula_index is not None
            else {"index": None, "query_line": None, "description": "", "source_formula": "", "adapted_formula": ""}
        )
        failed_queries.append(
            {
                **query,
                "failure_kind": "verifyta_error",
                "failure_reason": _query_failure_reason("verifyta_error"),
                "probability_delta": 1.0 if query.get("expected_probability") is not None else None,
                "probability_threshold": probability_threshold if query.get("expected_probability") is not None else None,
                "verifyta_output": stderr_without_warnings[-4000:],
            }
        )

    return {
        "status": "failed" if failed_queries else "ok",
        "verified_formula_count": verified_formula_count,
        "failed_query_count": len(failed_queries),
        "failed_queries": failed_queries,
        "probability_threshold": probability_threshold,
        "probability_checked_query_count": len(probability_checked_queries),
        "probability_checked_queries": probability_checked_queries,
        "stdout_path": str(stdout_path) if stdout_path else None,
        "stderr_path": str(stderr_path) if stderr_path else None,
        "adapted_query_path": str(adapted_query_path) if adapted_query_path else None,
        "source_query_path": str(source_query_path) if source_query_path else None,
        "failure_reason": _semantic_failure_reason("verifyta_adapted", {"failed_query_count": len(failed_queries)})
        if failed_queries
        else None,
    }


def _build_uppaal_feedback_text(
    *,
    selected_liras_path: Path,
    internal_stage: dict,
    adapted_stage: dict,
    internal_analysis: dict,
    adapted_analysis: dict,
    run_dir: Path,
) -> str:
    parts: list[str] = []
    parts.append("### WHAT TO FIX")
    failed_stage = internal_stage if internal_stage.get("result") != "ok" else adapted_stage
    failure_type = failed_stage.get("failure_type")
    if failure_type == "invalid_model_reference":
        parts.append("The adapted queries reference names that do not exist in the generated UPPAAL XML. Regenerate the LIRAs model so exported XML names match the intended scenario, or adapt the queries to the actual XML identifiers.")
    elif failure_type == "invalid_query":
        parts.append("The adapted UPPAAL queries are not valid for this XML model. Check syntax, types, and referenced identifiers.")
    elif failure_type == "timeout":
        parts.append("UPPAAL did not finish within the configured timeout. Simplify the model/query or increase the verifyta timeout.")
    elif failure_type == "semantic":
        parts.append("UPPAAL ran successfully, but at least one property/query did not match the expected behavior. Repair the LIRAs model behavior, not just query syntax.")
    else:
        parts.append("Check the diagnostics below and repair the LIRAs model or generated queries accordingly.")

    failed_queries = (internal_analysis.get("failed_queries") or []) + (adapted_analysis.get("failed_queries") or [])
    if failed_queries:
        parts.append("")
        parts.append("### FAILED QUERIES")
        for failed in failed_queries:
            parts.append(f"Query {failed.get('index')}")
            if failed.get("failure_kind"):
                parts.append(f"Failure kind: {failed['failure_kind']}")
            if failed.get("failure_reason"):
                parts.append(f"Reason: {failed['failure_reason']}")
            if failed.get("description"):
                parts.append(f"Description: {failed['description']}")
            if failed.get("source_formula"):
                parts.append(f"Original formula: {failed['source_formula']}")
            if failed.get("adapted_formula"):
                parts.append(f"Adapted formula: {failed['adapted_formula']}")
            if failed.get("expected_probability") is not None:
                parts.append(f"Expected probability: {failed['expected_probability']}")
            if failed.get("obtained_probability") is not None:
                parts.append(f"Obtained probability: {failed['obtained_probability']}")
            if failed.get("probability_delta") is not None:
                parts.append(f"Probability delta: {failed['probability_delta']}")
            if failed.get("probability_threshold") is not None:
                parts.append(f"Allowed probability delta threshold: {failed['probability_threshold']}")
            if failed.get("verifyta_output"):
                parts.append("UPPAAL output:")
                parts.append(str(failed["verifyta_output"]).strip())
            parts.append("")
    else:
        parts.append("")
        parts.append("No specific unsatisfied query was parsed. The failure likely happened before UPPAAL could evaluate the formulas.")

    stderr_candidates = [
        run_dir / "uppaal" / "verifyta_internal.stderr.txt",
        run_dir / "uppaal" / "verifyta_adapted.stderr.txt",
    ]
    diagnostic_text = "\n".join(
        filtered
        for path in stderr_candidates
        for filtered in (_filter_verifyta_warning_lines(_strip_ansi(_read_text_if_exists(path))),)
        if filtered
    )
    if diagnostic_text:
        parts.append("### VERIFYTA DIAGNOSTICS")
        parts.append(diagnostic_text[-12000:])

    wrong_liras = _read_text_if_exists(selected_liras_path)
    parts.append("### INCORRECT LIRAS CODE")
    parts.append(wrong_liras[-24000:] if len(wrong_liras) > 24000 else wrong_liras)

    return "\n".join(parts).strip()


def _record_pipeline_error(
    *,
    metadata_path: Path,
    step: str,
    command: list[str],
    exit_code: int,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    extra: Optional[dict] = None,
) -> None:
    stage_name = {
        "dsl_generator": "dsl_generation",
        "lira_cli": "xml_export",
        "query_adapter": "query_adaptation",
    }.get(step, step)
    failure_type = "timeout" if exit_code == 124 else "execution"
    stderr_preview = _filter_verifyta_warning_lines(_strip_ansi(_read_text_if_exists(stderr_path)))[-4000:] if stderr_path else ""
    reason = f"{stage_name} failed with exit code {exit_code}."
    if stderr_preview:
        reason = f"{stage_name} failed: {_first_error_line(stderr_preview)}"
    if step == "lira_cli":
        reason = "LIRAs-to-XML export failed, so UPPAAL verification could not run."
    stage = _stage_record(
        stage_name,
        "failed",
        failure_type=failure_type,
        failure_reason=reason,
        failure_details=_compact_dict(
            {
                "step": step,
                "command": " ".join(command),
                "exit_code": int(exit_code),
                "stdout_path": str(stdout_path) if stdout_path else None,
                "stderr_path": str(stderr_path) if stderr_path else None,
                "stderr_preview": stderr_preview,
            }
        ),
    )
    payload = {
        "last_command": " ".join(command),
        "last_exit_code": int(exit_code),
        "last_stdout_path": str(stdout_path) if stdout_path else None,
        "last_stderr_path": str(stderr_path) if stderr_path else None,
    }
    if step != "verifyta":
        _update_stage_metadata(metadata_path, stage)
        payload.update({"cycle_result": "failed", **_failure_payload(stage)})
    if extra:
        payload.update(extra)
    _update_pipeline_metadata(metadata_path, payload)
    print(f"[PIPELINE_ERROR] step={step} exit_code={exit_code}")


def _run_lira_cli_to_xml(
    *,
    config: dict,
    run_dir: Path,
    metadata_path: Path,
    liras_path: Path,
) -> Optional[Path]:
    if not bool(config.get("enable_xml_export", True)):
        _update_stage_metadata(
            metadata_path,
            _stage_record(
                "xml_export",
                "skipped",
                details={"reason": "XML export disabled by configuration."},
                artifacts=_stage_artifacts(selected_liras_path=liras_path),
            ),
        )
        return None

    lira_cli_jar = _resolve_path(str(config["lira_cli_jar"]))
    xml_dir = run_dir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    xml_path = xml_dir / f"{liras_path.stem}.xml"

    command = ["java", "-jar", str(lira_cli_jar), str(liras_path), str(xml_path)]
    timeout_sec = int(config.get("lira_cli_timeout", 120))
    stdout_path = xml_dir / "lira_cli.stdout.txt"
    stderr_path = xml_dir / "lira_cli.stderr.txt"

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("java not found on PATH\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="lira_cli",
            command=command,
            exit_code=127,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        return None
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text((exc.stdout or ""), encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") + f"\nTimed out after {timeout_sec}s\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="lira_cli",
            command=command,
            exit_code=124,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        return None

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    if completed.returncode != 0:
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="lira_cli",
            command=command,
            exit_code=completed.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        return None

    if not xml_path.exists():
        stderr_text = stderr_path.read_text(encoding="utf-8")
        stderr_path.write_text(stderr_text + "\nXML output file was not generated\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="lira_cli",
            command=command,
            exit_code=2,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        return None

    # Some liras-cli builds create a directory named "*.xml" and place the
    # actual model file inside it. Normalize this into a single XML file path
    # so downstream tooling (verifyta) always receives a file.
    if xml_path.is_dir():
        nested_xml_files = sorted(xml_path.rglob("*.xml"))
        if not nested_xml_files:
            stderr_text = stderr_path.read_text(encoding="utf-8")
            stderr_path.write_text(
                stderr_text + f"\nXML output path is a directory with no .xml files: {xml_path}\n",
                encoding="utf-8",
            )
            _record_pipeline_error(
                metadata_path=metadata_path,
                step="lira_cli",
                command=command,
                exit_code=2,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            return None

        # Keep behavior deterministic when multiple XML files are emitted.
        # Always preserve the originally requested output name (xml_path).
        selected_xml = nested_xml_files[0]
        temp_flat_xml = xml_dir / f"{xml_path.name}.flat.tmp"
        shutil.copyfile(selected_xml, temp_flat_xml)
        shutil.rmtree(xml_path)
        temp_flat_xml.rename(xml_path)

    _update_stage_metadata(
        metadata_path,
        _stage_record(
            "xml_export",
            "ok",
            details={"exit_code": int(completed.returncode)},
            artifacts=_stage_artifacts(
                selected_liras_path=liras_path,
                compiled_xml_path=xml_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ),
        ),
    )
    _update_pipeline_metadata(
        metadata_path,
        {
            "compiled_xml_path": str(xml_path),
            "last_command": " ".join(command),
            "last_exit_code": int(completed.returncode),
            "last_stdout_path": str(stdout_path),
            "last_stderr_path": str(stderr_path),
        },
    )
    return xml_path


def _run_verifyta(
    *,
    config: dict,
    run_dir: Path,
    metadata_path: Path,
    compiled_xml_path: Path,
    query_path: Optional[Path] = None,
    run_name: str = "default",
) -> dict:
    safe_run_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in run_name).strip("_")
    safe_run_name = safe_run_name or "default"
    metadata_prefix = f"verifyta_{safe_run_name}"

    if not bool(config.get("enable_uppaal", False)):
        stage = _stage_record(
            metadata_prefix,
            "skipped",
            execution_result="skipped",
            semantic_result="skipped",
            details={"reason": "UPPAAL verification disabled by configuration."},
            artifacts=_stage_artifacts(xml_path=compiled_xml_path, query_path=query_path),
        )
        _update_stage_metadata(metadata_path, stage)
        return stage

    uppaal_dir = run_dir / "uppaal"
    uppaal_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = uppaal_dir / f"verifyta_{safe_run_name}.stdout.txt"
    stderr_path = uppaal_dir / f"verifyta_{safe_run_name}.stderr.txt"
    timeout_sec = int(config.get("verifyta_timeout", 120))

    try:
        verifyta_bin = _resolve_verifyta_command(config)
    except FileNotFoundError as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc) + "\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="verifyta",
            command=["verifyta", str(compiled_xml_path)],
            exit_code=127,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        stderr = _filter_verifyta_warning_lines(_strip_ansi(_read_text_if_exists(stderr_path)))
        failure_type = _classify_verifyta_error(stderr, 127)
        return _stage_record(
            metadata_prefix,
            "failed",
            execution_result="failed",
            semantic_result="not_evaluated",
            failure_type=failure_type,
            failure_reason=_verifyta_execution_reason(metadata_prefix, failure_type, stderr),
            failure_details={
                "exit_code": 127,
                "error_message": _first_error_line(stderr),
                "stderr_path": str(stderr_path),
                "stdout_path": str(stdout_path),
            },
            artifacts=_stage_artifacts(xml_path=compiled_xml_path, query_path=query_path, stdout_path=stdout_path, stderr_path=stderr_path),
        )

    command = [verifyta_bin, str(compiled_xml_path)]
    if query_path is not None and query_path.exists():
        command.append(str(query_path))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("verifyta binary not found\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="verifyta",
            command=command,
            exit_code=127,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        stderr = _filter_verifyta_warning_lines(_strip_ansi(_read_text_if_exists(stderr_path)))
        failure_type = _classify_verifyta_error(stderr, 127)
        return _stage_record(
            metadata_prefix,
            "failed",
            execution_result="failed",
            semantic_result="not_evaluated",
            failure_type=failure_type,
            failure_reason=_verifyta_execution_reason(metadata_prefix, failure_type, stderr),
            failure_details={
                "exit_code": 127,
                "error_message": _first_error_line(stderr),
                "stderr_path": str(stderr_path),
                "stdout_path": str(stdout_path),
            },
            artifacts=_stage_artifacts(xml_path=compiled_xml_path, query_path=query_path, stdout_path=stdout_path, stderr_path=stderr_path),
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text((exc.stdout or ""), encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") + f"\nTimed out after {timeout_sec}s\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="verifyta",
            command=command,
            exit_code=124,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        stderr = _filter_verifyta_warning_lines(_strip_ansi(_read_text_if_exists(stderr_path)))
        failure_type = _classify_verifyta_error(stderr, 124)
        return _stage_record(
            metadata_prefix,
            "failed",
            execution_result="failed",
            semantic_result="not_evaluated",
            failure_type=failure_type,
            failure_reason=_verifyta_execution_reason(metadata_prefix, failure_type, stderr),
            failure_details={
                "exit_code": 124,
                "timeout_seconds": timeout_sec,
                "error_message": _first_error_line(stderr),
                "stderr_path": str(stderr_path),
                "stdout_path": str(stdout_path),
            },
            artifacts=_stage_artifacts(xml_path=compiled_xml_path, query_path=query_path, stdout_path=stdout_path, stderr_path=stderr_path),
        )

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    if completed.returncode != 0:
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="verifyta",
            command=command,
            exit_code=completed.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        stderr = _filter_verifyta_warning_lines(_strip_ansi(completed.stderr or ""))
        failure_type = _classify_verifyta_error(stderr, completed.returncode)
        return _stage_record(
            metadata_prefix,
            "failed",
            execution_result="failed",
            semantic_result="not_evaluated",
            failure_type=failure_type,
            failure_reason=_verifyta_execution_reason(metadata_prefix, failure_type, stderr),
            failure_details={
                "exit_code": int(completed.returncode),
                "error_message": _first_error_line(stderr),
                "stderr_preview": stderr[-4000:],
                "stderr_path": str(stderr_path),
                "stdout_path": str(stdout_path),
            },
            artifacts=_stage_artifacts(xml_path=compiled_xml_path, query_path=query_path, stdout_path=stdout_path, stderr_path=stderr_path),
        )

    stage = _stage_record(
        metadata_prefix,
        "ok",
        execution_result="ok",
        semantic_result="not_evaluated",
        details={
            "command": " ".join(command),
            "exit_code": int(completed.returncode),
        },
        artifacts=_stage_artifacts(
            xml_path=compiled_xml_path,
            query_path=query_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        ),
    )
    _update_stage_metadata(metadata_path, stage)
    _update_pipeline_metadata(
        metadata_path,
        {
            "last_command": " ".join(command),
            "last_exit_code": int(completed.returncode),
            "last_stdout_path": str(stdout_path),
            "last_stderr_path": str(stderr_path),
        },
    )
    return stage


def _finalize_verifyta_stage(
    execution_stage: dict,
    analysis: dict,
    *,
    command_was_required: bool,
) -> dict:
    if execution_stage.get("result") == "skipped":
        return execution_stage

    artifacts = execution_stage.get("artifacts") if isinstance(execution_stage.get("artifacts"), dict) else {}
    details = dict(execution_stage.get("details") or {})
    details.update(
        {
            "verified_formula_count": analysis.get("verified_formula_count", 0),
            "failed_query_count": analysis.get("failed_query_count", 0),
            "probability_threshold": analysis.get("probability_threshold"),
            "probability_checked_query_count": analysis.get("probability_checked_query_count", 0),
        }
    )

    if execution_stage.get("execution_result") == "failed":
        return _stage_record(
            str(execution_stage.get("stage")),
            "failed",
            execution_result="failed",
            semantic_result="not_evaluated",
            failure_type=execution_stage.get("failure_type"),
            failure_reason=execution_stage.get("failure_reason"),
            failure_details=execution_stage.get("failure_details"),
            details=details,
            artifacts=artifacts,
        )

    if not command_was_required:
        return _stage_record(
            str(execution_stage.get("stage")),
            "skipped",
            execution_result="skipped",
            semantic_result="skipped",
            details=details,
            artifacts=artifacts,
        )

    failed_queries = analysis.get("failed_queries") or []
    if failed_queries:
        return _stage_record(
            str(execution_stage.get("stage")),
            "failed",
            execution_result="ok",
            semantic_result="failed",
            failure_type="semantic",
            failure_reason=_semantic_failure_reason(str(execution_stage.get("stage")), analysis),
            failure_details={
                "failed_query_count": len(failed_queries),
                "failure_kinds": {
                    kind: sum(1 for item in failed_queries if item.get("failure_kind") == kind)
                    for kind in sorted({item.get("failure_kind") for item in failed_queries if isinstance(item, dict)})
                },
                "failed_queries": failed_queries,
            },
            details=details,
            artifacts=artifacts,
        )

    return _stage_record(
        str(execution_stage.get("stage")),
        "ok",
        execution_result="ok",
        semantic_result="ok",
        details=details,
        artifacts=artifacts,
    )


def _cycle_record_from_metadata(
    *,
    cycle_index: int,
    run_dir: Path,
    metadata_path: Path,
    base: dict,
    continue_feedback_loop: bool,
    exit_code: int,
    uppaal_feedback: Optional[str] = None,
) -> dict:
    metadata = _read_run_metadata(metadata_path)
    record = dict(base)
    record.update(
        {
            "cycle": cycle_index,
            "run_dir": str(run_dir),
            "metadata_path": str(metadata_path),
            "cycle_result": metadata.get("cycle_result", "failed" if exit_code else "ok"),
            "failed_stage": metadata.get("failed_stage"),
            "failure_type": metadata.get("failure_type"),
            "failure_reason": metadata.get("failure_reason"),
            "failure_details": metadata.get("failure_details"),
            "stages": metadata.get("stages") if isinstance(metadata.get("stages"), list) else [],
            "exit_code": exit_code,
            "continue_feedback_loop": continue_feedback_loop,
        }
    )
    for key in ("selected_liras_path", "compiled_xml_path", "adapted_query_path", "query_source_path"):
        if metadata.get(key):
            record[key] = metadata.get(key)
    if uppaal_feedback is not None:
        record["uppaal_feedback"] = uppaal_feedback
        record["uppaal_feedback_chars"] = len(uppaal_feedback)
    return record


def _pre_cycle_failure_record(cycle_index: int, outer_run_dir: Path, exc: Exception) -> dict:
    message = str(exc)
    failure_type = "configuration" if isinstance(exc, (FileNotFoundError, ValueError, RuntimeError)) else "execution"
    stage = _stage_record(
        "dsl_generation",
        "failed",
        failure_type=failure_type,
        failure_reason=f"DSL generation could not start: {message}",
        failure_details={
            "operation": "build_generator_from_config",
            "error_type": type(exc).__name__,
            "error_message": message,
            "traceback": traceback.format_exc()[-12000:],
        },
    )
    return {
        "cycle": cycle_index,
        "run_dir": str(outer_run_dir / f"ciclo{cycle_index}"),
        "metadata_path": None,
        "cycle_result": "failed",
        "failed_stage": "dsl_generation",
        "failure_type": failure_type,
        "failure_reason": stage["failure_reason"],
        "failure_details": stage["failure_details"],
        "stages": [stage],
        "exit_code": 1,
        "continue_feedback_loop": False,
    }


def _run_pipeline_cycle(
    *,
    config: dict,
    outer_run_dir: Path,
    cycle_index: int,
    uppaal_feedback: Optional[str],
) -> dict:
    from dsl_generator import build_generator_from_config

    cycle_dir = outer_run_dir / f"ciclo{cycle_index}"
    cycle_config = dict(config)
    cycle_config["pipeline_cycle"] = cycle_index
    cycle_config["run_dir_override"] = str(cycle_dir)
    if uppaal_feedback:
        cycle_config["uppaal_feedback"] = uppaal_feedback

    generator = build_generator_from_config(cycle_config)
    generator.run_automated_session(cycle_config)

    if not generator.run_dir or not generator.run_metadata_path:
        print("[PIPELINE_ERROR] step=dsl_generator exit_code=1")
        stage = _stage_record(
            "dsl_generation",
            "failed",
            failure_type="execution",
            failure_reason="DSL generation failed before creating run metadata.",
            failure_details={"operation": "dsl_generator.run_automated_session"},
        )
        return {
            "cycle": cycle_index,
            "run_dir": str(cycle_dir),
            "exit_code": 1,
            "cycle_result": "failed",
            "failed_stage": "dsl_generation",
            "failure_type": "execution",
            "failure_reason": stage["failure_reason"],
            "failure_details": stage["failure_details"],
            "stages": [stage],
            "continue_feedback_loop": False,
        }

    run_dir = Path(generator.run_dir)
    metadata_path = Path(generator.run_metadata_path)
    run_metadata = _read_run_metadata(metadata_path)
    probability_threshold = _probability_threshold(cycle_config)
    cycle_record = {
        "cycle": cycle_index,
        "run_dir": str(run_dir),
        "metadata_path": str(metadata_path),
        "uppaal_feedback_included": bool(uppaal_feedback),
        "uppaal_feedback_chars": len(uppaal_feedback or ""),
        "verifyta_probability_delta_threshold": probability_threshold,
    }

    success_liras_path = _find_success_liras_path(run_metadata, run_dir)
    dsl_stage = _dsl_generation_stage_from_metadata(run_metadata, success_liras_path)
    _update_stage_metadata(metadata_path, dsl_stage)
    if success_liras_path is None:
        return _cycle_record_from_metadata(
            cycle_index=cycle_index,
            run_dir=run_dir,
            metadata_path=metadata_path,
            base=cycle_record,
            continue_feedback_loop=False,
            exit_code=1,
        )

    _update_pipeline_metadata(
        metadata_path,
        {
            "pipeline_runner_version": "v3_structured_stage_metadata",
            "cycle_result": "running",
            "failed_stage": None,
            "failure_type": None,
            "failure_reason": None,
            "failure_details": None,
            "selected_liras_path": str(success_liras_path),
        },
    )
    cycle_record["selected_liras_path"] = str(success_liras_path)

    compiled_xml_path = _run_lira_cli_to_xml(
        config=cycle_config,
        run_dir=run_dir,
        metadata_path=metadata_path,
        liras_path=success_liras_path,
    )
    if bool(cycle_config.get("enable_xml_export", True)) and compiled_xml_path is None:
        return _cycle_record_from_metadata(
            cycle_index=cycle_index,
            run_dir=run_dir,
            metadata_path=metadata_path,
            base=cycle_record,
            continue_feedback_loop=False,
            exit_code=1,
        )

    if compiled_xml_path is None:
        _update_pipeline_metadata(
            metadata_path,
            {
                "cycle_result": "ok",
                "failed_stage": None,
                "failure_type": None,
                "failure_reason": None,
                "failure_details": None,
            },
        )
        return _cycle_record_from_metadata(
            cycle_index=cycle_index,
            run_dir=run_dir,
            metadata_path=metadata_path,
            base=cycle_record,
            continue_feedback_loop=False,
            exit_code=0,
        )
    cycle_record["compiled_xml_path"] = str(compiled_xml_path)
    xml_stage = _stage_record(
        "xml_export",
        "ok",
        details={"exit_code": 0},
        artifacts=_stage_artifacts(
            selected_liras_path=success_liras_path,
            compiled_xml_path=compiled_xml_path,
            stdout_path=run_dir / "xml" / "lira_cli.stdout.txt",
            stderr_path=run_dir / "xml" / "lira_cli.stderr.txt",
        ),
    )

    _update_pipeline_metadata(
        metadata_path,
        {
            "cycle_result": "running",
        },
    )

    internal_execution_stage = _run_verifyta(
        config=cycle_config,
        run_dir=run_dir,
        metadata_path=metadata_path,
        compiled_xml_path=compiled_xml_path,
        run_name="internal",
    )
    internal_analysis = _parse_verifyta_failures(
        stdout_path=run_dir / "uppaal" / "verifyta_internal.stdout.txt",
        stderr_path=run_dir / "uppaal" / "verifyta_internal.stderr.txt",
        adapted_query_path=None,
        source_query_path=None,
        probability_threshold=probability_threshold,
    )
    internal_stage = _finalize_verifyta_stage(
        internal_execution_stage,
        internal_analysis,
        command_was_required=bool(cycle_config.get("enable_uppaal", False)),
    )
    _update_stage_metadata(metadata_path, internal_stage)

    from query_adapter import generate_adapted_queries

    query_result = generate_adapted_queries(
        config=cycle_config,
        generator=generator,
        run_dir=run_dir,
        compiled_xml_path=compiled_xml_path,
        project_root=PROJECT_ROOT,
    )
    # Query adaptation uses the generator telemetry writer, which rewrites the
    # run metadata from its in-memory copy. Re-apply stages produced before the
    # query LLM call so the final ordered stage list stays complete.
    _update_stage_metadata(metadata_path, dsl_stage)
    _update_stage_metadata(metadata_path, xml_stage)
    _update_stage_metadata(metadata_path, internal_stage)
    query_stage = _stage_record(
        "query_adaptation",
        query_result.status,
        failure_type=query_result.failure_type,
        failure_reason=query_result.failure_reason,
        failure_details=query_result.failure_details,
        artifacts=_stage_artifacts(
            source_query_path=query_result.source_query_path,
            adapted_query_path=query_result.adapted_query_path,
            source_queries_copy_path=query_result.source_queries_copy_path,
            query_adapter_response_path=query_result.raw_response_path,
            query_xml_context_path=query_result.xml_context_path,
            queries_dir=getattr(generator, "run_queries_dir", None) or (run_dir / "queries"),
        ),
    )
    _update_stage_metadata(metadata_path, query_stage)
    _update_pipeline_metadata(
        metadata_path,
        {
            "query_source_path": str(query_result.source_query_path) if query_result.source_query_path else None,
            "adapted_query_path": str(query_result.adapted_query_path) if query_result.adapted_query_path else None,
        },
    )
    if query_result.status == "failed":
        return _cycle_record_from_metadata(
            cycle_index=cycle_index,
            run_dir=run_dir,
            metadata_path=metadata_path,
            base=cycle_record,
            continue_feedback_loop=False,
            exit_code=1,
        )

    adapted_query_path = query_result.adapted_query_path

    adapted_execution_stage = _stage_record(
        "verifyta_adapted",
        "skipped",
        execution_result="skipped",
        semantic_result="skipped",
        details={"reason": "Query adaptation disabled or no adapted query file produced."},
    )
    if adapted_query_path is not None:
        adapted_execution_stage = _run_verifyta(
            config=cycle_config,
            run_dir=run_dir,
            metadata_path=metadata_path,
            compiled_xml_path=compiled_xml_path,
            query_path=adapted_query_path,
            run_name="adapted",
        )
    adapted_analysis = _parse_verifyta_failures(
        stdout_path=run_dir / "uppaal" / "verifyta_adapted.stdout.txt",
        stderr_path=run_dir / "uppaal" / "verifyta_adapted.stderr.txt",
        adapted_query_path=adapted_query_path,
        source_query_path=query_result.source_query_path,
        probability_threshold=probability_threshold,
    )
    adapted_stage = _finalize_verifyta_stage(
        adapted_execution_stage,
        adapted_analysis,
        command_was_required=bool(cycle_config.get("enable_uppaal", False)) and adapted_query_path is not None,
    )
    _update_stage_metadata(metadata_path, adapted_stage)
    cycle_record.update({"query_source_path": str(query_result.source_query_path) if query_result.source_query_path else None})
    _update_pipeline_metadata(
        metadata_path,
        {
            "verifyta_probability_delta_threshold": probability_threshold,
            "verifyta_internal_analysis": internal_analysis,
            "verifyta_adapted_analysis": adapted_analysis,
        },
    )

    if not bool(cycle_config.get("enable_uppaal", False)):
        _update_pipeline_metadata(
            metadata_path,
            {
                "cycle_result": "ok",
                "failed_stage": None,
                "failure_type": None,
                "failure_reason": None,
                "failure_details": None,
            },
        )
        return _cycle_record_from_metadata(
            cycle_index=cycle_index,
            run_dir=run_dir,
            metadata_path=metadata_path,
            base=cycle_record,
            continue_feedback_loop=False,
            exit_code=0,
        )

    if internal_stage["result"] == "failed" or adapted_stage["result"] == "failed":
        failed_stage = internal_stage if internal_stage["result"] == "failed" else adapted_stage
        next_feedback = _build_uppaal_feedback_text(
            selected_liras_path=success_liras_path,
            internal_stage=internal_stage,
            adapted_stage=adapted_stage,
            internal_analysis=internal_analysis,
            adapted_analysis=adapted_analysis,
            run_dir=run_dir,
        )
        feedback_path = run_dir / "uppaal" / "uppaal_feedback_for_next_cycle.txt"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(next_feedback + "\n", encoding="utf-8")
        _update_pipeline_metadata(
            metadata_path,
            {
                "cycle_result": "failed",
                **_failure_payload(failed_stage),
                "uppaal_feedback_path": str(feedback_path),
            },
        )
        return _cycle_record_from_metadata(
            cycle_index=cycle_index,
            run_dir=run_dir,
            metadata_path=metadata_path,
            base={**cycle_record, "uppaal_feedback_path": str(feedback_path)},
            continue_feedback_loop=True,
            exit_code=1,
            uppaal_feedback=next_feedback,
        )

    _update_pipeline_metadata(
        metadata_path,
        {
            "cycle_result": "ok",
            "failed_stage": None,
            "failure_type": None,
            "failure_reason": None,
            "failure_details": None,
        },
    )
    return _cycle_record_from_metadata(
        cycle_index=cycle_index,
        run_dir=run_dir,
        metadata_path=metadata_path,
        base=cycle_record,
        continue_feedback_loop=False,
        exit_code=0,
    )


def _run_pipeline(config: dict) -> int:
    _validate_pipeline_config(config)
    max_cycles = _coerce_positive_int(config.get("max_uppaal_feedback_cycles", 1), default=1)
    outer_run_dir = _create_outer_run_dir(config)
    global_metadata_path = _init_global_pipeline_metadata(config, outer_run_dir, max_cycles)

    uppaal_feedback = None
    last_cycle_record: Optional[dict] = None

    for cycle_index in range(1, max_cycles + 1):
        print(f"\n=== UPPAAL FEEDBACK CYCLE {cycle_index}/{max_cycles} ===")
        try:
            cycle_record = _run_pipeline_cycle(
                config=config,
                outer_run_dir=outer_run_dir,
                cycle_index=cycle_index,
                uppaal_feedback=uppaal_feedback,
            )
        except Exception as exc:
            cycle_record = _pre_cycle_failure_record(cycle_index, outer_run_dir, exc)
            print(f"[PIPELINE_ERROR] stage=dsl_generation error={exc}")
        last_cycle_record = cycle_record

        cycle_record_for_metadata = dict(cycle_record)
        cycle_record_for_metadata.pop("uppaal_feedback", None)
        _append_global_cycle_metadata(global_metadata_path, cycle_record_for_metadata)

        if cycle_record.get("exit_code") == 0:
            _finalize_global_pipeline_metadata(
                global_metadata_path,
                {
                    "overall_result": "ok",
                    "failed_stage": None,
                    "failure_type": None,
                    "failure_reason": None,
                    "failure_details": None,
                    "successful_cycle": cycle_index,
                    "last_cycle_run_dir": cycle_record.get("run_dir"),
                    "last_cycle_metadata_path": cycle_record.get("metadata_path"),
                },
            )
            return 0

        if not cycle_record.get("continue_feedback_loop") or cycle_index >= max_cycles:
            break

        uppaal_feedback = str(cycle_record.get("uppaal_feedback") or "").strip() or None
        print(f"[UPPAAL_FEEDBACK] Retrying generation with feedback from cycle {cycle_index}")

    _finalize_global_pipeline_metadata(
        global_metadata_path,
        {
            "overall_result": "failed",
            "failed_stage": (last_cycle_record or {}).get("failed_stage"),
            "failure_type": (last_cycle_record or {}).get("failure_type"),
            "failure_reason": (last_cycle_record or {}).get("failure_reason"),
            "failure_details": (last_cycle_record or {}).get("failure_details"),
            "last_cycle_run_dir": (last_cycle_record or {}).get("run_dir"),
            "last_cycle_metadata_path": (last_cycle_record or {}).get("metadata_path"),
        },
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: LIRAS generation -> XML compilation -> query adaptation -> verifyta verification."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to pipeline config.json (default: config.json)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    config = _load_json(cfg_path)
    return _run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
