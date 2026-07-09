#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _HAS_MATPLOTLIB = True
except Exception:
    plt = None
    Line2D = None
    Patch = None
    _HAS_MATPLOTLIB = False


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "Runs"
DEFAULT_OUTPUT = ROOT / "Report" / "model_runs_analysis.html"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except Exception:
        return default
    return text if text else default


def _safe_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _duration_seconds(start: Any, end: Any) -> float | None:
    started = _parse_dt(start)
    finished = _parse_dt(end)
    if not started or not finished:
        return None
    return max((finished - started).total_seconds(), 0.0)


def _is_cycle_metadata(path: Path, metadata: dict[str, Any]) -> bool:
    if any(part.startswith("ciclo") for part in path.parts):
        return True
    return metadata.get("pipeline_cycle") is not None


def _model_label(metadata: dict[str, Any], meta_path: Path, runs_dir: Path) -> str:
    model = _safe_str(metadata.get("generation_model"))
    if model and model != "unknown":
        return model
    model = _safe_str(metadata.get("repair_model"))
    if model and model != "unknown":
        return model
    try:
        return meta_path.relative_to(runs_dir).parts[0]
    except Exception:
        return "unknown"


def _scenario_label(metadata: dict[str, Any], meta_path: Path, runs_dir: Path) -> str:
    scenario = _safe_str(metadata.get("scenario"))
    if scenario:
        return scenario
    try:
        return meta_path.relative_to(runs_dir).parts[1]
    except Exception:
        return "unknown"


def _state_label(metadata: dict[str, Any]) -> str:
    return _safe_str(metadata.get("overall_result") or metadata.get("status"), "unknown")


def _is_success(metadata: dict[str, Any]) -> bool:
    state = _state_label(metadata).lower()
    return state in {"ok", "success", "success_no_output"}


def _outcome_label(metadata: dict[str, Any]) -> str:
    state = _state_label(metadata).lower()
    if state in {"ok", "success", "success_no_output"}:
        return "success"
    if state in {"failed", "crashed", "setup_error", "max_iterations_reached", "error"}:
        return "failed"
    if state in {"running", "started"}:
        return "running"
    return "unknown"


def _cycle_count(metadata: dict[str, Any]) -> int:
    cycles = metadata.get("cycles")
    if isinstance(cycles, list):
        return len(cycles)

    iterations = metadata.get("iterations")
    if isinstance(iterations, list):
        return len(iterations)

    summary = metadata.get("summary")
    if isinstance(summary, dict):
        try:
            return int(summary.get("iterations_recorded") or 0)
        except Exception:
            return 0
    return 0


def _dsl_generation_iteration_samples(metadata: dict[str, Any]) -> list[int]:
    def from_details(details: dict[str, Any]) -> int | None:
        first_success = details.get("first_success_iteration")
        if first_success is not None:
            try:
                # Iteration ids are zero-based (ITER0, ITER1, ...), so +1 is the
                # number of attempts needed to produce the accepted DSL.
                return int(first_success) + 1
            except Exception:
                pass
        recorded = details.get("iterations_recorded")
        if recorded is not None:
            try:
                return int(recorded)
            except Exception:
                pass
        return None

    cycles = metadata.get("cycles")
    samples: list[int] = []
    if isinstance(cycles, list):
        for cycle in cycles:
            if not isinstance(cycle, dict):
                continue
            stages = cycle.get("stages") if isinstance(cycle.get("stages"), list) else []
            for stage in stages:
                if not isinstance(stage, dict) or stage.get("stage") != "dsl_generation":
                    continue
                details = stage.get("details") if isinstance(stage.get("details"), dict) else {}
                sample = from_details(details)
                if sample is not None:
                    samples.append(sample)
        if samples:
            return samples

    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    sample = from_details(summary)
    if sample is not None:
        return [sample]

    iterations = metadata.get("iterations")
    if isinstance(iterations, list):
        return [len(iterations)]

    return []


def _cycle_failure_summaries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    cycles = metadata.get("cycles")
    if not isinstance(cycles, list):
        return []

    rows: list[dict[str, Any]] = []
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue

        stages = cycle.get("stages") if isinstance(cycle.get("stages"), list) else []
        failed_stages: list[dict[str, str]] = []
        dsl_iterations: int | None = None

        for stage in stages:
            if not isinstance(stage, dict):
                continue
            if stage.get("stage") == "dsl_generation":
                details = stage.get("details") if isinstance(stage.get("details"), dict) else {}
                samples = _dsl_generation_iteration_samples({"summary": details})
                if samples:
                    dsl_iterations = samples[0]
            if _safe_str(stage.get("result")).lower() != "failed":
                continue
            failure_details = stage.get("failure_details") if isinstance(stage.get("failure_details"), dict) else {}
            failed_stages.append(
                {
                    "stage": _safe_str(stage.get("stage"), "unknown"),
                    "failure_type": _safe_str(stage.get("failure_type"), "unknown"),
                    "failure_reason": _safe_str(stage.get("failure_reason") or failure_details.get("error_message"), ""),
                }
            )

        top_details = cycle.get("failure_details") if isinstance(cycle.get("failure_details"), dict) else {}
        rows.append(
            {
                "cycle": cycle.get("cycle"),
                "result": _safe_str(cycle.get("cycle_result"), "unknown"),
                "failed_stage": _safe_str(cycle.get("failed_stage"), "none"),
                "failure_type": _safe_str(cycle.get("failure_type"), "none"),
                "failure_reason": _safe_str(
                    cycle.get("failure_reason") or top_details.get("error_message"),
                    "",
                ),
                "dsl_iterations": dsl_iterations,
                "failed_stages": failed_stages,
            }
        )

    return rows


def _dsl_token_totals(metadata: dict[str, Any]) -> dict[str, int | None]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "available_calls": 0,
    }
    found = False
    cycles = metadata.get("cycles")
    if isinstance(cycles, list):
        for cycle in cycles:
            if not isinstance(cycle, dict):
                continue
            stages = cycle.get("stages") if isinstance(cycle.get("stages"), list) else []
            for stage in stages:
                if not isinstance(stage, dict) or stage.get("stage") != "dsl_generation":
                    continue
                details = stage.get("details") if isinstance(stage.get("details"), dict) else {}
                found = True
                totals["prompt_tokens"] += int(details.get("prompt_tokens_total") or 0)
                totals["completion_tokens"] += int(details.get("completion_tokens_total") or 0)
                totals["total_tokens"] += int(details.get("total_tokens_total") or 0)
                totals["available_calls"] += int(details.get("token_usage_available_calls") or 0)

    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    if not found and summary:
        found = True
        totals["prompt_tokens"] = int(summary.get("prompt_tokens_total") or 0)
        totals["completion_tokens"] = int(summary.get("completion_tokens_total") or 0)
        totals["total_tokens"] = int(summary.get("total_tokens_total") or 0)
        totals["available_calls"] = int(summary.get("token_usage_available_calls") or 0)

    if not found or int(totals["available_calls"] or 0) <= 0:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "available_calls": 0,
        }
    return totals


def _dsl_generation_llm_durations_for_cycle(cycle_dir: Path) -> list[float]:
    durations: list[float] = []
    allowed_kinds = {"generate", "repair"}
    prompt_path = cycle_dir / "llm_prompts.jsonl"
    response_path = cycle_dir / "llm_responses.jsonl"
    prompts = [
        row for row in _read_jsonl(prompt_path)
        if _safe_str(row.get("kind")) in allowed_kinds
    ]
    responses = [
        row for row in _read_jsonl(response_path)
        if _safe_str(row.get("kind")) in allowed_kinds
    ]
    used_response_indexes: set[int] = set()
    for prompt in prompts:
        prompt_dt = _parse_dt(prompt.get("timestamp"))
        if not prompt_dt:
            continue
        prompt_kind = _safe_str(prompt.get("kind"))
        best_index: int | None = None
        best_dt: datetime | None = None
        for index, response in enumerate(responses):
            if index in used_response_indexes or _safe_str(response.get("kind")) != prompt_kind:
                continue
            response_dt = _parse_dt(response.get("timestamp"))
            if not response_dt or response_dt < prompt_dt:
                continue
            if best_dt is None or response_dt < best_dt:
                best_index = index
                best_dt = response_dt
        if best_index is None or best_dt is None:
            continue
        used_response_indexes.add(best_index)
        durations.append(max((best_dt - prompt_dt).total_seconds(), 0.0))
    return durations


def _dsl_generation_llm_durations(run_dir: Path) -> list[float]:
    durations: list[float] = []
    for cycle_dir in sorted(run_dir.glob("ciclo*")):
        if cycle_dir.is_dir():
            durations.extend(_dsl_generation_llm_durations_for_cycle(cycle_dir))
    return durations


def _dsl_generation_llm_duration_for_cycle(cycle_dir: Path) -> float | None:
    durations = _dsl_generation_llm_durations_for_cycle(cycle_dir)
    return round(sum(durations), 3) if durations else None


def _dsl_generation_llm_token_samples_for_cycle(cycle_dir: Path) -> dict[str, list[int]]:
    samples = {
        "completion_tokens": [],
        "total_tokens": [],
    }
    for row in _read_jsonl(cycle_dir / "hf_debug_responses.jsonl"):
        if _safe_str(row.get("kind")) not in {"generate", "repair"}:
            continue
        response_obj = row.get("response_obj") if isinstance(row.get("response_obj"), dict) else {}
        usage = response_obj.get("usage") if isinstance(response_obj.get("usage"), dict) else {}
        if usage.get("completion_tokens") is not None:
            samples["completion_tokens"].append(int(usage.get("completion_tokens") or 0))
        if usage.get("total_tokens") is not None:
            samples["total_tokens"].append(int(usage.get("total_tokens") or 0))
    return samples


def _dsl_generation_llm_token_samples(run_dir: Path) -> dict[str, list[int]]:
    samples = {
        "completion_tokens": [],
        "total_tokens": [],
    }
    for cycle_dir in sorted(run_dir.glob("ciclo*")):
        if not cycle_dir.is_dir():
            continue
        cycle_samples = _dsl_generation_llm_token_samples_for_cycle(cycle_dir)
        samples["completion_tokens"].extend(cycle_samples["completion_tokens"])
        samples["total_tokens"].extend(cycle_samples["total_tokens"])
    return samples


def _hf_debug_token_totals_for_dir(run_dir: Path) -> dict[str, int | None]:
    prompt_tokens = 0
    output_tokens = 0
    total_tokens = 0
    reasoning_tokens = 0
    found_output = False
    found_reasoning = False

    for debug_path in sorted(run_dir.glob("hf_debug_responses.jsonl")):
        for row in _read_jsonl(debug_path):
            response_obj = row.get("response_obj") if isinstance(row.get("response_obj"), dict) else {}
            usage = response_obj.get("usage") if isinstance(response_obj.get("usage"), dict) else {}
            if not usage:
                continue
            if usage.get("prompt_tokens") is not None:
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
            if usage.get("completion_tokens") is not None:
                output_tokens += int(usage.get("completion_tokens") or 0)
                found_output = True
            if usage.get("total_tokens") is not None:
                total_tokens += int(usage.get("total_tokens") or 0)
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
                reasoning_tokens += int(details.get("reasoning_tokens") or 0)
                found_reasoning = True

    return {
        "prompt_tokens": prompt_tokens if found_output else None,
        "output_tokens": output_tokens if found_output else None,
        "total_tokens": total_tokens if found_output else None,
        "reasoning_tokens": reasoning_tokens if found_reasoning else None,
    }


def _all_llm_token_totals(metadata: dict[str, Any], run_dir: Path) -> dict[str, int | None]:
    totals = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    found_output = False
    found_reasoning = False

    for cycle_dir in sorted(run_dir.glob("ciclo*")):
        if not cycle_dir.is_dir():
            continue
        debug_totals = _hf_debug_token_totals_for_dir(cycle_dir)
        if debug_totals["output_tokens"] is not None:
            found_output = True
            totals["prompt_tokens"] += int(debug_totals["prompt_tokens"] or 0)
            totals["output_tokens"] += int(debug_totals["output_tokens"] or 0)
            totals["total_tokens"] += int(debug_totals["total_tokens"] or 0)
        if debug_totals["reasoning_tokens"] is not None:
            found_reasoning = True
            totals["reasoning_tokens"] += int(debug_totals["reasoning_tokens"] or 0)

    if not found_output:
        cycles = metadata.get("cycles")
        if isinstance(cycles, list):
            for cycle in cycles:
                cycle_metadata = _read_json(Path(_safe_str(cycle.get("metadata_path")))) if isinstance(cycle, dict) else None
                telemetry = cycle_metadata.get("telemetry") if isinstance(cycle_metadata, dict) and isinstance(cycle_metadata.get("telemetry"), dict) else {}
                if telemetry.get("completion_tokens_total") is None:
                    continue
                found_output = True
                totals["prompt_tokens"] += int(telemetry.get("prompt_tokens_total") or 0)
                totals["output_tokens"] += int(telemetry.get("completion_tokens_total") or 0)
                totals["total_tokens"] += int(telemetry.get("total_tokens_total") or 0)

    return {
        "prompt_tokens": totals["prompt_tokens"] if found_output else None,
        "output_tokens": totals["output_tokens"] if found_output else None,
        "total_tokens": totals["total_tokens"] if found_output else None,
        "reasoning_tokens": totals["reasoning_tokens"] if found_reasoning else None,
    }


def _cycle_dir_from_metadata(cycle: dict[str, Any], fallback_run_dir: Path, cycle_index: int) -> Path:
    raw_dir = _safe_str(cycle.get("run_dir"))
    if raw_dir:
        return Path(raw_dir)
    raw_metadata = _safe_str(cycle.get("metadata_path"))
    if raw_metadata:
        return Path(raw_metadata).parent
    cycle_number = cycle.get("cycle")
    if cycle_number is not None:
        return fallback_run_dir / f"ciclo{cycle_number}"
    return fallback_run_dir / f"ciclo{cycle_index + 1}"


def _per_cycle_metrics(metadata: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    cycles = metadata.get("cycles")
    if not isinstance(cycles, list):
        return []

    rows: list[dict[str, Any]] = []
    for cycle_index, cycle in enumerate(cycles):
        if not isinstance(cycle, dict):
            continue
        dsl_iterations: int | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        token_usage_available_calls = 0

        stages = cycle.get("stages") if isinstance(cycle.get("stages"), list) else []
        for stage in stages:
            if not isinstance(stage, dict) or stage.get("stage") != "dsl_generation":
                continue
            details = stage.get("details") if isinstance(stage.get("details"), dict) else {}
            samples = _dsl_generation_iteration_samples({"summary": details})
            if samples:
                dsl_iterations = samples[0]
            token_usage_available_calls = int(details.get("token_usage_available_calls") or 0)
            if token_usage_available_calls > 0:
                prompt_tokens = int(details.get("prompt_tokens_total") or 0)
                completion_tokens = int(details.get("completion_tokens_total") or 0)
                total_tokens = int(details.get("total_tokens_total") or 0)

        cycle_dir = _cycle_dir_from_metadata(cycle, run_dir, cycle_index)
        debug_token_totals = _hf_debug_token_totals_for_dir(cycle_dir)
        dsl_token_samples = _dsl_generation_llm_token_samples_for_cycle(cycle_dir)
        dsl_generation_time_samples = _dsl_generation_llm_durations_for_cycle(cycle_dir)
        rows.append(
            {
                "cycle_index": cycle_index,
                "cycle": cycle.get("cycle", cycle_index + 1),
                "label": f"Ciclo {cycle_index}",
                "dsl_iterations": dsl_iterations,
                "dsl_generation_time_seconds": round(sum(dsl_generation_time_samples), 3) if dsl_generation_time_samples else None,
                "dsl_generation_time_samples": [round(value, 3) for value in dsl_generation_time_samples],
                "dsl_prompt_tokens": prompt_tokens,
                "dsl_completion_tokens": completion_tokens,
                "dsl_total_tokens": total_tokens,
                "dsl_token_usage_available_calls": token_usage_available_calls,
                "dsl_completion_token_samples": dsl_token_samples["completion_tokens"],
                "dsl_total_token_samples": dsl_token_samples["total_tokens"],
                "llm_output_tokens": debug_token_totals["output_tokens"],
                "llm_reasoning_tokens": debug_token_totals["reasoning_tokens"],
            }
        )
    return rows


def _last_iteration(metadata: dict[str, Any]) -> dict[str, Any]:
    iterations = metadata.get("iterations")
    if not isinstance(iterations, list):
        return {}
    for item in reversed(iterations):
        if isinstance(item, dict):
            return item
    return {}


def _load_cycle_metadata(cycle: dict[str, Any]) -> dict[str, Any] | None:
    raw_path = _safe_str(cycle.get("metadata_path"))
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.exists():
        return None
    return _read_json(path)


def _last_interesting_cycle(metadata: dict[str, Any]) -> dict[str, Any]:
    cycles = metadata.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return {}
    for cycle in reversed(cycles):
        if isinstance(cycle, dict) and _safe_str(cycle.get("cycle_result")).lower() == "failed":
            return cycle
    for cycle in reversed(cycles):
        if isinstance(cycle, dict):
            return cycle
    return {}


def _stage_failed_query_summary(stage: Any) -> tuple[int, list[str], Counter[str]]:
    if not isinstance(stage, dict):
        return 0, [], Counter()
    failure_details = stage.get("failure_details")
    if not isinstance(failure_details, dict):
        return 0, [], Counter()
    failed_queries = failure_details.get("failed_queries")
    if not isinstance(failed_queries, list):
        return 0, [], Counter()
    count = len(failed_queries)
    descriptions: list[str] = []
    kinds: Counter[str] = Counter()
    for item in failed_queries:
        if not isinstance(item, dict):
            continue
        kind = _safe_str(item.get("failure_kind"), "unknown")
        kinds[kind] += 1
        desc = _safe_str(item.get("description"))
        formula = _safe_str(item.get("adapted_formula") or item.get("source_formula"))
        if desc:
            descriptions.append(desc)
        elif formula:
            descriptions.append(formula)
    return count, descriptions[:5], kinds


def _failure_info(metadata: dict[str, Any]) -> dict[str, Any]:
    if _is_success(metadata):
        successful_cycle = metadata.get("successful_cycle")
        detail = "Pipeline completed"
        if successful_cycle is not None:
            detail = f"Successful cycle: {successful_cycle}"
        elif metadata.get("status"):
            detail = f"Final status: {_state_label(metadata)}"
        return {
            "outcome": "success",
            "failure_category": "none",
            "failure_detail": detail,
            "failed_queries": 0,
            "failed_query_examples": [],
            "failure_kinds": {},
        }

    if not isinstance(metadata.get("cycles"), list):
        breaking_error = metadata.get("breaking_error") if isinstance(metadata.get("breaking_error"), dict) else {}
        last_it = _last_iteration(metadata)
        score = last_it.get("compiler_error_score") if isinstance(last_it.get("compiler_error_score"), dict) else {}
        status = _state_label(metadata)
        category = _safe_str(
            breaking_error.get("type") or last_it.get("ended_because") or status,
            "unknown",
        )
        detail = _safe_str(breaking_error.get("message"), "")
        if not detail and score:
            detail = (
                f"Compiler: {score.get('error_lines', '?')} error(s), "
                f"{score.get('warning_lines', '?')} warning(s)"
            )
        if not detail:
            detail = f"Final status: {status}"
        return {
            "outcome": _outcome_label(metadata),
            "failure_category": category,
            "failure_detail": detail,
            "failed_queries": 0,
            "failed_query_examples": [],
            "failure_kinds": {},
        }

    cycle = _last_interesting_cycle(metadata)
    category = _safe_str(metadata.get("failure_type") or cycle.get("failure_type") or cycle.get("failed_stage"), "unknown")
    detail = _safe_str(metadata.get("failure_reason") or cycle.get("failure_reason"), "No failure reason recorded")
    failed_queries_total = 0
    query_examples: list[str] = []
    failure_kinds: Counter[str] = Counter()

    stages = cycle.get("stages") if isinstance(cycle.get("stages"), list) else []
    for stage in stages:
        count, examples, kinds = _stage_failed_query_summary(stage)
        failed_queries_total += count
        query_examples.extend(examples)
        failure_kinds.update(kinds)

    return {
        "outcome": "failed",
        "failure_category": category,
        "failure_detail": detail,
        "failed_queries": failed_queries_total,
        "failed_query_examples": query_examples[:5],
        "failure_kinds": dict(failure_kinds),
    }


def _build_record(metadata: dict[str, Any], meta_path: Path, runs_dir: Path) -> dict[str, Any]:
    cycles = metadata.get("cycles") if isinstance(metadata.get("cycles"), list) else []
    failure = _failure_info(metadata)
    duration = _duration_seconds(metadata.get("run_started_at"), metadata.get("run_finished_at"))
    model = _model_label(metadata, meta_path, runs_dir)
    cycle_count = _cycle_count(metadata)
    dsl_generation_iterations = _dsl_generation_iteration_samples(metadata)
    cycle_failures = _cycle_failure_summaries(metadata)
    dsl_tokens = _dsl_token_totals(metadata)
    dsl_generation_time_samples = _dsl_generation_llm_durations(meta_path.parent)
    dsl_generation_time = (
        round(sum(dsl_generation_time_samples), 3)
        if dsl_generation_time_samples
        else None
    )
    per_cycle_metrics = _per_cycle_metrics(metadata, meta_path.parent)
    llm_tokens = _all_llm_token_totals(metadata, meta_path.parent)
    dsl_token_samples = _dsl_generation_llm_token_samples(meta_path.parent)

    return {
        "run_id": _safe_str(metadata.get("run_id") or meta_path.parent.name),
        "model": model,
        "model_dir": _safe_rel(meta_path.parent.parents[3]) if len(meta_path.parent.parents) > 3 else "",
        "scenario": _scenario_label(metadata, meta_path, runs_dir),
        "system_prompt": _safe_str(metadata.get("system_prompt"), "unknown"),
        "repair_prompt": _safe_str(metadata.get("repair_prompt"), "unknown"),
        "shots": metadata.get("shots"),
        "repair_shots": metadata.get("repair_shots"),
        "llm_seed": metadata.get("llm_seed"),
        "started_at": metadata.get("run_started_at"),
        "finished_at": metadata.get("run_finished_at"),
        "duration_seconds": duration,
        "dsl_generation_iterations": dsl_generation_iterations,
        "dsl_generation_time_seconds": dsl_generation_time,
        "dsl_generation_time_samples": [round(value, 3) for value in dsl_generation_time_samples],
        "dsl_prompt_tokens": dsl_tokens["prompt_tokens"],
        "dsl_completion_tokens": dsl_tokens["completion_tokens"],
        "dsl_total_tokens": dsl_tokens["total_tokens"],
        "dsl_token_usage_available_calls": dsl_tokens["available_calls"],
        "dsl_completion_token_samples": dsl_token_samples["completion_tokens"],
        "dsl_total_token_samples": dsl_token_samples["total_tokens"],
        "llm_prompt_tokens": llm_tokens["prompt_tokens"],
        "llm_output_tokens": llm_tokens["output_tokens"],
        "llm_total_tokens": llm_tokens["total_tokens"],
        "llm_reasoning_tokens": llm_tokens["reasoning_tokens"],
        "per_cycle_metrics": per_cycle_metrics,
        "cycle_failures": cycle_failures,
        "pipeline_state": _state_label(metadata),
        "failed_stage": _safe_str(metadata.get("failed_stage"), "none"),
        "successful_cycle": metadata.get("successful_cycle"),
        "cycles": cycle_count,
        "outcome": failure["outcome"],
        "failure_category": failure["failure_category"],
        "failure_detail": failure["failure_detail"],
        "failed_queries": failure["failed_queries"],
        "failed_query_examples": failure["failed_query_examples"],
        "failure_kinds": failure["failure_kinds"],
        "metadata_path": _safe_rel(meta_path),
        "run_dir": _safe_rel(meta_path.parent),
    }


def _collect_records(runs_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for meta_path in sorted(runs_dir.rglob("run_metadata.json")):
        metadata = _read_json(meta_path)
        if not metadata or _is_cycle_metadata(meta_path, metadata):
            continue
        records.append(_build_record(metadata, meta_path, runs_dir))
    return records


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for model, model_records_iter in _group_by(records, "model").items():
        model_records = list(model_records_iter)
        failures = [r for r in model_records if r["outcome"] == "failed"]
        reasons = Counter(r["failure_category"] for r in failures)
        feedback_cycle_errors: Counter[str] = Counter()
        for record in model_records:
            cycle_failures = record.get("cycle_failures")
            if not isinstance(cycle_failures, list):
                continue
            for cycle in cycle_failures:
                if not isinstance(cycle, dict):
                    continue
                if _safe_str(cycle.get("result")).lower() != "failed":
                    continue
                error = _safe_str(cycle.get("failure_type"), "unknown")
                if error in {"", "none", "unknown"}:
                    error = _safe_str(cycle.get("failed_stage"), "unknown")
                feedback_cycle_errors[error] += 1
        outcomes = Counter(r["outcome"] for r in model_records)
        cycles = [int(r.get("cycles") or 0) for r in model_records]
        total_cycles = sum(cycles)
        durations = [
            float(r["duration_seconds"])
            for r in model_records
            if r.get("duration_seconds") is not None
        ]
        dsl_generation_times = [
            float(value)
            for r in model_records
            for value in (r.get("dsl_generation_time_samples") or [])
        ]
        dsl_total_tokens = [
            float(r["dsl_total_tokens"])
            for r in model_records
            if r.get("dsl_total_tokens") is not None
        ]
        dsl_completion_token_samples = [
            float(value)
            for r in model_records
            for value in (r.get("dsl_completion_token_samples") or [])
        ]
        llm_output_tokens = [
            float(r["llm_output_tokens"])
            for r in model_records
            if r.get("llm_output_tokens") is not None
        ]
        llm_reasoning_tokens = [
            float(r["llm_reasoning_tokens"])
            for r in model_records
            if r.get("llm_reasoning_tokens") is not None
        ]
        dsl_generation_iterations = [
            float(v)
            for r in model_records
            for v in (r.get("dsl_generation_iterations") or [])
        ]
        success_counts_by_scenario = [
            sum(1 for r in scenario_records if r.get("outcome") == "success")
            for scenario_records in _group_by(model_records, "scenario").values()
        ]
        success = outcomes.get("success", 0)
        by_model[model] = {
            "model": model,
            "total": len(model_records),
            "success": success,
            "failed": len(failures),
            "running": outcomes.get("running", 0),
            "unknown": outcomes.get("unknown", 0),
            "success_rate": (
                success / len(model_records)
                if model_records
                else 0
            ),
            "total_cycles": total_cycles,
            "avg_cycles": round(sum(cycles) / len(cycles), 2) if cycles else 0,
            "max_cycles": max(cycles) if cycles else 0,
            "avg_duration_seconds": round(sum(durations) / len(durations), 2) if durations else None,
            "avg_dsl_generation_time_seconds": (
                round(sum(dsl_generation_times) / len(dsl_generation_times), 2)
                if dsl_generation_times
                else None
            ),
            "avg_dsl_total_tokens": (
                round(sum(dsl_total_tokens) / len(dsl_total_tokens), 2)
                if dsl_total_tokens
                else None
            ),
            "avg_dsl_completion_tokens_per_generated_dsl": (
                round(sum(dsl_completion_token_samples) / len(dsl_completion_token_samples), 2)
                if dsl_completion_token_samples
                else None
            ),
            "avg_llm_output_tokens": (
                round(sum(llm_output_tokens) / len(llm_output_tokens), 2)
                if llm_output_tokens
                else None
            ),
            "avg_llm_reasoning_tokens": (
                round(sum(llm_reasoning_tokens) / len(llm_reasoning_tokens), 2)
                if llm_reasoning_tokens
                else None
            ),
            "cycle_box": _box_stats([float(v) for v in cycles]),
            "dsl_generation_iteration_box": _box_stats(dsl_generation_iterations),
            "dsl_generation_time_box": _box_stats(dsl_generation_times),
            "dsl_total_tokens_box": _box_stats(dsl_total_tokens),
            "success_count_box": _box_stats([float(v) for v in success_counts_by_scenario]),
            "outcomes": dict(outcomes),
            "reasons": dict(reasons),
            "feedback_cycle_errors": dict(feedback_cycle_errors),
        }

    return {
        "total_runs": len(records),
        "success": sum(1 for r in records if r["outcome"] == "success"),
        "failed": sum(1 for r in records if r["outcome"] == "failed"),
        "running": sum(1 for r in records if r["outcome"] == "running"),
        "unknown": sum(1 for r in records if r["outcome"] == "unknown"),
        "models": by_model,
        "failure_categories": dict(Counter(r["failure_category"] for r in records if r["outcome"] == "failed")),
        "scenarios": dict(Counter(r["scenario"] for r in records)),
    }


def _group_by(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_safe_str(record.get(key), "unknown")].append(record)
    return dict(grouped)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _box_stats(values: list[float]) -> dict[str, Any]:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return {
            "count": 0,
            "min": 0,
            "q1": 0,
            "median": 0,
            "q3": 0,
            "max": 0,
        }
    return {
        "count": len(clean),
        "min": round(clean[0], 3),
        "q1": round(_percentile(clean, 0.25), 3),
        "median": round(_percentile(clean, 0.5), 3),
        "q3": round(_percentile(clean, 0.75), 3),
        "max": round(clean[-1], 3),
    }


def _short_model_label(model: str) -> str:
    text = model.replace("openai/", "").replace("google/", "")
    text = text.replace("Qwen/", "").replace(":groq", "")
    return text


def _filter_key(
    model: str = "",
    scenario: str = "",
    outcome: str = "",
    reason: str = "",
    cycle_index: str = "",
) -> str:
    return json.dumps([model, scenario, outcome, reason, cycle_index], ensure_ascii=False, separators=(",", ":"))


def _filter_records(
    records: list[dict[str, Any]],
    *,
    model: str = "",
    scenario: str = "",
    outcome: str = "",
    reason: str = "",
    cycle_index: str = "",
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if model and record.get("model") != model:
            continue
        if scenario and record.get("scenario") != scenario:
            continue
        if outcome and record.get("outcome") != outcome:
            continue
        if reason and record.get("failure_category") != reason:
            continue
        if cycle_index:
            try:
                selected_cycle = int(cycle_index)
            except Exception:
                selected_cycle = -1
            per_cycle = record.get("per_cycle_metrics") if isinstance(record.get("per_cycle_metrics"), list) else []
            if not any(isinstance(cycle, dict) and cycle.get("cycle_index") == selected_cycle for cycle in per_cycle):
                continue
        rows.append(record)
    return rows


def _write_boxplot_figures(records: list[dict[str, Any]], output_html: Path) -> dict[str, Any]:
    if not _HAS_MATPLOTLIB or plt is None:
        return {}

    if not records:
        return {}

    asset_dir = output_html.with_suffix("")
    asset_dir = asset_dir.parent / f"{asset_dir.name}_assets" / "boxplots"
    asset_dir.mkdir(parents=True, exist_ok=True)

    def clean_values(values: list[Any]) -> list[float]:
        cleaned: list[float] = []
        for value in values:
            try:
                if value is None:
                    continue
                cleaned.append(float(value))
            except Exception:
                continue
        return cleaned

    def cycle_metric_values(model_records: list[dict[str, Any]], cycle_index: str, field: str) -> list[float]:
        if not cycle_index:
            return []
        try:
            selected_cycle = int(cycle_index)
        except Exception:
            return []
        return clean_values([
            cycle.get(field)
            for record in model_records
            for cycle in (record.get("per_cycle_metrics") or [])
            if isinstance(cycle, dict) and cycle.get("cycle_index") == selected_cycle
        ])

    def per_generated_dsl_token_values(model_records: list[dict[str, Any]], cycle_index: str = "") -> list[float]:
        if cycle_index:
            try:
                selected_cycle = int(cycle_index)
            except Exception:
                return []
            return clean_values([
                value
                for record in model_records
                for cycle in (record.get("per_cycle_metrics") or [])
                for value in (cycle.get("dsl_completion_token_samples") or [])
                if isinstance(cycle, dict) and cycle.get("cycle_index") == selected_cycle
            ])
        return clean_values([
            value
            for record in model_records
            for value in (record.get("dsl_completion_token_samples") or [])
        ])

    def metric_values(grouped: dict[str, list[dict[str, Any]]], model: str, metric: str, cycle_index: str = "") -> list[float]:
        model_records = grouped[model]
        if metric == "cycles":
            return clean_values([r.get("cycles") for r in model_records])
        if metric == "dsl_iterations":
            if cycle_index:
                return cycle_metric_values(model_records, cycle_index, "dsl_iterations")
            return clean_values([
                value
                for r in model_records
                for value in (r.get("dsl_generation_iterations") or [])
            ])
        if metric == "dsl_generation_time":
            if cycle_index:
                try:
                    selected_cycle = int(cycle_index)
                except Exception:
                    return []
                return clean_values([
                    value
                    for record in model_records
                    for cycle in (record.get("per_cycle_metrics") or [])
                    for value in (cycle.get("dsl_generation_time_samples") or [])
                    if isinstance(cycle, dict) and cycle.get("cycle_index") == selected_cycle
                ])
            return clean_values([
                value
                for r in model_records
                for value in (r.get("dsl_generation_time_samples") or [])
            ])
        if metric == "dsl_tokens_per_generated_dsl":
            return per_generated_dsl_token_values(model_records, cycle_index)
        return []

    metrics = [
        ("cycles", "Cicli pipeline per run", "Cicli nella run"),
        ("dsl_iterations", "DSL generati per ciclo", "Numero di DSL generati"),
        ("dsl_generation_time", "Tempo per singola chiamata DSL", "Secondi per generate/repair"),
        ("dsl_tokens_per_generated_dsl", "Output token per singolo DSL", "completion_tokens"),
    ]

    def draw_metric(ax: Any, rows: list[dict[str, Any]], metric: str, title: str, xlabel: str, cycle_index: str = "") -> None:
        grouped = _group_by(rows, "model")
        model_names = [
            model for model in sorted(grouped.keys(), key=lambda m: (-len(grouped[m]), m))
            if metric_values(grouped, model, metric, cycle_index)
        ]
        if not model_names:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{title}\nNessun dato disponibile", ha="center", va="center", fontsize=11)
            return
        labels = [_short_model_label(model) for model in model_names]
        values_by_model = [metric_values(grouped, model, metric, cycle_index) for model in model_names]
        ax.boxplot(
            values_by_model,
            vert=False,
            tick_labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            boxprops={"facecolor": "#bfdbfe", "edgecolor": "#1d4ed8", "linewidth": 1.2},
            medianprops={"color": "#172554", "linewidth": 1.8},
            whiskerprops={"color": "#475569", "linewidth": 1.1},
            capprops={"color": "#475569", "linewidth": 1.1},
            meanprops={"marker": "D", "markerfacecolor": "#f59e0b", "markeredgecolor": "#b45309", "markersize": 4},
            flierprops={"marker": "o", "markerfacecolor": "#fee2e2", "markeredgecolor": "#ef4444", "markersize": 3, "alpha": 0.8},
        )
        ax.set_title(title, pad=8, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Modello")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    def save_filtered_boxplots(rows: list[dict[str, Any]], key: str, title_suffix: str, cycle_index: str = "") -> str:
        grouped = _group_by(rows, "model")
        model_count = max(len(grouped), 1)
        panel_height = max(2.8, 0.34 * model_count + 1.35)
        fig_height = panel_height * len(metrics) + 1.0
        fig, axes = plt.subplots(len(metrics), 1, figsize=(11.6, fig_height))
        if len(metrics) == 1:
            axes = [axes]
        fig.suptitle(f"Boxplot per modello - {title_suffix}", fontsize=15, fontweight="bold", y=0.995)
        for ax, (metric, title, xlabel) in zip(axes, metrics):
            draw_metric(ax, rows, metric, title, xlabel, cycle_index)
        if Patch is not None and Line2D is not None:
            legend_handles = [
                Patch(facecolor="#bfdbfe", edgecolor="#1d4ed8", label="Box Q1-Q3"),
                Line2D([0], [0], color="#172554", linewidth=2, label="Mediana"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor="#f59e0b", markeredgecolor="#b45309", markersize=5, label="Media"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#fee2e2", markeredgecolor="#ef4444", markersize=4, label="Outlier"),
            ]
            fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=True, fontsize=9)
        fig.tight_layout(rect=(0, 0.025, 1, 0.985))
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        path = asset_dir / f"boxplots_{digest}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        try:
            return str(path.relative_to(output_html.parent))
        except Exception:
            return _safe_rel(path)

    values = {
        "model": [""] + sorted({_safe_str(r.get("model")) for r in records if _safe_str(r.get("model"))}),
        "scenario": [""] + sorted({_safe_str(r.get("scenario")) for r in records if _safe_str(r.get("scenario"))}),
        "outcome": [""] + sorted({_safe_str(r.get("outcome")) for r in records if _safe_str(r.get("outcome"))}),
        "reason": [""] + sorted({_safe_str(r.get("failure_category")) for r in records if _safe_str(r.get("failure_category"))}),
        "cycle_index": [""] + [
            str(value)
            for value in sorted({
                int(cycle.get("cycle_index"))
                for record in records
                for cycle in (record.get("per_cycle_metrics") or [])
                if isinstance(cycle, dict) and cycle.get("cycle_index") is not None
            })
        ],
    }
    filter_boxplots: dict[str, str] = {}
    for model, scenario, outcome, reason, cycle_index in itertools.product(
        values["model"],
        values["scenario"],
        values["outcome"],
        values["reason"],
        values["cycle_index"],
    ):
        if model:
            continue
        rows = _filter_records(records, model=model, scenario=scenario, outcome=outcome, reason=reason, cycle_index=cycle_index)
        if not rows:
            continue
        key = _filter_key(model, scenario, outcome, reason, cycle_index)
        suffix_parts = [
            f"modello={model or 'tutti'}",
            f"scenario={scenario or 'tutti'}",
            f"esito={outcome or 'tutti'}",
            f"motivo={reason or 'tutti'}",
            f"ciclo={cycle_index if cycle_index else 'tutti'}",
        ]
        filter_boxplots[key] = save_filtered_boxplots(rows, key, ", ".join(suffix_parts), cycle_index)

    return {
        "filter_boxplots": filter_boxplots,
        "default_filter_key": _filter_key(),
        "filter_dimensions": values,
    }


def _build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False).replace("</script", "<\\/script")
    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analisi Run per Modello</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #64748b;
      --line: #dbe3ef;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --good: #15803d;
      --bad: #b91c1c;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 80% -10%, #dbeafe 0, #dbeafe 24%, transparent 44%),
        radial-gradient(circle at -10% 110%, #fef3c7 0, #fef3c7 22%, transparent 42%),
        var(--bg);
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 22px; display: grid; gap: 16px; }}
    .hero {{
      background: linear-gradient(100deg, #1d4ed8 0%, #172554 100%);
      color: white;
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.20);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 0; color: #dbeafe; }}
    .cards, .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .card, .panel, .model-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }}
    .card {{ padding: 14px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    .value {{ font-size: 28px; font-weight: 800; margin-top: 5px; }}
    .model-card {{ padding: 14px; display: grid; gap: 10px; }}
    .model-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }}
    .model-name {{ font-weight: 800; overflow-wrap: anywhere; }}
    .rate {{ color: var(--accent); font-weight: 800; }}
    .reason-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .chip {{ border: 1px solid var(--line); background: #f8fafc; border-radius: 999px; padding: 4px 8px; font-size: 12px; }}
    .charts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    @media (max-width: 1050px) {{ .charts {{ grid-template-columns: 1fr; }} }}
    .chart {{ padding: 14px; display: grid; gap: 12px; }}
    .chart h2 {{ margin: 0; padding: 0; border: 0; background: transparent; font-size: 16px; }}
    .chart-note {{ color: var(--muted); font-size: 12px; }}
    .figure-card {{ grid-column: span 2; }}
    @media (max-width: 1050px) {{ .figure-card {{ grid-column: span 1; }} }}
    .figure-card img {{
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
    }}
    .model-summary {{ grid-column: 1 / -1; }}
    .summary-table {{ overflow: auto; }}
    .summary-table table {{ min-width: 820px; }}
    .summary-table td:not(:first-child),
    .summary-table th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .summary-table .model-cell {{ max-width: 360px; overflow-wrap: anywhere; font-weight: 700; }}
    .boxplot-row {{ display: grid; grid-template-columns: minmax(120px, 1.1fr) minmax(220px, 2fr) 120px; gap: 10px; align-items: center; }}
    .boxplot-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }}
    .boxplot-axis {{
      position: relative;
      height: 34px;
      border-radius: 8px;
      background: #eef2f7;
      overflow: hidden;
    }}
    .boxplot-x-axis {{
      display: grid;
      grid-template-columns: minmax(120px, 1.1fr) minmax(220px, 2fr) 120px;
      gap: 10px;
      align-items: start;
      margin-top: -4px;
    }}
    .boxplot-scale {{
      position: relative;
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 8px;
      color: var(--muted);
      font-size: 11px;
      font-variant-numeric: tabular-nums;
      padding-top: 10px;
      border-top: 1px solid #94a3b8;
    }}
    .boxplot-scale span {{
      position: relative;
    }}
    .boxplot-scale span::before {{
      content: "";
      position: absolute;
      top: -10px;
      left: 0;
      width: 1px;
      height: 6px;
      background: #94a3b8;
    }}
    .boxplot-scale span:nth-child(2),
    .boxplot-scale span:nth-child(3),
    .boxplot-scale span:nth-child(4) {{ text-align: center; }}
    .boxplot-scale span:nth-child(2)::before,
    .boxplot-scale span:nth-child(3)::before,
    .boxplot-scale span:nth-child(4)::before {{ left: 50%; }}
    .boxplot-scale span:last-child {{ text-align: right; }}
    .boxplot-scale span:last-child::before {{ left: auto; right: 0; }}
    .boxplot-whisker {{
      position: absolute;
      top: 16px;
      height: 2px;
      background: #475569;
    }}
    .boxplot-box {{
      position: absolute;
      top: 9px;
      height: 16px;
      border: 1px solid #1d4ed8;
      background: #bfdbfe;
      border-radius: 4px;
    }}
    .boxplot-median {{
      position: absolute;
      top: 6px;
      width: 2px;
      height: 22px;
      background: #172554;
    }}
    .boxplot-value {{ text-align: right; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted); }}
    .controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; padding: 12px; }}
    .control label {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; margin-bottom: 4px; }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 9px 10px;
      background: white;
      color: var(--ink);
      font-size: 14px;
    }}
    .main-grid {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(340px, .8fr); gap: 14px; align-items: start; }}
    @media (max-width: 1050px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}
    .panel h2 {{ margin: 0; padding: 13px 14px; border-bottom: 1px solid var(--line); font-size: 16px; background: #f8fafc; }}
    .panel.chart h2 {{ padding: 0; border-bottom: 0; background: transparent; }}
    .table-wrap {{ max-height: 62vh; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; text-align: left; padding: 9px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    td {{ padding: 8px 9px; border-bottom: 1px solid #edf2f7; vertical-align: top; }}
    tr {{ cursor: pointer; }}
    tbody tr:hover, tbody tr.active {{ background: var(--accent-soft); }}
    .status {{ display: inline-block; border-radius: 999px; padding: 3px 8px; font-weight: 700; font-size: 12px; }}
    .status.success {{ color: var(--good); background: #dcfce7; }}
    .status.failed {{ color: var(--bad); background: #fee2e2; }}
    .status.running {{ color: var(--warn); background: #fef3c7; }}
    .status.unknown {{ color: var(--muted); background: #e2e8f0; }}
    .mono {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; }}
    .muted {{ color: var(--muted); }}
    .clamp {{ max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .detail {{ padding: 14px; display: grid; gap: 12px; }}
    .detail-grid {{ display: grid; gap: 8px; }}
    .kv {{ display: grid; grid-template-columns: 130px 1fr; gap: 10px; }}
    .kv b {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .kv span {{ overflow-wrap: anywhere; }}
    .cycle-detail {{ overflow: auto; }}
    .cycle-detail table {{ min-width: 760px; }}
    .cycle-detail td {{ vertical-align: top; }}
    .stage-list {{ display: grid; gap: 4px; }}
    .stage-item {{ border: 1px solid var(--line); border-radius: 8px; padding: 5px 7px; background: #f8fafc; }}
    .stage-item b {{ color: var(--bad); }}
    pre {{
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
      max-height: 260px;
      white-space: pre-wrap;
      font-size: 12px;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Analisi run per modello</h1>
      <p>Riepilogo dinamico delle run in <span class="mono">Runs/</span>: successi, fallimenti e motivi principali.</p>
    </section>

    <section class="cards" id="cards"></section>
    <section class="charts" id="charts"></section>
    <section class="model-grid" id="modelCards"></section>

    <section class="panel controls">
      <div class="control"><label>Cerca</label><input id="search" placeholder="run, modello, scenario, motivo" /></div>
      <div class="control"><label>Modello</label><select id="model"></select></div>
      <div class="control"><label>Scenario</label><select id="scenario"></select></div>
      <div class="control"><label>Esito</label><select id="outcome"></select></div>
      <div class="control"><label>Motivo fallimento</label><select id="reason"></select></div>
      <div class="control"><label>Ciclo</label><select id="cycle"></select></div>
    </section>

    <section class="main-grid">
      <div class="panel">
        <h2>Run <span id="rowCount" class="muted"></span></h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Modello</th>
                <th>Scenario</th>
                <th>Esito</th>
                <th>Motivo</th>
                <th>Seed</th>
                <th>Cicli</th>
                <th>Durata</th>
                <th>Dettaglio</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <h2>Dettaglio run</h2>
        <div class="detail" id="detail">Seleziona una run.</div>
      </div>
    </section>
  </div>

  <script id="payload" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent || '{{}}');
    const records = Array.isArray(payload.records) ? payload.records : [];
    const summary = payload.summary || {{}};
    const figures = payload.figures || {{}};
    const el = {{
      cards: document.getElementById('cards'),
      charts: document.getElementById('charts'),
      modelCards: document.getElementById('modelCards'),
      search: document.getElementById('search'),
      model: document.getElementById('model'),
      scenario: document.getElementById('scenario'),
      outcome: document.getElementById('outcome'),
      reason: document.getElementById('reason'),
      cycle: document.getElementById('cycle'),
      rows: document.getElementById('rows'),
      rowCount: document.getElementById('rowCount'),
      detail: document.getElementById('detail'),
    }};
    let selected = null;

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}
    function fmtNum(value) {{ return Number(value || 0).toLocaleString(); }}
    function fmtPct(value) {{ return (Number(value || 0) * 100).toFixed(1) + '%'; }}
    function fmtDuration(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      const sec = Number(value);
      if (sec < 60) return sec.toFixed(1) + 's';
      const min = Math.floor(sec / 60);
      return min + 'm ' + Math.round(sec % 60) + 's';
    }}
    function uniq(values) {{ return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b))); }}
    function fillSelect(node, values, label) {{
      node.innerHTML = '<option value="">' + esc(label) + '</option>' + values.map(v => '<option value="' + esc(v) + '">' + esc(v) + '</option>').join('');
    }}
    function fillCycleSelect() {{
      const values = ((figures.filter_dimensions || {{}}).cycle_index || [])
        .filter(v => v !== '')
        .sort((a, b) => Number(a) - Number(b));
      el.cycle.innerHTML = '<option value="">Tutti</option>' +
        values.map(v => '<option value="' + esc(v) + '">Ciclo ' + esc(v) + '</option>').join('');
    }}
    function card(label, value) {{
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = '<div class="label">' + esc(label) + '</div><div class="value">' + esc(value) + '</div>';
      el.cards.appendChild(div);
    }}
    const modelDisplayOrder = new Map([
      ['Qwen/Qwen3.5-9B', 0],
      ['zai-org/GLM-5.2', 1],
      ['openai/gpt-oss-20b', 2],
      ['Qwen/Qwen3.6-35B-A3B', 3],
      ['google/gemma-4-31B-it', 4],
    ]);
    function compareModels(a, b) {{
      const ai = modelDisplayOrder.has(a.model) ? modelDisplayOrder.get(a.model) : 999;
      const bi = modelDisplayOrder.has(b.model) ? modelDisplayOrder.get(b.model) : 999;
      if (ai !== bi) return ai - bi;
      return String(a.model || '').localeCompare(String(b.model || ''));
    }}
    function summarizeRecords(rows) {{
      const byModel = new Map();
      const selectedCycle = el.cycle && el.cycle.value !== '' ? Number(el.cycle.value) : null;
      for (const r of rows) {{
        const key = r.model || 'unknown';
        if (!byModel.has(key)) {{
          byModel.set(key, {{model: key, total: 0, success: 0, failed: 0, running: 0, unknown: 0, cycles: [], dslTimes: [], completionTokenSamples: [], feedbackCycleErrors: {{}}}});
        }}
        const item = byModel.get(key);
        item.total += 1;
        if (r.outcome === 'success') item.success += 1;
        else if (r.outcome === 'failed') item.failed += 1;
        else if (r.outcome === 'running') item.running += 1;
        else item.unknown += 1;
        const cycles = Number(r.cycles);
        if (Number.isFinite(cycles)) item.cycles.push(cycles);
        const cycleFailures = Array.isArray(r.cycle_failures) ? r.cycle_failures : [];
        for (const cycle of cycleFailures) {{
          if (!cycle || String(cycle.result || '').toLowerCase() !== 'failed') continue;
          if (selectedCycle !== null && Number(cycle.cycle) - 1 !== selectedCycle) continue;
          let error = String(cycle.failure_type || 'unknown');
          if (!error || error === 'none' || error === 'unknown') error = String(cycle.failed_stage || 'unknown');
          item.feedbackCycleErrors[error] = (item.feedbackCycleErrors[error] || 0) + 1;
        }}
        if (selectedCycle !== null) {{
          const perCycle = Array.isArray(r.per_cycle_metrics) ? r.per_cycle_metrics : [];
          for (const cycle of perCycle) {{
            if (Number(cycle.cycle_index) !== selectedCycle) continue;
            const timeSamples = Array.isArray(cycle.dsl_generation_time_samples) ? cycle.dsl_generation_time_samples : [];
            for (const value of timeSamples) {{
              const sample = Number(value);
              if (Number.isFinite(sample)) item.dslTimes.push(sample);
            }}
            const completionSamples = Array.isArray(cycle.dsl_completion_token_samples) ? cycle.dsl_completion_token_samples : [];
            for (const value of completionSamples) {{
              const sample = Number(value);
              if (Number.isFinite(sample)) item.completionTokenSamples.push(sample);
            }}
          }}
        }} else {{
          const timeSamples = Array.isArray(r.dsl_generation_time_samples) ? r.dsl_generation_time_samples : [];
          for (const value of timeSamples) {{
            const sample = Number(value);
            if (Number.isFinite(sample)) item.dslTimes.push(sample);
          }}
          const completionSamples = Array.isArray(r.dsl_completion_token_samples) ? r.dsl_completion_token_samples : [];
          for (const value of completionSamples) {{
            const sample = Number(value);
            if (Number.isFinite(sample)) item.completionTokenSamples.push(sample);
          }}
        }}
      }}
      return [...byModel.values()].map(item => ({{
        model: item.model,
        total: item.total,
        success: item.success,
        failed: item.failed,
        running: item.running,
        unknown: item.unknown,
        success_rate: item.success / Math.max(item.total, 1),
        total_cycles: item.cycles.reduce((a, b) => a + b, 0),
        feedback_cycle_errors: item.feedbackCycleErrors,
        avg_cycles: item.cycles.length ? item.cycles.reduce((a, b) => a + b, 0) / item.cycles.length : 0,
        avg_dsl_generation_time_seconds: item.dslTimes.length ? item.dslTimes.reduce((a, b) => a + b, 0) / item.dslTimes.length : null,
        avg_dsl_completion_tokens_per_generated_dsl: item.completionTokenSamples.length ? item.completionTokenSamples.reduce((a, b) => a + b, 0) / item.completionTokenSamples.length : null,
      }})).sort(compareModels);
    }}
    function modelSummaryTable(rows) {{
      const body = rows.map(row => {{
        const total = Number(row.total || 0);
        const denom = Math.max(total, 1);
        return '<tr>' +
          '<td class="model-cell">' + esc(row.model) + '</td>' +
          '<td>' + fmtNum(row.success) + '/' + fmtNum(denom) + '</td>' +
          '<td>' + fmtNum(row.failed) + '/' + fmtNum(denom) + '</td>' +
          '<td>' + Number(row.avg_cycles || 0).toFixed(2) + '</td>' +
          '<td>' + fmtDuration(row.avg_dsl_generation_time_seconds) + '</td>' +
          '<td>' + (row.avg_dsl_completion_tokens_per_generated_dsl === null || row.avg_dsl_completion_tokens_per_generated_dsl === undefined ? '-' : fmtNum(Math.round(row.avg_dsl_completion_tokens_per_generated_dsl))) + '</td>' +
        '</tr>';
      }}).join('');
      return '<article class="panel chart model-summary"><h2>Riepilogo per modello</h2>' +
        '<div class="chart-note">Sintesi numerica delle run: completion_tokens medio calcolato sui singoli DSL generati.</div>' +
        '<div class="summary-table"><table><thead><tr>' +
        '<th>Modello</th><th>Successi</th><th>Falliti</th><th>Cicli medi/run</th><th>Tempo medio/chiamata DSL</th><th>Output token medi/DSL</th>' +
        '</tr></thead><tbody>' + body + '</tbody></table></div></article>';
    }}
    function figureChart(title, note, path) {{
      if (!path) return '';
      return '<article class="panel chart figure-card"><h2>' + esc(title) + '</h2>' +
        '<div class="chart-note">' + esc(note) + '</div>' +
        '<img src="' + esc(path) + '" alt="' + esc(title) + '" />' +
        '</article>';
    }}
    function filterFigureKey() {{
      return JSON.stringify([el.model.value || '', el.scenario.value || '', el.outcome.value || '', el.reason.value || '', el.cycle.value || '']);
    }}
    function boxplotFigureForCurrentFilters() {{
      const plots = figures.filter_boxplots || {{}};
      return plots[filterFigureKey()] || plots[figures.default_filter_key] || '';
    }}
    function renderCharts() {{
      const rows = filteredRows();
      const searchNote = el.search.value.trim()
        ? ' Il campo Cerca filtra tabella e riepilogo; i boxplot Matplotlib seguono i filtri a tendina pre-generati.'
        : '';
      const chartBlocks = [
        modelSummaryTable(summarizeRecords(rows)),
      ];
      if (!el.model.value) {{
        chartBlocks.push(
          figureChart(
            'Distribuzioni per modello',
            'Ogni boxplot usa i datapoint reali: cicli per run, DSL generati per ciclo, tempo di ogni chiamata generate/repair, e completion_tokens di ogni DSL generato. Query adaptation escluse.' + searchNote,
            boxplotFigureForCurrentFilters()
          ) || '<article class="panel chart figure-card"><h2>Distribuzioni per modello</h2><div class="muted">Nessun grafico disponibile per i filtri correnti.</div></article>'
        );
      }}
      el.charts.innerHTML = chartBlocks.join('');
    }}
    function renderTop() {{
      const rows = filteredRows();
      const models = summarizeRecords(rows);
      const counts = rows.reduce((acc, r) => {{
        if (r.outcome === 'success') acc.success += 1;
        else if (r.outcome === 'failed') acc.failed += 1;
        else if (r.outcome === 'running') acc.running += 1;
        else acc.unknown += 1;
        return acc;
      }}, {{success: 0, failed: 0, running: 0, unknown: 0}});
      el.cards.innerHTML = '';
      card('Run totali', fmtNum(rows.length));
      card('Riuscite', fmtNum(counts.success));
      card('Fallite', fmtNum(counts.failed));
      card('In corso', fmtNum(counts.running));
      card('Success rate', fmtPct(counts.success / Math.max(rows.length, 1)));
      el.modelCards.innerHTML = models.map(m => {{
        const cycleErrors = Object.entries(m.feedback_cycle_errors || {{}})
          .sort((a, b) => b[1] - a[1])
          .map(([name, count]) => '<span class="chip">' + esc(name) + ': ' + fmtNum(count) + '</span>')
          .join('');
        const failed = Number(m.failed || 0);
        const success = Number(m.success || 0);
        const total = Number(m.total || 0);
        const totalCycles = Number(m.total_cycles || 0);
        return '<article class="model-card">' +
          '<div class="model-head"><div class="model-name">' + esc(m.model) + '</div><div class="rate">' + fmtNum(success) + '/' + fmtNum(total) + '</div></div>' +
          '<div>Cicli totali: <b>' + fmtNum(totalCycles) + '</b></div>' +
          '<div class="label">Errori cicli feedback</div>' +
          '<div class="reason-list">' + (cycleErrors || '<span class="chip">nessun ciclo fallito</span>') + '</div>' +
          '</article>';
      }}).join('');
    }}
    function setupFilters() {{
      fillSelect(el.model, uniq(records.map(r => r.model)), 'Tutti');
      fillSelect(el.scenario, uniq(records.map(r => r.scenario)), 'Tutti');
      fillSelect(el.outcome, uniq(records.map(r => r.outcome)), 'Tutti');
      fillSelect(el.reason, uniq(records.map(r => r.failure_category)), 'Tutti');
      fillCycleSelect();
    }}
    function filteredRows() {{
      const q = el.search.value.trim().toLowerCase();
      return records.filter(r => {{
        if (el.model.value && r.model !== el.model.value) return false;
        if (el.scenario.value && r.scenario !== el.scenario.value) return false;
        if (el.outcome.value && r.outcome !== el.outcome.value) return false;
        if (el.reason.value && r.failure_category !== el.reason.value) return false;
        if (el.cycle.value) {{
          const selectedCycle = Number(el.cycle.value);
          const perCycle = Array.isArray(r.per_cycle_metrics) ? r.per_cycle_metrics : [];
          if (!perCycle.some(c => Number(c.cycle_index) === selectedCycle)) return false;
        }}
        if (!q) return true;
        return [r.run_id, r.model, r.scenario, r.outcome, r.failure_category, r.failure_detail, r.llm_seed, r.metadata_path]
          .join(' ').toLowerCase().includes(q);
      }}).sort((a, b) => String(b.started_at || '').localeCompare(String(a.started_at || '')));
    }}
    function renderRows() {{
      const rows = filteredRows();
      el.rowCount.textContent = '(' + rows.length + ' visibili)';
      el.rows.innerHTML = rows.map(r => {{
        const active = selected && selected.run_id === r.run_id ? ' class="active"' : '';
        return '<tr data-run="' + esc(r.run_id) + '"' + active + '>' +
          '<td class="mono">' + esc(r.run_id) + '</td>' +
          '<td class="clamp">' + esc(r.model) + '</td>' +
          '<td>' + esc(r.scenario) + '</td>' +
          '<td><span class="status ' + esc(r.outcome) + '">' + esc(r.outcome) + '</span></td>' +
          '<td>' + esc(r.failure_category) + '</td>' +
          '<td class="mono">' + esc(r.llm_seed ?? '-') + '</td>' +
          '<td>' + fmtNum(r.cycles) + '</td>' +
          '<td>' + fmtDuration(r.duration_seconds) + '</td>' +
          '<td class="clamp muted">' + esc(r.failure_detail || '-') + '</td>' +
          '</tr>';
      }}).join('') || '<tr><td colspan="9" class="muted">Nessuna run trovata.</td></tr>';
      for (const tr of el.rows.querySelectorAll('tr[data-run]')) {{
        tr.addEventListener('click', () => {{
          selected = records.find(r => r.run_id === tr.dataset.run);
          renderRows();
          renderDetail(selected);
        }});
      }}
      if (!selected && rows.length) {{
        selected = rows[0];
        renderDetail(selected);
        renderRows();
      }}
    }}
    function renderDetail(r) {{
      if (!r) {{
        el.detail.textContent = 'Nessuna run selezionata.';
        return;
      }}
      const examples = Array.isArray(r.failed_query_examples) && r.failed_query_examples.length
        ? '<pre>' + esc(r.failed_query_examples.join('\\n\\n')) + '</pre>'
        : '<span class="muted">Nessuna query fallita registrata.</span>';
      const cycleRows = Array.isArray(r.cycle_failures) && r.cycle_failures.length
        ? r.cycle_failures.map(c => {{
            const stages = Array.isArray(c.failed_stages) && c.failed_stages.length
              ? '<div class="stage-list">' + c.failed_stages.map(s =>
                  '<div class="stage-item"><b>' + esc(s.stage) + '</b> · ' +
                  esc(s.failure_type || '-') + '<br>' + esc(s.failure_reason || '-') + '</div>'
                ).join('') + '</div>'
              : '<span class="muted">-</span>';
            return '<tr>' +
              '<td class="mono">' + esc(c.cycle ?? '-') + '</td>' +
              '<td>' + esc(c.result || '-') + '</td>' +
              '<td>' + esc(c.dsl_iterations ?? '-') + '</td>' +
              '<td>' + esc(c.failed_stage || '-') + '</td>' +
              '<td>' + esc(c.failure_type || '-') + '</td>' +
              '<td class="clamp" title="' + esc(c.failure_reason || '-') + '">' + esc(c.failure_reason || '-') + '</td>' +
              '<td>' + stages + '</td>' +
            '</tr>';
          }}).join('')
        : '';
      const cycleSection = cycleRows
        ? '<div class="cycle-detail"><div class="label">Motivo di fail per ciclo</div>' +
          '<table><thead><tr><th>Ciclo</th><th>Risultato</th><th>Iter DSL</th><th>Failed stage</th><th>Tipo</th><th>Motivo ciclo</th><th>Stage falliti</th></tr></thead>' +
          '<tbody>' + cycleRows + '</tbody></table></div>'
        : '<div><div class="label">Motivo di fail per ciclo</div><span class="muted">Questa run non contiene metadata pipeline per ciclo.</span></div>';
      el.detail.innerHTML =
        '<div class="detail-grid">' +
        '<div class="kv"><b>Run</b><span class="mono">' + esc(r.run_id) + '</span></div>' +
        '<div class="kv"><b>Modello</b><span>' + esc(r.model) + '</span></div>' +
        '<div class="kv"><b>Scenario</b><span>' + esc(r.scenario) + '</span></div>' +
        '<div class="kv"><b>Seed</b><span class="mono">' + esc(r.llm_seed ?? '-') + '</span></div>' +
        '<div class="kv"><b>Esito</b><span>' + esc(r.outcome) + '</span></div>' +
        '<div class="kv"><b>Motivo</b><span>' + esc(r.failure_category) + '</span></div>' +
        '<div class="kv"><b>Dettaglio</b><span>' + esc(r.failure_detail) + '</span></div>' +
        '<div class="kv"><b>Cicli</b><span>' + fmtNum(r.cycles) + '</span></div>' +
        '<div class="kv"><b>Durata</b><span>' + fmtDuration(r.duration_seconds) + '</span></div>' +
        '<div class="kv"><b>Tempo gen.</b><span>' + fmtDuration(r.dsl_generation_time_seconds) + '</span></div>' +
        '<div class="kv"><b>Token DSL</b><span>' + (r.dsl_total_tokens === null || r.dsl_total_tokens === undefined ? '-' : fmtNum(r.dsl_total_tokens)) + '</span></div>' +
        '<div class="kv"><b>Metadata</b><span class="mono">' + esc(r.metadata_path) + '</span></div>' +
        '</div>' +
        cycleSection +
        '<div><div class="label">Esempi query fallite</div>' + examples + '</div>';
    }}
    function update() {{
      if (selected && !filteredRows().some(r => r.run_id === selected.run_id)) selected = null;
      renderTop();
      renderCharts();
      renderRows();
    }}
    setupFilters();
    renderTop();
    renderCharts();
    [el.search, el.model, el.scenario, el.outcome, el.reason, el.cycle].forEach(node => {{
      node.addEventListener('input', update);
      node.addEventListener('change', update);
    }});
    renderRows();
  </script>
</body>
</html>
"""


def build_site(runs_dir: Path, output_html: Path, print_summary: bool = False) -> None:
    records = _collect_records(runs_dir)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    figures = _write_boxplot_figures(records, output_html)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs_dir": _safe_rel(runs_dir),
        "summary": _build_summary(records),
        "figures": figures,
        "records": records,
    }
    output_html.write_text(_build_html(payload), encoding="utf-8")
    if print_summary:
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
        if figures:
            print(json.dumps({"figures": figures}, indent=2, ensure_ascii=False))
    print(f"[OK] Analysis written: {output_html}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a model-level HTML analysis for LIRAS runs.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Runs directory to scan")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary to stdout")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    build_site(runs_dir, output, print_summary=args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
