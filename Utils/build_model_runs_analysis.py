#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import re
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


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _usage_reasoning_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        value = _to_int_or_none(details.get("reasoning_tokens"))
        if value is not None:
            return value
    return _to_int_or_none(usage.get("reasoning_tokens"))


def _usage_output_tokens(usage: dict[str, Any]) -> int | None:
    prompt = _to_int_or_none(usage.get("prompt_tokens"))
    total = _to_int_or_none(usage.get("total_tokens"))
    completion = _to_int_or_none(usage.get("completion_tokens"))
    reasoning = _usage_reasoning_tokens(usage)

    # OpenAI-compatible providers report reasoning_tokens as a subset of
    # completion_tokens, not as an additional quantity. Prefer the provider's
    # total-minus-prompt invariant and never add reasoning a second time.
    if prompt is not None and total is not None and total >= prompt and total > 0:
        return total - prompt
    if completion is not None:
        return completion
    return reasoning


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
                prompt_total = int(details.get("prompt_tokens_total") or 0)
                completion_total = int(details.get("completion_tokens_total") or 0)
                total_total = int(details.get("total_tokens_total") or 0)
                totals["prompt_tokens"] += prompt_total
                totals["completion_tokens"] += (
                    total_total - prompt_total
                    if total_total >= prompt_total and total_total > 0
                    else completion_total
                )
                totals["total_tokens"] += total_total
                totals["available_calls"] += int(details.get("token_usage_available_calls") or 0)

    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    if not found and summary:
        found = True
        prompt_total = int(summary.get("prompt_tokens_total") or 0)
        completion_total = int(summary.get("completion_tokens_total") or 0)
        total_total = int(summary.get("total_tokens_total") or 0)
        totals["prompt_tokens"] = prompt_total
        totals["completion_tokens"] = (
            total_total - prompt_total
            if total_total >= prompt_total and total_total > 0
            else completion_total
        )
        totals["total_tokens"] = total_total
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
        "reasoning_tokens": [],
    }
    for row in _read_jsonl(cycle_dir / "hf_debug_responses.jsonl"):
        if _safe_str(row.get("kind")) not in {"generate", "repair"}:
            continue
        response_obj = row.get("response_obj") if isinstance(row.get("response_obj"), dict) else {}
        usage = response_obj.get("usage") if isinstance(response_obj.get("usage"), dict) else {}
        output_tokens = _usage_output_tokens(usage)
        if output_tokens is not None:
            samples["completion_tokens"].append(output_tokens)
        if usage.get("total_tokens") is not None:
            samples["total_tokens"].append(int(usage.get("total_tokens") or 0))
        reasoning_tokens = _usage_reasoning_tokens(usage)
        if reasoning_tokens is not None:
            samples["reasoning_tokens"].append(reasoning_tokens)
    return samples


def _dsl_generation_llm_token_samples(run_dir: Path) -> dict[str, list[int]]:
    samples = {
        "completion_tokens": [],
        "total_tokens": [],
        "reasoning_tokens": [],
    }
    for cycle_dir in sorted(run_dir.glob("ciclo*")):
        if not cycle_dir.is_dir():
            continue
        cycle_samples = _dsl_generation_llm_token_samples_for_cycle(cycle_dir)
        samples["completion_tokens"].extend(cycle_samples["completion_tokens"])
        samples["total_tokens"].extend(cycle_samples["total_tokens"])
        samples["reasoning_tokens"].extend(cycle_samples["reasoning_tokens"])
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
            # Efficiency and token/success metrics measure DSL production only.
            # Query-adaptation calls are intentionally excluded.
            if _safe_str(row.get("kind")) not in {"generate", "repair"}:
                continue
            response_obj = row.get("response_obj") if isinstance(row.get("response_obj"), dict) else {}
            usage = response_obj.get("usage") if isinstance(response_obj.get("usage"), dict) else {}
            if not usage:
                continue
            if usage.get("prompt_tokens") is not None:
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
            usage_output = _usage_output_tokens(usage)
            if usage_output is not None:
                output_tokens += usage_output
                found_output = True
            if usage.get("total_tokens") is not None:
                total_tokens += int(usage.get("total_tokens") or 0)
            usage_reasoning = _usage_reasoning_tokens(usage)
            if usage_reasoning is not None:
                reasoning_tokens += usage_reasoning
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
        # Stage-level DSL telemetry is captured before query adaptation and is
        # therefore the correct fallback when raw Hugging Face logs are absent.
        dsl_totals = _dsl_token_totals(metadata)
        if dsl_totals["completion_tokens"] is not None:
            found_output = True
            totals["prompt_tokens"] = int(dsl_totals["prompt_tokens"] or 0)
            totals["output_tokens"] = int(dsl_totals["completion_tokens"] or 0)
            totals["total_tokens"] = int(dsl_totals["total_tokens"] or 0)

    return {
        "prompt_tokens": totals["prompt_tokens"] if found_output else None,
        "output_tokens": totals["output_tokens"] if found_output else None,
        "total_tokens": totals["total_tokens"] if found_output else None,
        "reasoning_tokens": totals["reasoning_tokens"] if found_reasoning else None,
    }


def _cycle_dir_from_metadata(cycle: dict[str, Any], fallback_run_dir: Path, cycle_index: int) -> Path:
    cycle_number = cycle.get("cycle")
    local_cycle_dir = fallback_run_dir / f"ciclo{cycle_number if cycle_number is not None else cycle_index + 1}"
    if local_cycle_dir.is_dir():
        return local_cycle_dir

    raw_dir = _safe_str(cycle.get("run_dir"))
    if raw_dir and Path(raw_dir).is_dir():
        return Path(raw_dir)
    raw_metadata = _safe_str(cycle.get("metadata_path"))
    if raw_metadata and Path(raw_metadata).parent.is_dir():
        return Path(raw_metadata).parent
    return local_cycle_dir


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
                raw_completion_tokens = int(details.get("completion_tokens_total") or 0)
                total_tokens = int(details.get("total_tokens_total") or 0)
                completion_tokens = (
                    total_tokens - prompt_tokens
                    if total_tokens >= prompt_tokens and total_tokens > 0
                    else raw_completion_tokens
                )

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
                "llm_reasoning_token_samples": dsl_token_samples["reasoning_tokens"],
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


def _first_cycle_result(metadata: dict[str, Any]) -> dict[str, Any]:
    cycles = metadata.get("cycles")
    if not isinstance(cycles, list) or not cycles or not isinstance(cycles[0], dict):
        return {
            "outcome": _outcome_label(metadata),
            "failure_category": "none" if _is_success(metadata) else _safe_str(metadata.get("failure_type"), "unknown"),
            "failure_detail": "No cycle metadata available",
        }

    cycle = cycles[0]
    result = _safe_str(cycle.get("cycle_result"), "unknown").lower()
    if result in {"ok", "success", "success_no_output"}:
        return {
            "outcome": "success",
            "failure_category": "none",
            "failure_detail": "First cycle completed",
        }

    top_details = cycle.get("failure_details") if isinstance(cycle.get("failure_details"), dict) else {}
    category = _safe_str(cycle.get("failure_type") or cycle.get("failed_stage"), "unknown")
    detail = _safe_str(cycle.get("failure_reason") or top_details.get("error_message"), "First cycle failed")
    return {
        "outcome": "failed",
        "failure_category": category,
        "failure_detail": detail,
    }


def _cycle_number_from_dir(path: Path) -> int:
    match = re.search(r"ciclo(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _liras_artifact(metadata: dict[str, Any], run_dir: Path) -> dict[str, str]:
    successful_cycle = metadata.get("successful_cycle")
    candidates: list[Path] = []
    if successful_cycle is not None:
        cycle_dir = run_dir / f"ciclo{successful_cycle}" / "dsl"
        candidates.extend(sorted(cycle_dir.glob("SUCCESS_*.LIRAs")))

    if not candidates:
        candidates.extend(sorted(run_dir.glob("ciclo*/dsl/SUCCESS_*.LIRAs")))
    if not candidates:
        candidates.extend(sorted(run_dir.glob("ciclo*/dsl/*.LIRAs")))

    if not candidates:
        return {"path": "", "code": ""}

    def sort_key(path: Path) -> tuple[int, float, str]:
        return (_cycle_number_from_dir(path), path.stat().st_mtime, path.name)

    selected = sorted(candidates, key=sort_key)[-1]
    try:
        code = selected.read_text(encoding="utf-8", errors="replace")
    except OSError:
        code = ""
    return {"path": _safe_rel(selected), "code": code}


def _build_record(
    metadata: dict[str, Any],
    meta_path: Path,
    runs_dir: Path,
    include_liras_code: bool = False,
) -> dict[str, Any]:
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
    first_cycle = _first_cycle_result(metadata)
    liras_artifact = _liras_artifact(metadata, meta_path.parent) if include_liras_code else {"path": "", "code": ""}

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
        "first_cycle_outcome": first_cycle["outcome"],
        "first_cycle_failure_category": first_cycle["failure_category"],
        "first_cycle_failure_detail": first_cycle["failure_detail"],
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
        "liras_path": liras_artifact["path"],
        "liras_code": liras_artifact["code"],
        "metadata_path": _safe_rel(meta_path),
        "run_dir": _safe_rel(meta_path.parent),
    }


def _collect_records(runs_dir: Path, include_liras_code: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for meta_path in sorted(runs_dir.rglob("run_metadata.json")):
        metadata = _read_json(meta_path)
        if not metadata or _is_cycle_metadata(meta_path, metadata):
            continue
        records.append(_build_record(metadata, meta_path, runs_dir, include_liras_code=include_liras_code))
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
            float(r["dsl_completion_tokens"])
            for r in model_records
            if r.get("dsl_completion_tokens") is not None
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


def _dashboard_name_for_runs_dir(runs_dir: Path) -> str:
    name = runs_dir.name or "Runs"
    labels = {
        "RunsNoCoT": "Runs NoCoT",
        "RunsCoT": "Runs CoT",
        "Runs2shotNoCoT": "Runs 2-shot NoCoT",
        "Runs2ShotCoT": "Runs 2-shot CoT",
    }
    return labels.get(name, name)


def _dashboard_theme_for_runs_dir(runs_dir: Path) -> str:
    name = runs_dir.name.lower()
    if "cot" in name and "nocot" not in name:
        return "red"
    return "blue"


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


def _first_cycle_metric(record: dict[str, Any]) -> dict[str, Any] | None:
    per_cycle = record.get("per_cycle_metrics") if isinstance(record.get("per_cycle_metrics"), list) else []
    for cycle in per_cycle:
        if isinstance(cycle, dict) and cycle.get("cycle_index") == 0:
            return cycle
    for cycle in per_cycle:
        if isinstance(cycle, dict):
            return cycle
    return None


def _simulate_no_feedback_record(record: dict[str, Any]) -> dict[str, Any]:
    first_cycle = _first_cycle_metric(record)
    cycle_failures = record.get("cycle_failures") if isinstance(record.get("cycle_failures"), list) else []
    first_failures = [
        cycle
        for cycle in cycle_failures
        if isinstance(cycle, dict) and int(cycle.get("cycle") or 0) == 1
    ]
    first_failure = first_failures[0] if first_failures else {}
    fallback_outcome = "success" if int(record.get("successful_cycle") or 0) == 1 else "failed"
    outcome = _safe_str(record.get("first_cycle_outcome"), fallback_outcome)
    failed = outcome != "success"

    def first_cycle_scalar(field: str, fallback_field: str | None = None) -> Any:
        if first_cycle and first_cycle.get(field) is not None:
            return first_cycle.get(field)
        total = _to_float_or_none(record.get(field))
        cycles = _to_int_or_none(record.get("cycles")) or 1
        if total is not None:
            if cycles <= 1:
                return record.get(field)
            return _format_number_like_source(total / cycles, record.get(field))
        if fallback_field:
            if first_cycle and first_cycle.get(fallback_field) is not None:
                return first_cycle.get(fallback_field)
            return record.get(fallback_field)
        return None

    def first_cycle_samples(sample_field: str, scalar_field: str) -> list[Any]:
        if first_cycle and isinstance(first_cycle.get(sample_field), list) and first_cycle.get(sample_field):
            return first_cycle.get(sample_field) or []
        cycles = _to_int_or_none(record.get("cycles")) or 1
        if cycles <= 1 and isinstance(record.get(sample_field), list) and record.get(sample_field):
            return record.get(sample_field) or []
        scalar = first_cycle_scalar(scalar_field)
        return _single_sample_list(scalar)

    dsl_time_samples = first_cycle_samples("dsl_generation_time_samples", "dsl_generation_time_seconds")
    dsl_completion_samples = first_cycle_samples("dsl_completion_token_samples", "dsl_completion_tokens")
    dsl_total_samples = first_cycle_samples("dsl_total_token_samples", "dsl_total_tokens")
    simulated = dict(record)
    simulated.update(
        {
            "simulated_no_feedback_loop": True,
            "original_outcome": record.get("outcome"),
            "original_failure_category": record.get("failure_category"),
            "original_failure_detail": record.get("failure_detail"),
            "outcome": outcome,
            "failure_category": (
                _safe_str(record.get("first_cycle_failure_category"))
                or _safe_str(first_failure.get("failure_type") or first_failure.get("failed_stage"))
                or "first_cycle_failed"
                if failed
                else "none"
            ),
            "failure_detail": (
                _safe_str(record.get("first_cycle_failure_detail"))
                or _safe_str(first_failure.get("failure_reason"))
                or "First cycle failed"
                if failed
                else "First cycle completed"
            ),
            "failed_queries": record.get("failed_queries") if failed else 0,
            "cycles": 1 if first_cycle else min(int(record.get("cycles") or 0), 1),
            "dsl_generation_iterations": (
                [first_cycle.get("dsl_iterations")]
                if first_cycle and first_cycle.get("dsl_iterations") is not None
                else []
            ),
            "dsl_generation_time_seconds": first_cycle_scalar("dsl_generation_time_seconds"),
            "dsl_generation_time_samples": dsl_time_samples,
            "dsl_prompt_tokens": first_cycle.get("dsl_prompt_tokens") if first_cycle else record.get("dsl_prompt_tokens"),
            "dsl_completion_tokens": first_cycle_scalar("dsl_completion_tokens"),
            "dsl_total_tokens": first_cycle_scalar("dsl_total_tokens"),
            "dsl_completion_token_samples": dsl_completion_samples,
            "dsl_total_token_samples": dsl_total_samples,
            "llm_output_tokens": first_cycle_scalar("llm_output_tokens", "dsl_completion_tokens"),
            "llm_reasoning_tokens": first_cycle_scalar("llm_reasoning_tokens"),
            "per_cycle_metrics": [first_cycle] if first_cycle else [],
            "cycle_failures": first_failures if failed else [],
        }
    )
    return simulated


def _simulate_no_feedback_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_simulate_no_feedback_record(record) for record in records]


_CYCLE_SUCCESS_RESULTS = {"ok", "success", "success_no_output"}


def _cycle_number(cycle: dict[str, Any], fallback: int = 1) -> int:
    value = _to_int_or_none(cycle.get("cycle"))
    if value is not None:
        return value
    index = _to_int_or_none(cycle.get("cycle_index"))
    return (index + 1) if index is not None else fallback


def _cycle_summary_for(record: dict[str, Any], cycle_number: int) -> dict[str, Any] | None:
    cycle_failures = record.get("cycle_failures") if isinstance(record.get("cycle_failures"), list) else []
    for item in cycle_failures:
        if isinstance(item, dict) and _to_int_or_none(item.get("cycle")) == cycle_number:
            return item
    if _to_int_or_none(record.get("successful_cycle")) == cycle_number:
        return {"cycle": cycle_number, "result": "ok"}
    return None


def _cycle_result(record: dict[str, Any], cycle: dict[str, Any]) -> str:
    summary = _cycle_summary_for(record, _cycle_number(cycle))
    return _safe_str(summary.get("result") if summary else "").lower()


def _first_sample(values: Any) -> Any:
    if isinstance(values, list) and values:
        return values[0]
    return None


def _single_sample_list(value: Any) -> list[Any]:
    return [] if value is None else [value]


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _format_number_like_source(value: float | None, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    return int(value) if value.is_integer() else value


def _syntactic_first_attempt_output_tokens(cycle: dict[str, Any], first_completion: Any) -> Any:
    attempts = _to_int_or_none(cycle.get("dsl_iterations")) or 1
    original_output = _to_float_or_none(cycle.get("llm_output_tokens"))
    first_output = _to_float_or_none(first_completion)
    if attempts <= 1:
        return cycle.get("llm_output_tokens") if original_output is not None else first_completion
    if first_output is not None:
        return _format_number_like_source(first_output, first_completion)
    # Without per-call samples, the DSL-stage total is the closest available
    # fallback. It still excludes query-adaptation usage.
    return cycle.get("dsl_completion_tokens")


def _syntactic_first_attempt_cycle(cycle: dict[str, Any]) -> dict[str, Any]:
    attempts = _to_int_or_none(cycle.get("dsl_iterations")) or 0
    if attempts <= 1:
        view = dict(cycle)
        view["simulated_no_syntactic_feedback_loop"] = True
        return view

    first_time = _first_sample(cycle.get("dsl_generation_time_samples"))
    first_completion = _first_sample(cycle.get("dsl_completion_token_samples"))
    first_total = _first_sample(cycle.get("dsl_total_token_samples"))
    first_reasoning = _first_sample(cycle.get("llm_reasoning_token_samples"))

    view = dict(cycle)
    view.update(
        {
            "simulated_no_syntactic_feedback_loop": True,
            "dsl_iterations": 1,
            "dsl_generation_time_seconds": (
                first_time if first_time is not None else cycle.get("dsl_generation_time_seconds")
            ),
            "dsl_generation_time_samples": _single_sample_list(first_time),
            "dsl_completion_tokens": (
                first_completion if first_completion is not None else cycle.get("dsl_completion_tokens")
            ),
            "dsl_total_tokens": first_total if first_total is not None else cycle.get("dsl_total_tokens"),
            "dsl_completion_token_samples": _single_sample_list(first_completion),
            "dsl_total_token_samples": _single_sample_list(first_total),
            "llm_output_tokens": _syntactic_first_attempt_output_tokens(cycle, first_completion),
            "llm_reasoning_tokens": (
                first_reasoning if first_reasoning is not None else cycle.get("llm_reasoning_tokens")
            ),
        }
    )
    return view


def _syntactic_failure_row(cycle: dict[str, Any]) -> dict[str, Any]:
    attempts = _to_int_or_none(cycle.get("dsl_iterations")) or 0
    cycle_number = _cycle_number(cycle)
    return {
        "cycle": cycle_number,
        "result": "failed",
        "dsl_iterations": 1,
        "failed_stage": "dsl_generation",
        "failure_type": "syntactic_feedback_loop",
        "failure_reason": (
            "The initial DSL candidate did not compile; compiler-guided repair "
            f"was required ({attempts} attempts recorded)."
        ),
        "failed_stages": [
            {
                "stage": "dsl_generation",
                "failure_type": "syntactic_feedback_loop",
                "failure_reason": (
                    "Without syntactic feedback, execution stops after the initial "
                    "rejected DSL candidate."
                ),
            }
        ],
    }


def _aggregate_simulated_cycles(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    time_samples = [
        value
        for cycle in cycles
        for value in (cycle.get("dsl_generation_time_samples") or [])
    ]
    completion_samples = [
        value
        for cycle in cycles
        for value in (cycle.get("dsl_completion_token_samples") or [])
    ]
    total_samples = [
        value
        for cycle in cycles
        for value in (cycle.get("dsl_total_token_samples") or [])
    ]
    iterations = [
        cycle.get("dsl_iterations")
        for cycle in cycles
        if cycle.get("dsl_iterations") is not None
    ]

    def sum_numeric(values: list[Any]) -> int | float | None:
        numbers: list[float] = []
        for value in values:
            try:
                if value is not None:
                    numbers.append(float(value))
            except Exception:
                continue
        if not numbers:
            return None
        total = sum(numbers)
        return int(total) if total.is_integer() else total

    return {
        "cycles": len(cycles),
        "dsl_generation_iterations": iterations,
        "dsl_generation_time_seconds": sum_numeric(time_samples),
        "dsl_generation_time_samples": time_samples,
        "dsl_completion_tokens": sum_numeric(completion_samples),
        "dsl_total_tokens": sum_numeric(total_samples),
        "dsl_completion_token_samples": completion_samples,
        "dsl_total_token_samples": total_samples,
        "llm_output_tokens": sum_numeric([
            cycle.get("llm_output_tokens") if cycle.get("llm_output_tokens") is not None else cycle.get("dsl_completion_tokens")
            for cycle in cycles
        ]),
        "llm_reasoning_tokens": sum_numeric([
            cycle.get("llm_reasoning_tokens")
            for cycle in cycles
        ]),
    }


def _simulate_no_syntactic_feedback_record(record: dict[str, Any]) -> dict[str, Any]:
    per_cycle = record.get("per_cycle_metrics") if isinstance(record.get("per_cycle_metrics"), list) else []
    if not per_cycle:
        return dict(record, simulated_no_syntactic_feedback_loop=True)

    included_cycles: list[dict[str, Any]] = []
    simulated_failures: list[dict[str, Any]] = []
    success_cycle: dict[str, Any] | None = None
    had_syntactic_repair = False

    for index, cycle in enumerate(per_cycle):
        if not isinstance(cycle, dict):
            continue
        view = _syntactic_first_attempt_cycle(cycle)
        included_cycles.append(view)
        attempts = _to_int_or_none(cycle.get("dsl_iterations")) or 1
        if attempts > 1:
            had_syntactic_repair = True
            simulated_failures.append(_syntactic_failure_row(cycle))
            # This cycle reached its later stages only because compiler-guided
            # repair recovered ITER0.  Without syntactic feedback, the run stops
            # here and cannot reuse semantic feedback or candidates recorded in
            # any subsequent cycle.
            break
        result = _cycle_result(record, cycle)
        if result in _CYCLE_SUCCESS_RESULTS:
            success_cycle = view
            break
        summary = _cycle_summary_for(record, _cycle_number(cycle, index + 1))
        if summary and _safe_str(summary.get("result")).lower() == "failed":
            simulated_failures.append(summary)

    aggregate = _aggregate_simulated_cycles(included_cycles)
    if not had_syntactic_repair:
        for field in (
            "dsl_generation_iterations",
            "dsl_generation_time_seconds",
            "dsl_generation_time_samples",
            "dsl_completion_tokens",
            "dsl_total_tokens",
            "dsl_completion_token_samples",
            "dsl_total_token_samples",
            "llm_output_tokens",
            "llm_reasoning_tokens",
        ):
            aggregate[field] = record.get(field)
    failed = success_cycle is None
    last_failure = simulated_failures[-1] if simulated_failures else {}
    simulated = dict(record)
    simulated.update(
        {
            "simulated_no_syntactic_feedback_loop": True,
            "original_outcome": record.get("original_outcome", record.get("outcome")),
            "original_failure_category": record.get("original_failure_category", record.get("failure_category")),
            "original_failure_detail": record.get("original_failure_detail", record.get("failure_detail")),
            "outcome": "failed" if failed else "success",
            "failure_category": (
                _safe_str(last_failure.get("failure_type") or last_failure.get("failed_stage"))
                or "syntactic_feedback_loop"
                if failed
                else "none"
            ),
            "failure_detail": (
                _safe_str(last_failure.get("failure_reason"))
                or "No cycle completed with the first DSL generation attempt"
                if failed
                else "Cycle completed with the first DSL generation attempt"
            ),
            "failed_queries": record.get("failed_queries") if failed else 0,
            "successful_cycle": _cycle_number(success_cycle) if success_cycle else None,
            "per_cycle_metrics": included_cycles,
            "cycle_failures": simulated_failures if failed else [
                failure
                for failure in simulated_failures
                if _safe_str(failure.get("result")).lower() == "failed"
            ],
        }
    )
    simulated.update(aggregate)
    return simulated


def _simulate_no_syntactic_feedback_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_simulate_no_syntactic_feedback_record(record) for record in records]


def _write_boxplot_figures(records: list[dict[str, Any]], output_html: Path, theme: str = "blue") -> dict[str, Any]:
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
        ("dsl_tokens_per_generated_dsl", "Output token per singolo DSL", "output_tokens"),
    ]
    palette = {
        "blue": {
            "box_face": "#bfdbfe",
            "box_edge": "#1d4ed8",
            "median": "#172554",
            "mean_face": "#f59e0b",
            "mean_edge": "#b45309",
            "flier_face": "#fee2e2",
            "flier_edge": "#ef4444",
        },
        "red": {
            "box_face": "#fecaca",
            "box_edge": "#b91c1c",
            "median": "#7f1d1d",
            "mean_face": "#f97316",
            "mean_edge": "#c2410c",
            "flier_face": "#fee2e2",
            "flier_edge": "#dc2626",
        },
    }.get(theme, {})

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
            boxprops={"facecolor": palette.get("box_face", "#bfdbfe"), "edgecolor": palette.get("box_edge", "#1d4ed8"), "linewidth": 1.2},
            medianprops={"color": palette.get("median", "#172554"), "linewidth": 1.8},
            whiskerprops={"color": "#475569", "linewidth": 1.1},
            capprops={"color": "#475569", "linewidth": 1.1},
            meanprops={"marker": "D", "markerfacecolor": palette.get("mean_face", "#f59e0b"), "markeredgecolor": palette.get("mean_edge", "#b45309"), "markersize": 4},
            flierprops={"marker": "o", "markerfacecolor": palette.get("flier_face", "#fee2e2"), "markeredgecolor": palette.get("flier_edge", "#ef4444"), "markersize": 3, "alpha": 0.8},
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
                Patch(facecolor=palette.get("box_face", "#bfdbfe"), edgecolor=palette.get("box_edge", "#1d4ed8"), label="Box Q1-Q3"),
                Line2D([0], [0], color=palette.get("median", "#172554"), linewidth=2, label="Mediana"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor=palette.get("mean_face", "#f59e0b"), markeredgecolor=palette.get("mean_edge", "#b45309"), markersize=5, label="Media"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get("flier_face", "#fee2e2"), markeredgecolor=palette.get("flier_edge", "#ef4444"), markersize=4, label="Outlier"),
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

    def filter_values(source_records: list[dict[str, Any]]) -> dict[str, list[str]]:
        return {
            "model": [""] + sorted({_safe_str(r.get("model")) for r in source_records if _safe_str(r.get("model"))}),
            "scenario": [""] + sorted({_safe_str(r.get("scenario")) for r in source_records if _safe_str(r.get("scenario"))}),
            "outcome": [""] + sorted({_safe_str(r.get("outcome")) for r in source_records if _safe_str(r.get("outcome"))}),
            "reason": [""] + sorted({_safe_str(r.get("failure_category")) for r in source_records if _safe_str(r.get("failure_category"))}),
            "cycle_index": [""] + [
                str(value)
                for value in sorted({
                    int(cycle.get("cycle_index"))
                    for record in source_records
                    for cycle in (record.get("per_cycle_metrics") or [])
                    if isinstance(cycle, dict) and cycle.get("cycle_index") is not None
                })
            ],
        }

    def merged_filter_values(left: dict[str, list[str]], right: dict[str, list[str]]) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for key in ("model", "scenario", "outcome", "reason", "cycle_index"):
            values = {value for value in left.get(key, []) + right.get(key, []) if value}
            if key == "cycle_index":
                merged[key] = [""] + sorted(values, key=lambda value: int(value))
            else:
                merged[key] = [""] + sorted(values)
        return merged

    def build_filter_boxplots(
        source_records: list[dict[str, Any]],
        source_values: dict[str, list[str]],
        *,
        key_prefix: str = "",
        title_prefix: str = "",
    ) -> dict[str, str]:
        filter_boxplots: dict[str, str] = {}
        for model, scenario, outcome, reason, cycle_index in itertools.product(
            source_values["model"],
            source_values["scenario"],
            source_values["outcome"],
            source_values["reason"],
            source_values["cycle_index"],
        ):
            if model:
                continue
            rows = _filter_records(
                source_records,
                model=model,
                scenario=scenario,
                outcome=outcome,
                reason=reason,
                cycle_index=cycle_index,
            )
            if not rows:
                continue
            key = _filter_key(model, scenario, outcome, reason, cycle_index)
            suffix_parts = [
                f"{title_prefix}modello={model or 'tutti'}",
                f"scenario={scenario or 'tutti'}",
                f"esito={outcome or 'tutti'}",
                f"motivo={reason or 'tutti'}",
                f"ciclo={cycle_index if cycle_index else 'tutti'}",
            ]
            filter_boxplots[key] = save_filtered_boxplots(
                rows,
                f"{key_prefix}{key}",
                ", ".join(suffix_parts),
                cycle_index,
            )
        return filter_boxplots

    no_feedback_records = _simulate_no_feedback_records(records)
    no_syntactic_feedback_records = _simulate_no_syntactic_feedback_records(records)
    no_feedback_no_syntactic_records = _simulate_no_syntactic_feedback_records(no_feedback_records)
    values = filter_values(records)
    no_feedback_values = filter_values(no_feedback_records)
    no_syntactic_feedback_values = filter_values(no_syntactic_feedback_records)
    no_feedback_no_syntactic_values = filter_values(no_feedback_no_syntactic_records)

    return {
        "filter_boxplots": build_filter_boxplots(records, values),
        "filter_boxplots_no_feedback": build_filter_boxplots(
            no_feedback_records,
            no_feedback_values,
            key_prefix="no_feedback:",
            title_prefix="no feedback loop, ",
        ),
        "filter_boxplots_no_syntactic_feedback": build_filter_boxplots(
            no_syntactic_feedback_records,
            no_syntactic_feedback_values,
            key_prefix="no_syntactic_feedback:",
            title_prefix="no syntactic feedback loop, ",
        ),
        "filter_boxplots_no_feedback_no_syntactic_feedback": build_filter_boxplots(
            no_feedback_no_syntactic_records,
            no_feedback_no_syntactic_values,
            key_prefix="no_feedback_no_syntactic_feedback:",
            title_prefix="no semantic/syntactic feedback loop, ",
        ),
        "default_filter_key": _filter_key(),
        "filter_dimensions": merged_filter_values(
            merged_filter_values(values, no_feedback_values),
            merged_filter_values(no_syntactic_feedback_values, no_feedback_no_syntactic_values),
        ),
    }


def _build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False).replace("</script", "<\\/script")
    dashboard_name = html.escape(_safe_str(payload.get("dashboard_name"), "Runs"))
    runs_dir_label = html.escape(_safe_str(payload.get("runs_dir"), "Runs"))
    theme = _safe_str(payload.get("dashboard_theme"), "blue")
    if theme == "red":
        accent = "#dc2626"
        accent_soft = "#fee2e2"
        bg_wash_a = "#fee2e2"
        hero_gradient = "linear-gradient(100deg, #dc2626 0%, #7f1d1d 100%)"
        hero_copy = "#fee2e2"
        boxplot_edge = "#b91c1c"
        boxplot_face = "#fecaca"
        boxplot_median = "#7f1d1d"
    else:
        accent = "#2563eb"
        accent_soft = "#dbeafe"
        bg_wash_a = "#dbeafe"
        hero_gradient = "linear-gradient(100deg, #1d4ed8 0%, #172554 100%)"
        hero_copy = "#dbeafe"
        boxplot_edge = "#1d4ed8"
        boxplot_face = "#bfdbfe"
        boxplot_median = "#172554"
    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analisi Run per Modello - {dashboard_name}</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #64748b;
      --line: #dbe3ef;
      --accent: {accent};
      --accent-soft: {accent_soft};
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
        radial-gradient(circle at 80% -10%, {bg_wash_a} 0, {bg_wash_a} 24%, transparent 44%),
        radial-gradient(circle at -10% 110%, #fef3c7 0, #fef3c7 22%, transparent 42%),
        var(--bg);
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 22px; display: grid; gap: 16px; }}
    .hero {{
      background: {hero_gradient};
      color: white;
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.20);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 0; color: {hero_copy}; }}
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
      border: 1px solid {boxplot_edge};
      background: {boxplot_face};
      border-radius: 4px;
    }}
    .boxplot-median {{
      position: absolute;
      top: 6px;
      width: 2px;
      height: 22px;
      background: {boxplot_median};
    }}
    .boxplot-value {{ text-align: right; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted); }}
    .controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; padding: 12px; }}
    .control label {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; margin-bottom: 4px; }}
    .check-control {{ display: flex; flex-direction: column; justify-content: end; gap: 4px; }}
    .check-control .check-label {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
      text-transform: none;
      margin: 0;
    }}
    .check-control input {{ width: auto; }}
    .control-note {{ color: var(--muted); font-size: 11px; line-height: 1.25; }}
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
    .liras-code pre {{
      max-height: 520px;
      white-space: pre;
    }}
    .liras-path {{
      margin-bottom: 6px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{dashboard_name}</h1>
      <p>Analisi run per modello con grafici e boxplot. Cartella sorgente: <span class="mono">{runs_dir_label}</span>.</p>
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
      <div class="control check-control"><label class="check-label"><input id="noFeedbackLoop" type="checkbox" /> No semantic feedback loop</label><div class="control-note">Simula lo stop dopo il primo ciclo semantico, senza rimuovere run.</div></div>
      <div class="control check-control"><label class="check-label"><input id="noSyntacticFeedbackLoop" type="checkbox" /> No syntactic feedback loop</label><div class="control-note">Usa solo ITER0; se il candidato iniziale non compila, interrompe la run e non considera cicli semantici successivi.</div></div>
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
    const includeLirasCode = Boolean(payload.include_liras_code);
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
      noFeedbackLoop: document.getElementById('noFeedbackLoop'),
      noSyntacticFeedbackLoop: document.getElementById('noSyntacticFeedbackLoop'),
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
    function noFeedbackLoopMode() {{
      return Boolean(el.noFeedbackLoop && el.noFeedbackLoop.checked);
    }}
    function noSyntacticFeedbackLoopMode() {{
      return Boolean(el.noSyntacticFeedbackLoop && el.noSyntacticFeedbackLoop.checked);
    }}
    function firstCycleMetric(record) {{
      const perCycle = Array.isArray(record.per_cycle_metrics) ? record.per_cycle_metrics : [];
      return perCycle.find(c => Number(c.cycle_index) === 0) || perCycle[0] || null;
    }}
    function firstCycleFailures(record) {{
      const cycleFailures = Array.isArray(record.cycle_failures) ? record.cycle_failures : [];
      return cycleFailures.filter(c => Number(c.cycle) === 1);
    }}
    function simulatedNoFeedbackRecord(record) {{
      if (!noFeedbackLoopMode()) return record;
      const firstCycle = firstCycleMetric(record);
      const firstFailures = firstCycleFailures(record);
      const firstFailure = firstFailures[0] || null;
      const fallbackOutcome = Number(record.successful_cycle) === 1 ? 'success' : 'failed';
      const outcome = record.first_cycle_outcome || fallbackOutcome;
      const failed = outcome !== 'success';
      const failureCategory = failed
        ? (record.first_cycle_failure_category || (firstFailure && (firstFailure.failure_type || firstFailure.failed_stage)) || 'first_cycle_failed')
        : 'none';
      const failureDetail = failed
        ? (record.first_cycle_failure_detail || (firstFailure && firstFailure.failure_reason) || 'First cycle failed')
        : 'First cycle completed';
      function firstCycleScalar(field, fallbackField = null) {{
        if (firstCycle && firstCycle[field] !== null && firstCycle[field] !== undefined) return firstCycle[field];
        const total = finiteNumberOrNull(record[field]);
        const cycles = Math.max(Number(record.cycles || 1), 1);
        if (total !== null) return cycles <= 1 ? record[field] : total / cycles;
        if (fallbackField) {{
          if (firstCycle && firstCycle[fallbackField] !== null && firstCycle[fallbackField] !== undefined) return firstCycle[fallbackField];
          return record[fallbackField] ?? null;
        }}
        return null;
      }}
      function firstCycleSamples(sampleField, scalarField) {{
        if (firstCycle && Array.isArray(firstCycle[sampleField]) && firstCycle[sampleField].length) return firstCycle[sampleField];
        const cycles = Math.max(Number(record.cycles || 1), 1);
        if (cycles <= 1 && Array.isArray(record[sampleField]) && record[sampleField].length) return record[sampleField];
        const scalar = firstCycleScalar(scalarField);
        return singleSampleList(scalar);
      }}
      const dslTimeSamples = firstCycleSamples('dsl_generation_time_samples', 'dsl_generation_time_seconds');
      const dslCompletionSamples = firstCycleSamples('dsl_completion_token_samples', 'dsl_completion_tokens');
      const dslTotalSamples = firstCycleSamples('dsl_total_token_samples', 'dsl_total_tokens');
      const outputTokens = firstCycleScalar('llm_output_tokens', 'dsl_completion_tokens');
      return {{
        ...record,
        simulated_no_feedback_loop: true,
        original_outcome: record.outcome,
        original_failure_category: record.failure_category,
        original_failure_detail: record.failure_detail,
        outcome,
        failure_category: failureCategory,
        failure_detail: failureDetail,
        failed_queries: failed ? record.failed_queries : 0,
        cycles: firstCycle ? 1 : Math.min(Number(record.cycles || 0), 1),
        dsl_generation_iterations: firstCycle && firstCycle.dsl_iterations !== null && firstCycle.dsl_iterations !== undefined ? [firstCycle.dsl_iterations] : [],
        dsl_generation_time_seconds: firstCycleScalar('dsl_generation_time_seconds'),
        dsl_generation_time_samples: dslTimeSamples,
        dsl_prompt_tokens: firstCycle ? firstCycle.dsl_prompt_tokens : record.dsl_prompt_tokens,
        dsl_completion_tokens: firstCycleScalar('dsl_completion_tokens'),
        dsl_total_tokens: firstCycleScalar('dsl_total_tokens'),
        dsl_completion_token_samples: dslCompletionSamples,
        dsl_total_token_samples: dslTotalSamples,
        llm_output_tokens: outputTokens,
        llm_reasoning_tokens: firstCycleScalar('llm_reasoning_tokens'),
        per_cycle_metrics: firstCycle ? [firstCycle] : [],
        cycle_failures: failed ? firstFailures : [],
      }};
    }}
    const cycleSuccessResults = new Set(['ok', 'success', 'success_no_output']);
    function cycleNumber(cycle, fallback = 1) {{
      const direct = Number(cycle && cycle.cycle);
      if (Number.isFinite(direct) && direct > 0) return direct;
      const index = Number(cycle && cycle.cycle_index);
      return Number.isFinite(index) ? index + 1 : fallback;
    }}
    function cycleSummaryFor(record, cycleNum) {{
      const cycleFailures = Array.isArray(record.cycle_failures) ? record.cycle_failures : [];
      const found = cycleFailures.find(c => Number(c.cycle) === Number(cycleNum));
      if (found) return found;
      if (Number(record.successful_cycle) === Number(cycleNum)) return {{cycle: cycleNum, result: 'ok'}};
      return null;
    }}
    function cycleResult(record, cycle) {{
      const summary = cycleSummaryFor(record, cycleNumber(cycle));
      return String(summary && summary.result ? summary.result : '').toLowerCase();
    }}
    function firstSample(values) {{
      return Array.isArray(values) && values.length ? values[0] : null;
    }}
    function singleSampleList(value) {{
      return value === null || value === undefined ? [] : [value];
    }}
    function finiteNumberOrNull(value) {{
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }}
    function syntacticFirstAttemptOutputTokens(cycle, firstCompletion) {{
      const attempts = Number(cycle.dsl_iterations || 1);
      const originalOutput = finiteNumberOrNull(cycle.llm_output_tokens);
      const firstOutput = finiteNumberOrNull(firstCompletion);
      if (attempts <= 1) return originalOutput !== null ? cycle.llm_output_tokens : firstCompletion;
      return firstOutput !== null ? firstCompletion : cycle.dsl_completion_tokens;
    }}
    function syntacticFirstAttemptCycle(cycle) {{
      const attempts = Number(cycle.dsl_iterations || 0);
      if (attempts <= 1) {{
        return {{...cycle, simulated_no_syntactic_feedback_loop: true}};
      }}
      const firstTime = firstSample(cycle.dsl_generation_time_samples);
      const firstCompletion = firstSample(cycle.dsl_completion_token_samples);
      const firstTotal = firstSample(cycle.dsl_total_token_samples);
      const firstReasoning = firstSample(cycle.llm_reasoning_token_samples);
      return {{
        ...cycle,
        simulated_no_syntactic_feedback_loop: true,
        dsl_iterations: 1,
        dsl_generation_time_seconds: firstTime ?? cycle.dsl_generation_time_seconds,
        dsl_generation_time_samples: singleSampleList(firstTime),
        dsl_completion_tokens: firstCompletion ?? cycle.dsl_completion_tokens,
        dsl_total_tokens: firstTotal ?? cycle.dsl_total_tokens,
        dsl_completion_token_samples: singleSampleList(firstCompletion),
        dsl_total_token_samples: singleSampleList(firstTotal),
        llm_output_tokens: syntacticFirstAttemptOutputTokens(cycle, firstCompletion),
        llm_reasoning_tokens: firstReasoning ?? cycle.llm_reasoning_tokens,
      }};
    }}
    function syntacticFailureRow(cycle) {{
      const attempts = Number(cycle.dsl_iterations || 0);
      return {{
        cycle: cycleNumber(cycle),
        result: 'failed',
        dsl_iterations: 1,
        failed_stage: 'dsl_generation',
        failure_type: 'syntactic_feedback_loop',
        failure_reason: 'The initial DSL candidate did not compile; compiler-guided repair was required (' + attempts + ' attempts recorded).',
        failed_stages: [{{
          stage: 'dsl_generation',
          failure_type: 'syntactic_feedback_loop',
          failure_reason: 'Without syntactic feedback, execution stops after the initial rejected DSL candidate.',
        }}],
      }};
    }}
    function sumNumeric(values) {{
      const nums = values.map(v => Number(v)).filter(v => Number.isFinite(v));
      if (!nums.length) return null;
      return nums.reduce((a, b) => a + b, 0);
    }}
    function aggregateSimulatedCycles(cycles) {{
      const timeSamples = cycles.flatMap(c => Array.isArray(c.dsl_generation_time_samples) ? c.dsl_generation_time_samples : []);
      const completionSamples = cycles.flatMap(c => Array.isArray(c.dsl_completion_token_samples) ? c.dsl_completion_token_samples : []);
      const totalSamples = cycles.flatMap(c => Array.isArray(c.dsl_total_token_samples) ? c.dsl_total_token_samples : []);
      return {{
        cycles: cycles.length,
        dsl_generation_iterations: cycles.map(c => c.dsl_iterations).filter(v => v !== null && v !== undefined),
        dsl_generation_time_seconds: sumNumeric(timeSamples),
        dsl_generation_time_samples: timeSamples,
        dsl_completion_tokens: sumNumeric(completionSamples),
        dsl_total_tokens: sumNumeric(totalSamples),
        dsl_completion_token_samples: completionSamples,
        dsl_total_token_samples: totalSamples,
        llm_output_tokens: sumNumeric(cycles.map(c => c.llm_output_tokens ?? c.dsl_completion_tokens)),
        llm_reasoning_tokens: sumNumeric(cycles.map(c => c.llm_reasoning_tokens)),
      }};
    }}
    function simulatedNoSyntacticFeedbackRecord(record) {{
      if (!noSyntacticFeedbackLoopMode()) return record;
      const perCycle = Array.isArray(record.per_cycle_metrics) ? record.per_cycle_metrics : [];
      if (!perCycle.length) {{
        return {{...record, simulated_no_syntactic_feedback_loop: true}};
      }}
      const includedCycles = [];
      const simulatedFailures = [];
      let successCycle = null;
      let hadSyntacticRepair = false;
      for (let index = 0; index < perCycle.length; index += 1) {{
        const cycle = perCycle[index];
        if (!cycle) continue;
        const view = syntacticFirstAttemptCycle(cycle);
        includedCycles.push(view);
        const attempts = Number(cycle.dsl_iterations || 1);
        if (attempts > 1) {{
          hadSyntacticRepair = true;
          simulatedFailures.push(syntacticFailureRow(cycle));
          // Later cycles depend on the repaired candidate reaching semantic
          // verification, so they are unavailable when syntactic repair is off.
          break;
        }}
        const result = cycleResult(record, cycle);
        if (cycleSuccessResults.has(result)) {{
          successCycle = view;
          break;
        }}
        const summary = cycleSummaryFor(record, cycleNumber(cycle, index + 1));
        if (summary && String(summary.result || '').toLowerCase() === 'failed') simulatedFailures.push(summary);
      }}
      const aggregate = aggregateSimulatedCycles(includedCycles);
      if (!hadSyntacticRepair) {{
        for (const field of [
          'dsl_generation_iterations',
          'dsl_generation_time_seconds',
          'dsl_generation_time_samples',
          'dsl_completion_tokens',
          'dsl_total_tokens',
          'dsl_completion_token_samples',
          'dsl_total_token_samples',
          'llm_output_tokens',
          'llm_reasoning_tokens',
        ]) {{
          aggregate[field] = record[field];
        }}
      }}
      const failed = !successCycle;
      const lastFailure = simulatedFailures.length ? simulatedFailures[simulatedFailures.length - 1] : null;
      return {{
        ...record,
        ...aggregate,
        simulated_no_syntactic_feedback_loop: true,
        original_outcome: record.original_outcome || record.outcome,
        original_failure_category: record.original_failure_category || record.failure_category,
        original_failure_detail: record.original_failure_detail || record.failure_detail,
        outcome: failed ? 'failed' : 'success',
        failure_category: failed ? ((lastFailure && (lastFailure.failure_type || lastFailure.failed_stage)) || 'syntactic_feedback_loop') : 'none',
        failure_detail: failed ? ((lastFailure && lastFailure.failure_reason) || 'No cycle completed with the first DSL generation attempt') : 'Cycle completed with the first DSL generation attempt',
        failed_queries: failed ? record.failed_queries : 0,
        successful_cycle: successCycle ? cycleNumber(successCycle) : null,
        per_cycle_metrics: includedCycles,
        cycle_failures: failed ? simulatedFailures : simulatedFailures.filter(c => String(c.result || '').toLowerCase() === 'failed'),
      }};
    }}
    function activeRecords() {{
      return records.map(r => simulatedNoSyntacticFeedbackRecord(simulatedNoFeedbackRecord(r)));
    }}
    function summarizeRecords(rows) {{
      const byModel = new Map();
      const selectedCycle = el.cycle && el.cycle.value !== '' ? Number(el.cycle.value) : null;
      for (const r of rows) {{
        const key = r.model || 'unknown';
        if (!byModel.has(key)) {{
          byModel.set(key, {{model: key, total: 0, success: 0, failed: 0, running: 0, unknown: 0, cycles: [], totalTokens: 0, dslTimes: [], completionTokenSamples: [], feedbackCycleErrors: {{}}}});
        }}
        const item = byModel.get(key);
        item.total += 1;
        if (r.outcome === 'success') item.success += 1;
        else if (r.outcome === 'failed') item.failed += 1;
        else if (r.outcome === 'running') item.running += 1;
        else item.unknown += 1;
        const cycles = Number(r.cycles);
        if (Number.isFinite(cycles)) item.cycles.push(cycles);
        const totalTokens = Number(r.llm_output_tokens ?? r.dsl_completion_tokens);
        if (Number.isFinite(totalTokens) && totalTokens > 0) item.totalTokens += totalTokens;
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
        total_tokens_spent: item.totalTokens,
        efficiency_metric: item.totalTokens > 0 ? item.success / item.totalTokens : null,
        efficiency_per_million_tokens: item.totalTokens > 0 ? item.success / (item.totalTokens / 1000000) : null,
        tokens_per_success: item.success > 0 && item.totalTokens > 0 ? item.totalTokens / item.success : null,
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
          '<td>' + (row.tokens_per_success === null || row.tokens_per_success === undefined ? '-' : fmtNum(Math.round(row.tokens_per_success))) + '</td>' +
          '<td>' + (row.efficiency_per_million_tokens === null || row.efficiency_per_million_tokens === undefined ? '-' : Number(row.efficiency_per_million_tokens).toFixed(2)) + '</td>' +
        '</tr>';
      }}).join('');
      const activeModes = [];
      if (noFeedbackLoopMode()) activeModes.push('No semantic feedback loop: usa solo il primo ciclo semantico.');
      if (noSyntacticFeedbackLoopMode()) activeModes.push('No syntactic feedback loop: usa solo ITER0 e interrompe la run quando il candidato iniziale non compila.');
      const modeNote = activeModes.length
        ? ' Modalita simulate attive: ' + activeModes.join(' ') + ' Le run non vengono rimosse.'
        : '';
      return '<article class="panel chart model-summary"><h2>Riepilogo per modello</h2>' +
        '<div class="chart-note">Sintesi numerica delle run: output token medio calcolato sui singoli DSL generati. Token/successo = token di output generate/repair consumati dalle run visibili / successi; il reasoning e incluso una sola volta, mentre prompt token e query adaptation sono esclusi. Le run fallite contribuiscono al denominatore token quando incluse dai filtri.' + modeNote + '</div>' +
        '<div class="summary-table"><table><thead><tr>' +
        '<th>Modello</th><th>Successi</th><th>Falliti</th><th>Cicli medi/run</th><th>Tempo medio/chiamata DSL</th><th>Output token medi/DSL</th><th>Token/successo</th><th>Eff. successi/1M token</th>' +
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
      return JSON.stringify([el.model.value || '', el.scenario.value || '', el.outcome.value || '', el.reason.value || '', noFeedbackLoopMode() ? '' : (el.cycle.value || '')]);
    }}
    function boxplotFigureForCurrentFilters() {{
      let plots = figures.filter_boxplots || {{}};
      if (noFeedbackLoopMode() && noSyntacticFeedbackLoopMode()) {{
        plots = figures.filter_boxplots_no_feedback_no_syntactic_feedback || {{}};
      }} else if (noFeedbackLoopMode()) {{
        plots = figures.filter_boxplots_no_feedback || {{}};
      }} else if (noSyntacticFeedbackLoopMode()) {{
        plots = figures.filter_boxplots_no_syntactic_feedback || {{}};
      }}
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
        const simulatedNotes = [];
        if (noFeedbackLoopMode()) simulatedNotes.push('usa solo il primo ciclo semantico; successi tardivi diventano fallimenti');
        if (noSyntacticFeedbackLoopMode()) simulatedNotes.push('usa solo ITER0; un candidato iniziale non compilabile termina la run e rende indisponibili i cicli semantici successivi');
        const boxplotNote = simulatedNotes.length
          ? 'Boxplot simulati: ' + simulatedNotes.join('; ') + '. I token sono output token generate/repair, reasoning incluso una sola volta; prompt token e query adaptation esclusi.' + searchNote
          : 'Ogni boxplot usa i datapoint reali: cicli per run, DSL generati per ciclo, tempo di ogni chiamata generate/repair, e output token di ogni DSL generato. Il reasoning e contato come output; i prompt token sono esclusi. Query adaptation escluse.' + searchNote;
        chartBlocks.push(
          figureChart(
            'Distribuzioni per modello',
            boxplotNote,
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
      if (noFeedbackLoopMode()) card('Modalità', 'No semantic feedback loop');
      if (noSyntacticFeedbackLoopMode()) card('Modalità sintattica', 'Solo ITER0');
      el.modelCards.innerHTML = models.map(m => {{
        const cycleErrors = Object.entries(m.feedback_cycle_errors || {{}})
          .sort((a, b) => b[1] - a[1])
          .map(([name, count]) => '<span class="chip">' + esc(name) + ': ' + fmtNum(count) + '</span>')
          .join('');
        const failed = Number(m.failed || 0);
        const success = Number(m.success || 0);
        const total = Number(m.total || 0);
        const totalCycles = Number(m.total_cycles || 0);
        const tokenEfficiency = m.efficiency_per_million_tokens === null || m.efficiency_per_million_tokens === undefined ? '-' : Number(m.efficiency_per_million_tokens).toFixed(2);
        const tokensPerSuccess = m.tokens_per_success === null || m.tokens_per_success === undefined ? '-' : fmtNum(Math.round(m.tokens_per_success));
        return '<article class="model-card">' +
          '<div class="model-head"><div class="model-name">' + esc(m.model) + '</div><div class="rate">' + fmtNum(success) + '/' + fmtNum(total) + '</div></div>' +
          '<div>Cicli totali: <b>' + fmtNum(totalCycles) + '</b></div>' +
          '<div>Token/successo: <b>' + esc(tokensPerSuccess) + '</b></div>' +
          '<div>Efficienza token: <b>' + esc(tokenEfficiency) + '</b> successi/1M token</div>' +
          '<div class="label">Errori cicli feedback</div>' +
          '<div class="reason-list">' + (cycleErrors || '<span class="chip">nessun ciclo fallito</span>') + '</div>' +
          '</article>';
      }}).join('');
    }}
    function setupFilters() {{
      const dims = figures.filter_dimensions || {{}};
      const outcomeValues = records.flatMap(r => [r.outcome, r.first_cycle_outcome]).concat(dims.outcome || []);
      const reasonValues = records.flatMap(r => [r.failure_category, r.first_cycle_failure_category]).concat(dims.reason || []);
      fillSelect(el.model, uniq(records.map(r => r.model).concat(dims.model || [])), 'Tutti');
      fillSelect(el.scenario, uniq(records.map(r => r.scenario).concat(dims.scenario || [])), 'Tutti');
      fillSelect(el.outcome, uniq(outcomeValues), 'Tutti');
      fillSelect(el.reason, uniq(reasonValues), 'Tutti');
      fillCycleSelect();
    }}
    function filteredRows() {{
      const q = el.search.value.trim().toLowerCase();
      return activeRecords().filter(r => {{
        if (el.model.value && r.model !== el.model.value) return false;
        if (el.scenario.value && r.scenario !== el.scenario.value) return false;
        if (el.outcome.value && r.outcome !== el.outcome.value) return false;
        if (el.reason.value && r.failure_category !== el.reason.value) return false;
        if (!noFeedbackLoopMode() && el.cycle.value) {{
          const selectedCycle = Number(el.cycle.value);
          const perCycle = Array.isArray(r.per_cycle_metrics) ? r.per_cycle_metrics : [];
          if (!perCycle.some(c => Number(c.cycle_index) === selectedCycle)) return false;
        }}
        if (!q) return true;
        return [r.run_id, r.model, r.scenario, r.outcome, r.failure_category, r.failure_detail, r.llm_seed, r.metadata_path, includeLirasCode ? r.liras_path : '']
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
          selected = filteredRows().find(r => r.run_id === tr.dataset.run);
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
      const lirasSection = includeLirasCode
        ? (r.liras_code
          ? '<div class="liras-code"><div class="label">Codice LIRAs</div>' +
            '<div class="mono liras-path">' + esc(r.liras_path || '-') + '</div>' +
            '<pre>' + esc(r.liras_code) + '</pre></div>'
          : '<div><div class="label">Codice LIRAs</div><span class="muted">Nessun file .LIRAs trovato per questa run.</span></div>')
        : '';
      const lirasPathRow = includeLirasCode
        ? '<div class="kv"><b>LIRAs</b><span class="mono">' + esc(r.liras_path || '-') + '</span></div>'
        : '';
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
        '<div class="kv"><b>Output token DSL</b><span>' + (r.dsl_completion_tokens === null || r.dsl_completion_tokens === undefined ? '-' : fmtNum(r.dsl_completion_tokens)) + '</span></div>' +
        lirasPathRow +
        '<div class="kv"><b>Metadata</b><span class="mono">' + esc(r.metadata_path) + '</span></div>' +
        '</div>' +
        cycleSection +
        '<div><div class="label">Esempi query fallite</div>' + examples + '</div>' +
        lirasSection;
    }}
    function update() {{
      if (el.cycle) el.cycle.disabled = noFeedbackLoopMode();
      const visibleRows = filteredRows();
      if (selected) selected = visibleRows.find(r => r.run_id === selected.run_id) || null;
      renderTop();
      renderCharts();
      renderRows();
      if (selected) renderDetail(selected);
    }}
    setupFilters();
    if (el.cycle) el.cycle.disabled = noFeedbackLoopMode();
    renderTop();
    renderCharts();
    [el.search, el.model, el.scenario, el.outcome, el.reason, el.cycle, el.noFeedbackLoop, el.noSyntacticFeedbackLoop].forEach(node => {{
      node.addEventListener('input', update);
      node.addEventListener('change', update);
    }});
    renderRows();
  </script>
</body>
</html>
"""


def build_site(
    runs_dir: Path,
    output_html: Path,
    print_summary: bool = False,
    include_liras_code: bool = False,
    dashboard_name_override: str = "",
    dashboard_theme_override: str = "",
) -> None:
    records = _collect_records(runs_dir, include_liras_code=include_liras_code)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    dashboard_name = dashboard_name_override or _dashboard_name_for_runs_dir(runs_dir)
    dashboard_theme = dashboard_theme_override or _dashboard_theme_for_runs_dir(runs_dir)
    figures = _write_boxplot_figures(records, output_html, theme=dashboard_theme)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs_dir": _safe_rel(runs_dir),
        "dashboard_name": dashboard_name,
        "dashboard_theme": dashboard_theme,
        "summary": _build_summary(records),
        "figures": figures,
        "include_liras_code": include_liras_code,
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
    parser.add_argument(
        "--include-liras-code",
        action="store_true",
        help="Embed the selected .LIRAs artifact for each run in the HTML detail panel",
    )
    parser.add_argument("--dashboard-name", default="", help="Override the dashboard title shown in the HTML")
    parser.add_argument(
        "--dashboard-theme",
        choices=("blue", "red"),
        default="",
        help="Override dashboard color theme",
    )
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
    build_site(
        runs_dir,
        output,
        print_summary=args.summary,
        include_liras_code=args.include_liras_code,
        dashboard_name_override=args.dashboard_name,
        dashboard_theme_override=args.dashboard_theme,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
