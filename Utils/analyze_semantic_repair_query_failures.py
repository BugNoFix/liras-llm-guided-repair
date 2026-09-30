#!/usr/bin/env python3
"""Count source queries that fail semantic verification in executed cycles.

The script creates one CSV for each run group (CoT, NoCoT, 2-shot CoT and
2-shot NoCoT).  Every CSV contains all source queries for the scenarios found
in that run group, including queries whose counter is zero.

A source query is counted once for a cycle when both these conditions hold:

1. it occurs in ``verifyta_adapted_analysis.failed_queries``;
2. its ``probability_delta`` is strictly greater than its threshold;

The last available semantic cycle is included even though its failures cannot
start another repair cycle.  The resulting counts therefore describe query-
level verification failures, not semantic-repair transitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOTS = (
    "RunsCoT",
    "RunsNoCoT",
    "Runs2ShotCoT",
    "Runs2shotNoCot",
)
DEFAULT_QUERIES_DIR = PROJECT_ROOT / "Queries"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Report" / "Tables"

QUERY_HEADER_RE = re.compile(r"(?m)^###\s+Query\s+(\d+)\s*$")
CYCLE_DIR_RE = re.compile(r"^ciclo(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceQuery:
    scenario: str
    index: int
    description: str
    expected_probability: float | None
    original_formula: str
    source_file: Path


@dataclass(frozen=True)
class RunGroupSummary:
    run_root: Path
    output_path: Path
    run_count: int
    failure_cycle_count: int
    query_failure_count: int
    source_query_count: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scenario_key(value: Any) -> str:
    """Return the query-template stem from values such as ``name.txt``."""
    text = str(value or "").strip()
    return Path(text).stem if text else ""


def _extract_markdown_field(block: str, label: str) -> str:
    match = re.search(rf"(?m)^\*\*{re.escape(label)}:\*\*\s*(.*)$", block)
    return match.group(1).strip() if match else ""


def _parse_source_queries(path: Path, scenario: str) -> list[SourceQuery]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read source-query file {path}: {exc}") from exc

    headers = list(QUERY_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError(f"No '### Query N' blocks found in {path}")

    queries: list[SourceQuery] = []
    seen_indexes: set[int] = set()
    for position, header in enumerate(headers):
        index = int(header.group(1))
        if index in seen_indexes:
            raise ValueError(f"Duplicate Query {index} in {path}")
        seen_indexes.add(index)

        end = headers[position + 1].start() if position + 1 < len(headers) else len(text)
        block = text[header.end() : end].strip()
        description = _extract_markdown_field(block, "Description")
        expected_probability = _as_float(_extract_markdown_field(block, "Expected probability"))

        formula_match = re.search(
            r"(?ms)^\*\*Original formula:\*\*\s*\n?(.*?)(?=^\*\*[^\n]+:\*\*|\Z)",
            block,
        )
        original_formula = formula_match.group(1).strip() if formula_match else ""
        if not original_formula:
            raise ValueError(f"Missing original formula for Query {index} in {path}")

        queries.append(
            SourceQuery(
                scenario=scenario,
                index=index,
                description=description,
                expected_probability=expected_probability,
                original_formula=original_formula,
                source_file=path,
            )
        )

    return queries


def _top_level_run_metadata(run_root: Path) -> Iterable[Path]:
    """Yield only ``RUN_*/run_metadata.json``, excluding ciclo metadata."""
    for path in sorted(run_root.rglob("run_metadata.json")):
        if path.parent.name.startswith("RUN_"):
            yield path


def _cycle_metadata_by_number(run_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = CYCLE_DIR_RE.fullmatch(child.name)
        if not match:
            continue
        metadata_path = child / "run_metadata.json"
        if metadata_path.is_file():
            result[int(match.group(1))] = metadata_path
    return result


def _adapted_failed_queries(cycle: dict[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
    """Read adapted failures, with a stage-level fallback for older metadata."""
    analysis = cycle.get("verifyta_adapted_analysis")
    if isinstance(analysis, dict):
        failed = analysis.get("failed_queries")
        if isinstance(failed, list):
            queries = [item for item in failed if isinstance(item, dict)]
            threshold = _as_float(analysis.get("probability_threshold"))
            if threshold is None:
                threshold = _as_float(cycle.get("verifyta_probability_delta_threshold"))
            return queries, threshold

    stages = cycle.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict) or stage.get("stage") != "verifyta_adapted":
                continue
            failure_details = stage.get("failure_details")
            details = stage.get("details")
            failed = failure_details.get("failed_queries") if isinstance(failure_details, dict) else None
            if not isinstance(failed, list):
                continue
            threshold = details.get("probability_threshold") if isinstance(details, dict) else None
            if _as_float(threshold) is None:
                threshold = cycle.get("verifyta_probability_delta_threshold")
            return [item for item in failed if isinstance(item, dict)], _as_float(threshold)

    return [], _as_float(cycle.get("verifyta_probability_delta_threshold"))


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _safe_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "runs"


def analyze_run_group(run_root: Path, queries_dir: Path, output_dir: Path) -> RunGroupSummary:
    if not run_root.is_dir():
        raise ValueError(f"Run root does not exist or is not a directory: {run_root}")
    if not queries_dir.is_dir():
        raise ValueError(f"Queries directory does not exist: {queries_dir}")

    top_level_metadata = list(_top_level_run_metadata(run_root))
    if not top_level_metadata:
        raise ValueError(f"No top-level RUN_*/run_metadata.json found under {run_root}")

    scenario_names: set[str] = set()
    failure_counts: Counter[tuple[str, int]] = Counter()
    failure_cycle_count = 0

    for top_path in top_level_metadata:
        run = _read_json(top_path)
        scenario = _scenario_key(run.get("scenario"))
        if not scenario:
            raise ValueError(f"Missing scenario in {top_path}")
        scenario_names.add(scenario)

        cycle_paths = _cycle_metadata_by_number(top_path.parent)
        for _, cycle_path in sorted(cycle_paths.items()):
            cycle = _read_json(cycle_path)
            failed_queries, default_threshold = _adapted_failed_queries(cycle)
            failing_indexes: set[int] = set()
            for failed_query in failed_queries:
                delta = _as_float(failed_query.get("probability_delta"))
                threshold = _as_float(failed_query.get("probability_threshold"))
                if threshold is None:
                    threshold = default_threshold
                query_index = _as_int(failed_query.get("index"))
                if delta is None or threshold is None or query_index is None:
                    continue
                if delta > threshold:
                    failing_indexes.add(query_index)

            if failing_indexes:
                failure_cycle_count += 1
                for query_index in failing_indexes:
                    failure_counts[(scenario, query_index)] += 1

    source_queries: list[SourceQuery] = []
    source_keys: set[tuple[str, int]] = set()
    for scenario in sorted(scenario_names):
        source_path = queries_dir / f"{scenario}.j2"
        if not source_path.is_file():
            raise ValueError(f"Source-query file not found for scenario {scenario}: {source_path}")
        parsed = _parse_source_queries(source_path, scenario)
        source_queries.extend(parsed)
        source_keys.update((query.scenario, query.index) for query in parsed)

    unmatched = sorted(set(failure_counts) - source_keys)
    if unmatched:
        preview = ", ".join(f"{scenario} Query {index}" for scenario, index in unmatched[:10])
        raise ValueError(f"Failed-query indexes not found in source query files: {preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _safe_filename_component(run_root.name)
    output_path = output_dir / f"semantic_query_failure_counts_{suffix}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scenario",
                "query_index",
                "description",
                "expected_probability",
                "original_formula",
                "query_failure_count",
            ),
        )
        writer.writeheader()
        for query in sorted(source_queries, key=lambda item: (item.scenario, item.index)):
            writer.writerow(
                {
                    "scenario": query.scenario,
                    "query_index": query.index,
                    "description": query.description,
                    "expected_probability": (
                        "" if query.expected_probability is None else query.expected_probability
                    ),
                    "original_formula": query.original_formula,
                    "query_failure_count": failure_counts[(query.scenario, query.index)],
                }
            )

    return RunGroupSummary(
        run_root=run_root,
        output_path=output_path,
        run_count=len(top_level_metadata),
        failure_cycle_count=failure_cycle_count,
        query_failure_count=sum(failure_counts.values()),
        source_query_count=len(source_queries),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count source queries whose probability delta exceeded the threshold "
            "in every executed semantic cycle, including the last cycle."
        )
    )
    parser.add_argument(
        "--runs-root",
        action="append",
        dest="run_roots",
        metavar="DIR",
        help=(
            "Run-group directory to analyze. Repeat the option for multiple groups. "
            "Default: RunsCoT, RunsNoCoT, Runs2ShotCoT, Runs2shotNoCot."
        ),
    )
    parser.add_argument(
        "--queries-dir",
        default=str(DEFAULT_QUERIES_DIR),
        metavar="DIR",
        help="Directory containing the scenario .j2 source-query files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help="Destination directory for the per-run-group CSV files.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    requested_roots = args.run_roots or list(DEFAULT_RUN_ROOTS)
    run_roots = [_resolve_path(value) for value in requested_roots]
    queries_dir = _resolve_path(args.queries_dir)
    output_dir = _resolve_path(args.output_dir)

    summaries = [
        analyze_run_group(run_root, queries_dir, output_dir)
        for run_root in run_roots
    ]
    for summary in summaries:
        print(
            f"[OK] {summary.run_root.name}: runs={summary.run_count}, "
            f"source_queries={summary.source_query_count}, "
            f"cycles_with_query_failures={summary.failure_cycle_count}, "
            f"query_failures={summary.query_failure_count} -> {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
