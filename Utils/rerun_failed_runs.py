#!/usr/bin/env python3
"""Rerun failed LIRAS pipeline runs with their recorded settings.

By default this keeps the old failed runs and creates fresh RUN_* folders next
to them. Pass --replace-old to delete each old run after a new run is created.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_runner import _load_json, _run_pipeline  # noqa: E402
from Utils.rerun_same_run import (  # noqa: E402
    _build_rerun_config,
    _default_runs_root,
    _first_cycle_metadata,
    _read_json,
    _run_dirs,
    _safe_rel,
)


def _as_root(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _find_pipeline_run_dirs(roots: list[Path]) -> list[Path]:
    runs: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for metadata_path in root.rglob("run_metadata.json"):
            run_dir = metadata_path.parent
            if run_dir.name.startswith("RUN_"):
                runs.add(run_dir.resolve())
    return sorted(runs)


def _is_failed_run(metadata: dict[str, Any]) -> bool:
    overall = metadata.get("overall_result")
    if isinstance(overall, str) and overall.strip():
        return overall.strip().lower() not in {"ok", "success", "succeeded"}

    status = metadata.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip().lower() not in {"ok", "success", "succeeded"}

    derived = metadata.get("derived")
    if isinstance(derived, dict) and "success" in derived:
        return not bool(derived.get("success"))

    return False


def _failure_text(metadata: dict[str, Any]) -> str:
    parts = [
        metadata.get("overall_result"),
        metadata.get("status"),
        metadata.get("failed_stage"),
        metadata.get("failure_type"),
        metadata.get("failure_reason"),
    ]
    failure_details = metadata.get("failure_details")
    if failure_details is not None:
        parts.append(json.dumps(failure_details, ensure_ascii=False, default=str))
    return " ".join(str(part) for part in parts if part is not None)


def _candidate_failed_runs(
    roots: list[Path],
    *,
    failure_contains: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    needle = failure_contains.lower() if failure_contains else None
    for run_dir in _find_pipeline_run_dirs(roots):
        metadata_path = run_dir / "run_metadata.json"
        try:
            metadata = _read_json(metadata_path)
        except Exception as exc:
            print(f"[RERUN_FAILED] skip unreadable metadata: {_safe_rel(metadata_path)} ({exc})")
            continue
        if not _is_failed_run(metadata):
            continue
        if needle and needle not in _failure_text(metadata).lower():
            continue
        candidates.append((run_dir, metadata))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun failed LIRAS pipeline runs using settings recovered from run metadata."
    )
    parser.add_argument("--config", default="config.json", help="Base config for settings missing from metadata.")
    parser.add_argument(
        "--runs-root",
        action="append",
        default=None,
        help="Root to scan for failed runs. Can be passed more than once. Default: Runs.",
    )
    parser.add_argument("--failure-contains", help="Only rerun failures whose metadata contains this text.")
    parser.add_argument("--limit", type=int, help="Maximum number of failed runs to rerun.")
    parser.add_argument("--dry-run", action="store_true", help="List failed runs without launching anything.")
    parser.add_argument(
        "--replace-old",
        action="store_true",
        help="Delete each old failed run after a fresh RUN_* directory is created.",
    )
    parser.add_argument(
        "--delete-old-only-on-success",
        action="store_true",
        help="With --replace-old, delete the old run only if the rerun succeeds.",
    )
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop if one rerun exits non-zero.")
    parser.add_argument("--lira-cli-jar", help="Override lira_cli_jar for recovered configs.")
    args = parser.parse_args()

    config_path = _as_root(args.config)
    base_config = _load_json(config_path)
    roots = [_as_root(root) for root in (args.runs_root or ["Runs"])]
    candidates = _candidate_failed_runs(roots, failure_contains=args.failure_contains)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        candidates = candidates[: args.limit]

    print("[RERUN_FAILED] roots:", ", ".join(_safe_rel(root) for root in roots))
    print("[RERUN_FAILED] failed_runs_found:", len(candidates))

    if args.dry_run:
        for run_dir, metadata in candidates:
            print(
                "[RERUN_FAILED] dry-run:",
                _safe_rel(run_dir),
                "|",
                _failure_text(metadata)[:300],
            )
        return 0

    failures = 0
    rerun_count = 0
    for index, (old_run_dir, old_metadata) in enumerate(candidates, start=1):
        print(f"\n[RERUN_FAILED] {index}/{len(candidates)} old_run: {_safe_rel(old_run_dir)}")
        cycle_metadata = _first_cycle_metadata(old_metadata)
        try:
            config = _build_rerun_config(
                base_config=base_config,
                old_run_dir=old_run_dir,
                old_metadata=old_metadata,
                cycle_metadata=cycle_metadata,
                lira_cli_jar_override=args.lira_cli_jar,
            )
        except Exception as exc:
            failures += 1
            print(f"[RERUN_FAILED] skip: could not recover config: {exc}")
            if args.stop_on_failure:
                return 1
            continue

        output_root = _default_runs_root(config)
        before = _run_dirs(output_root)
        exit_code = _run_pipeline(config)
        after = _run_dirs(output_root)
        new_runs = sorted(after - before, key=lambda path: path.stat().st_mtime)
        if not new_runs:
            failures += 1
            print("[RERUN_FAILED] ERROR: no fresh RUN_* directory detected.")
            if args.stop_on_failure:
                return exit_code or 1
            continue

        rerun_count += 1
        new_run_dir = new_runs[-1]
        print("[RERUN_FAILED] new_run:", _safe_rel(new_run_dir))
        if exit_code != 0:
            failures += 1
            print(f"[RERUN_FAILED] rerun exit_code={exit_code}")

        should_delete = args.replace_old and (
            not args.delete_old_only_on_success or exit_code == 0
        )
        if should_delete:
            if old_run_dir == new_run_dir or old_run_dir not in before:
                failures += 1
                print("[RERUN_FAILED] refusing to delete old run: directory identity changed.")
            else:
                shutil.rmtree(old_run_dir)
                print("[RERUN_FAILED] deleted_old_run:", _safe_rel(old_run_dir))

        if exit_code != 0 and args.stop_on_failure:
            return exit_code

    print(f"\n[RERUN_FAILED] completed reruns={rerun_count} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
