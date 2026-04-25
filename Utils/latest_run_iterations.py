#!/usr/bin/env python3

"""Print folder name + iteration count for every run.

Scans run_metadata.json files under a runs root and prints one row per run with:
- configuration folder name (first folder under Runs)
- scenario
- iteration count
- metadata path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scenario_name_from_path(meta_path: Path) -> str:
    try:
        return meta_path.parents[3].name
    except IndexError:
        return "unknown"


def _config_folder_from_path(meta_path: Path, runs_root: Path) -> str:
    try:
        rel = meta_path.relative_to(runs_root)
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return "unknown"


def _all_runs(runs_root: Path) -> list[tuple[str, str, int, Path]]:
    rows: list[tuple[str, str, int, Path]] = []
    for meta_path in runs_root.glob("**/run_metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        folder = _config_folder_from_path(meta_path, runs_root)
        scenario = str(meta.get("scenario") or _scenario_name_from_path(meta_path))
        iterations = len(meta.get("iterations", []) or [])
        rows.append((folder, scenario, iterations, meta_path))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Print folder name + iteration count for every run.")
    parser.add_argument(
        "--runs-root",
        default="Runs",
        help="Root directory to scan (default: Runs)",
    )
    parser.add_argument(
        "--sort-by",
        choices=("folder", "scenario", "iter"),
        default="folder",
        help="Sort output by folder, scenario, or iteration count",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    rows = _all_runs(runs_root)
    if args.sort_by == "folder":
        rows.sort(key=lambda r: (r[0], r[1], r[3].as_posix()))
    elif args.sort_by == "scenario":
        rows.sort(key=lambda r: (r[1], r[0], r[3].as_posix()))
    else:
        rows.sort(key=lambda r: (-r[2], r[0], r[1]))

    if not rows:
        print(f"No run_metadata.json files found under {runs_root}")
        return 0

    folder_w = max(len("Folder"), max(len(row[0]) for row in rows))
    scenario_w = max(len("Scenario"), max(len(row[1]) for row in rows))
    iter_w = max(len("Iterazioni"), max(len(str(row[2])) for row in rows))
    total_iterations = sum(row[2] for row in rows)

    print(f"Runs in {runs_root}")
    print(f"Runs found: {len(rows)}  |  Total iterations: {total_iterations}")
    print()
    print(f"{'Folder':<{folder_w}}  {'Scenario':<{scenario_w}}  {'Iterazioni':>{iter_w}}  Path")
    separator = f"{'-' * folder_w}  {'-' * scenario_w}  {'-' * iter_w}  {'-' * 40}"
    print(separator)
    previous_folder = None
    for folder, scenario, iterations, meta_path in rows:
        if previous_folder is not None and folder != previous_folder:
            print(separator)
        print(f"{folder:<{folder_w}}  {scenario:<{scenario_w}}  {iterations:>{iter_w}}  {meta_path}")
        previous_folder = folder

    return 0


if __name__ == "__main__":
    raise SystemExit(main())