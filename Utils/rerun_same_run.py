#!/usr/bin/env python3
"""Rerun a previous pipeline run with the same recorded settings and seed.

The script accepts a run code (for example 20260706_100907 or
RUN_20260706_100907) or a path inside a run directory. It starts a fresh run in
the same results tree and removes the old run only after a new run directory is
created, unless --keep-old is passed.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_runner import _load_json, _run_pipeline  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _normalize_run_code(raw: str) -> str:
    code = raw.strip()
    if code.startswith("RUN_"):
        code = code[4:]
    return code


def _run_dir_from_path(path: Path) -> Path | None:
    current = path.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if candidate.name.startswith("RUN_") and (candidate / "run_metadata.json").exists():
            return candidate
    return None


def _find_run_dir(run_ref: str, runs_root: Path) -> Path:
    candidate = Path(run_ref).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    path_run_dir = _run_dir_from_path(candidate) if candidate.exists() else None
    if path_run_dir:
        return path_run_dir

    code = _normalize_run_code(run_ref)
    run_name = f"RUN_{code}"
    matches = [
        path
        for path in runs_root.rglob(run_name)
        if path.is_dir() and (path / "run_metadata.json").exists()
    ]
    if not matches:
        raise FileNotFoundError(f"Run not found under {_safe_rel(runs_root)}: {run_ref}")
    if len(matches) > 1:
        formatted = "\n".join(f"  - {_safe_rel(path)}" for path in matches)
        raise RuntimeError(f"Run code is ambiguous; pass a full path:\n{formatted}")
    return matches[0]


def _first_cycle_metadata(old_metadata: dict[str, Any]) -> dict[str, Any]:
    cycles = old_metadata.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return {}

    metadata_path = cycles[0].get("metadata_path")
    if not isinstance(metadata_path, str) or not metadata_path.strip():
        return {}

    path = Path(metadata_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    try:
        data = _read_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _relative_or_absolute(raw: Any, *, base: Path) -> Any:
    if not isinstance(raw, str) or not raw.strip():
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return raw
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _results_dir_from_old_run(old_run_dir: Path, cycle_metadata: dict[str, Any]) -> str:
    raw = cycle_metadata.get("results_dir")
    if isinstance(raw, str) and raw.strip():
        return str(_relative_or_absolute(raw, base=PROJECT_ROOT))

    # Expected layout: <results_dir>/<scenario>/<system_prompt>/RUN_<id>.
    try:
        return str(old_run_dir.parents[3].resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(old_run_dir.parents[3])


def _infer_lira_cli_jar(scenario: str, base_config: dict[str, Any], cycle_metadata: dict[str, Any]) -> str:
    raw = cycle_metadata.get("lira_cli_jar")
    if isinstance(raw, str) and raw.strip():
        return str(_relative_or_absolute(raw, base=PROJECT_ROOT))

    if base_config.get("scenario") == scenario and isinstance(base_config.get("lira_cli_jar"), str):
        return str(base_config["lira_cli_jar"])

    if scenario == "NL_Specification_3.txt":
        return "liras-cli-layout2.jar"
    return "liras-cli.jar"


def _copy_if_present(
    target: dict[str, Any],
    source: dict[str, Any],
    keys: list[str],
    *,
    relativize_paths: bool = False,
) -> None:
    for key in keys:
        if key not in source or source[key] is None:
            continue
        value = source[key]
        if relativize_paths:
            value = _relative_or_absolute(value, base=PROJECT_ROOT)
        target[key] = value


def _build_rerun_config(
    *,
    base_config: dict[str, Any],
    old_run_dir: Path,
    old_metadata: dict[str, Any],
    cycle_metadata: dict[str, Any],
    lira_cli_jar_override: str | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)

    _copy_if_present(
        config,
        old_metadata,
        [
            "generation_provider",
            "repair_provider",
            "query_provider",
            "system_prompt",
            "scenario",
            "generation_model",
            "repair_model",
            "shots",
            "repair_shots",
            "max_iterations",
            "max_uppaal_feedback_cycles",
        ],
    )
    _copy_if_present(
        config,
        cycle_metadata,
        [
            "project_id",
            "location",
            "generation_temperature",
            "generation_max_output_tokens",
            "llm_seed",
            "repair_temperature",
            "repair_max_output_tokens",
            "repair_stateless",
            "compiler_timeout",
            "use_generated_dsl_cache",
            "generated_dsl_source",
        ],
    )
    _copy_if_present(config, cycle_metadata, ["compiler_jar", "repair_prompt"], relativize_paths=True)

    scenario = str(config.get("scenario") or "")
    config["results_dir"] = _results_dir_from_old_run(old_run_dir, cycle_metadata)
    config["lira_cli_jar"] = lira_cli_jar_override or _infer_lira_cli_jar(
        scenario,
        base_config,
        cycle_metadata,
    )

    if "llm_seed" not in config or config["llm_seed"] is None:
        raise ValueError("Cannot recover llm_seed from the old run. Pass a run that has cycle metadata.")
    return config


def _default_runs_root(config: dict[str, Any]) -> Path:
    results_dir = Path(str(config["results_dir"])).expanduser()
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    scenario_name = str(config["scenario"]).replace(".txt", "")
    system_prompt = str(config["system_prompt"]).replace(".txt", "")
    return results_dir / scenario_name / system_prompt


def _run_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir() and path.name.startswith("RUN_")}


def _print_rerun_plan(old_run_dir: Path, config: dict[str, Any], root: Path) -> None:
    printable_keys = [
        "results_dir",
        "scenario",
        "system_prompt",
        "repair_prompt",
        "generation_model",
        "repair_model",
        "llm_seed",
        "shots",
        "repair_shots",
        "generation_temperature",
        "repair_temperature",
        "max_iterations",
        "max_uppaal_feedback_cycles",
        "lira_cli_jar",
        "compiler_jar",
        "query_model",
    ]
    print("[RERUN] old_run:", _safe_rel(old_run_dir))
    print("[RERUN] output_root:", _safe_rel(root))
    for key in printable_keys:
        if key in config:
            print(f"[RERUN] {key}: {config[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun an existing LIRAS pipeline run with the same recorded settings and seed."
    )
    parser.add_argument("run", help="Run code, RUN_<code>, or a path inside the old run.")
    parser.add_argument("--config", default="config.json", help="Base config used for settings not stored in metadata.")
    parser.add_argument("--runs-root", default="Runs", help="Root used when resolving a run code (default: Runs).")
    parser.add_argument("--lira-cli-jar", help="Override lira_cli_jar if the old run metadata did not store it.")
    parser.add_argument("--keep-old", action="store_true", help="Do not delete the old run after rerunning.")
    parser.add_argument("--delete-only-on-success", action="store_true", help="Delete the old run only if rerun exit code is 0.")
    parser.add_argument("--dry-run", action="store_true", help="Print recovered settings without launching or deleting anything.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    runs_root = Path(args.runs_root).expanduser()
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root

    base_config = _load_json(config_path)
    old_run_dir = _find_run_dir(args.run, runs_root).resolve()
    old_metadata_path = old_run_dir / "run_metadata.json"
    old_metadata = _read_json(old_metadata_path)
    cycle_metadata = _first_cycle_metadata(old_metadata)
    config = _build_rerun_config(
        base_config=base_config,
        old_run_dir=old_run_dir,
        old_metadata=old_metadata,
        cycle_metadata=cycle_metadata,
        lira_cli_jar_override=args.lira_cli_jar,
    )
    output_root = _default_runs_root(config)
    _print_rerun_plan(old_run_dir, config, output_root)

    if args.dry_run:
        print("[RERUN] dry-run: no run launched, old run kept.")
        return 0

    before = _run_dirs(output_root)
    exit_code = _run_pipeline(config)
    after = _run_dirs(output_root)
    new_runs = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not new_runs:
        print("[RERUN] ERROR: rerun finished but no new RUN_* directory was detected; old run kept.")
        return exit_code or 1

    new_run_dir = new_runs[-1]
    print("[RERUN] new_run:", _safe_rel(new_run_dir))
    if args.keep_old:
        print("[RERUN] --keep-old set: old run kept.")
        return exit_code
    if args.delete_only_on_success and exit_code != 0:
        print(f"[RERUN] rerun exit_code={exit_code}; old run kept because --delete-only-on-success is set.")
        return exit_code

    if old_run_dir == new_run_dir or old_run_dir not in before:
        print("[RERUN] Refusing to delete: old run directory identity changed unexpectedly.")
        return exit_code or 1

    shutil.rmtree(old_run_dir)
    print("[RERUN] deleted_old_run:", _safe_rel(old_run_dir))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
