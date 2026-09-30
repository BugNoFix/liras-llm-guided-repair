#!/usr/bin/env python3
"""Run NL_Specification_1/2/3 experiments for one or more models.

The script uses config.json as the base configuration, then overrides only the
scenario/model/result directory needed for each run. Generation and repair are
always forced to use the same model.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_runner import _run_pipeline  # noqa: E402


DEFAULT_SCENARIOS = (
    "NL_Specification_1.txt",
    "NL_Specification_2.txt",
    "NL_Specification_3.txt",
)

DEFAULT_LIRA_CLI_JAR = "liras-cli.jar"
LAYOUT2_LIRA_CLI_JAR = "liras-cli-layout2.jar"
DEFAULT_MODELS = (
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-35B-A3B",
    "google/gemma-4-31B-it",
    "openai/gpt-oss-20b",
    "zai-org/GLM-5.2",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_result_name(model: str) -> str:
    name = model.strip().rstrip("/")
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    return name.strip("-") or "model"


def _as_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _models_from_config(config: dict[str, Any]) -> list[str]:
    for key in ("models", "experiment_models", "generation_models"):
        models = _as_string_list(config.get(key))
        if models:
            return models

    generation_model = str(config.get("generation_model") or "").strip()
    repair_model = str(config.get("repair_model") or "").strip()
    if generation_model and repair_model and generation_model != repair_model:
        print(
            "[BATCH] WARNING: config.json has different generation_model and "
            "repair_model; using generation_model for both."
        )
    return list(DEFAULT_MODELS)


def _scenarios_from_args(raw_scenarios: list[str] | None) -> list[str]:
    if not raw_scenarios:
        return list(DEFAULT_SCENARIOS)

    scenarios: list[str] = []
    for raw in raw_scenarios:
        value = raw.strip()
        if not value:
            continue
        if value.isdigit():
            value = f"NL_Specification_{value}.txt"
        elif not value.endswith(".txt"):
            value = f"{value}.txt"
        scenarios.append(value)
    return scenarios


def _scenario_path(scenario: str) -> Path:
    path = Path(scenario).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / "Scenarios" / path.name


def _lira_cli_jar_for_scenario(scenario: str) -> str:
    if Path(scenario).name == "NL_Specification_3.txt":
        return LAYOUT2_LIRA_CLI_JAR
    return DEFAULT_LIRA_CLI_JAR


def _build_run_config(
    base_config: dict[str, Any],
    *,
    model: str,
    scenario: str,
    results_root: str,
    block_hf_thinking: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["scenario"] = scenario
    config["generation_model"] = model
    config["repair_model"] = model
    config["results_dir"] = str(Path(results_root) / _safe_result_name(model))
    config["lira_cli_jar"] = _lira_cli_jar_for_scenario(scenario)
    config["block_hf_thinking"] = block_hf_thinking
    return config


def _pipeline_runs_root(config: dict[str, Any]) -> Path:
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


def _planned_missing_runs(
    base_config: dict[str, Any],
    *,
    models: list[str],
    scenarios: list[str],
    target_runs_per_pair: int,
    results_root: str,
    block_hf_thinking: bool,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for model in models:
        for scenario in scenarios:
            run_config = _build_run_config(
                base_config,
                model=model,
                scenario=scenario,
                results_root=results_root,
                block_hf_thinking=block_hf_thinking,
            )
            output_root = _pipeline_runs_root(run_config)
            existing_runs = len(_run_dirs(output_root))
            missing_runs = max(0, target_runs_per_pair - existing_runs)
            plan.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "output_root": output_root,
                    "existing_runs": existing_runs,
                    "missing_runs": missing_runs,
                }
            )
    return plan


def _print_plan(
    models: list[str],
    scenarios: list[str],
    target_runs_per_pair: int,
    results_root: str,
    plan: list[dict[str, Any]],
) -> None:
    print("[BATCH] scenarios:", ", ".join(scenarios))
    print("[BATCH] models:", ", ".join(models))
    print("[BATCH] target_runs_per_model_per_scenario:", target_runs_per_pair)
    print("[BATCH] results_root:", results_root)
    print("[BATCH] total_missing_pipeline_runs:", sum(item["missing_runs"] for item in plan))
    for item in plan:
        existing_runs = int(item["existing_runs"])
        missing_runs = int(item["missing_runs"])
        status = "complete" if missing_runs == 0 else f"missing={missing_runs}"
        if existing_runs > target_runs_per_pair:
            status = "over_target"
        print(
            "[BATCH] target_status "
            f"model={item['model']} scenario={item['scenario']} "
            f"existing={existing_runs} target={target_runs_per_pair} {status} "
            f"root={item['output_root']}"
        )


def _base_seed_from_config(config: dict[str, Any]) -> int | None:
    raw_seed = config.get("llm_seed")
    if raw_seed is None:
        return None
    try:
        return int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("llm_seed in config.json must be an integer to auto-increment it.") from exc


def _short_debug_value(value: Any, *, limit: int = 240) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _is_gpt_oss_model(model: str) -> bool:
    normalized = (model or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    return "gpt-oss" in normalized or "gptoss" in compact


def _find_hf_thinking_signal(response_text: str, response_payload: Any) -> str | None:
    if re.search(r"</?\s*think(?:ing)?\b", response_text or "", flags=re.IGNORECASE):
        return "response_text contains <think> markup"

    thinking_keys = {
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "reasoning_trace",
        "thinking",
        "thinking_content",
        "thoughts",
    }
    token_keys = {
        "reasoning_tokens",
        "thinking_tokens",
    }

    def is_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, bool):
            return value
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return True

    def walk(value: Any, path: str = "response_obj") -> str | None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                key_norm = key_text.lower()
                item_path = f"{path}.{key_text}"
                if key_norm in token_keys and is_present(item):
                    return f"{item_path}={_short_debug_value(item)}"
                if key_norm in thinking_keys and is_present(item):
                    return f"{item_path}={_short_debug_value(item)}"
                found = walk(item, item_path)
                if found:
                    return found
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found = walk(item, f"{path}[{index}]")
                if found:
                    return found
        return None

    return walk(response_payload)


def _scan_run_for_hf_thinking(run_dir: Path) -> tuple[Path, int, str] | None:
    for log_path in sorted(run_dir.rglob("hf_debug_responses.jsonl")):
        with open(log_path, "r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                if not raw_line.strip():
                    continue
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if entry.get("kind") not in ("generate", "repair"):
                    continue
                if _is_gpt_oss_model(str(entry.get("model") or "")):
                    continue
                signal = _find_hf_thinking_signal(
                    str(entry.get("response_text") or ""),
                    entry.get("response_obj"),
                )
                if signal:
                    return log_path, line_number, signal
    return None


def _is_gateway_timeout_text(text: str) -> bool:
    normalized = (text or "").lower()
    return (
        "gateway timeout" in normalized
        or "504 gateway" in normalized
        or "http 504" in normalized
        or "status_code=504" in normalized
        or "status code: 504" in normalized
        or "error code: 504" in normalized
    )


def _discard_run(run_dir: Path, *, reason: str) -> None:
    print(f"[BATCH] DISCARDED: {reason} run_dir={run_dir}")
    shutil.rmtree(run_dir)
    print(f"[BATCH] deleted_run: {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LIRAS pipeline for NL_Specification_1/2/3 across one or "
            "more models, using config.json as the base configuration."
        )
    )
    parser.add_argument(
        "target_runs",
        nargs="?",
        type=int,
        help="Alias for --runs-per-model, e.g. pass 10 to reach 10 runs per model/scenario pair.",
    )
    parser.add_argument("--config", default="config.json", help="Base config file (default: config.json).")
    parser.add_argument(
        "-n",
        "--runs-per-model",
        type=int,
        default=1,
        help=(
            "Target number of RUN_* directories for each model/scenario pair. "
            "Only missing runs are launched (default: 1)."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_list",
        help="Model to test. Can be passed more than once. Forces generation and repair to this model.",
    )
    parser.add_argument(
        "--models",
        dest="models_csv",
        help="Comma-separated list of models to test. Ignored if --model is used.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario to run, e.g. 1, 2, 3, or NL_Specification_1. Defaults to all three.",
    )
    parser.add_argument(
        "--results-root",
        default="Runs",
        help="Root directory where per-model results folders are created (default: Runs).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without launching runs.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining runs after a failed pipeline run.",
    )
    parser.add_argument(
        "--block-hf-thinking",
        action="store_true",
        help=(
            "Discard the current run if generation/repair exposes Hugging Face "
            "reasoning/thinking fields, then continue with the next run."
        ),
    )
    args = parser.parse_args()

    if args.target_runs is not None:
        args.runs_per_model = args.target_runs
    if args.runs_per_model < 1:
        parser.error("target runs / --runs-per-model must be >= 1")

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    base_config = _read_json(config_path)
    if args.model_list:
        models = _as_string_list(args.model_list)
    elif args.models_csv:
        models = _as_string_list(args.models_csv)
    else:
        models = _models_from_config(base_config)

    scenarios = _scenarios_from_args(args.scenarios)
    missing = [scenario for scenario in scenarios if not _scenario_path(scenario).exists()]
    if missing:
        raise FileNotFoundError(f"scenario file(s) not found under Scenarios/: {', '.join(missing)}")
    if not models:
        raise ValueError(
            "No models found. Add generation_model to config.json, add a models list, "
            "or pass --model."
        )

    base_seed = _base_seed_from_config(base_config)
    plan = _planned_missing_runs(
        base_config,
        models=models,
        scenarios=scenarios,
        target_runs_per_pair=args.runs_per_model,
        results_root=args.results_root,
        block_hf_thinking=args.block_hf_thinking,
    )
    _print_plan(models, scenarios, args.runs_per_model, args.results_root, plan)
    print("[BATCH] block_hf_thinking:", args.block_hf_thinking)
    if base_seed is not None:
        print("[BATCH] seed_range_per_model_scenario:", f"{base_seed}..{base_seed + args.runs_per_model - 1}")
        print("[BATCH] new_run_seed_policy: seed = base_seed + current_existing_run_count")
    if args.dry_run:
        print("[BATCH] dry-run: no pipeline run launched.")
        return 0

    failures: list[tuple[str, str, int, int]] = []
    total = sum(item["missing_runs"] for item in plan)
    current = 0
    for model in models:
        thinking_streak = 0
        skip_model = False
        for scenario in scenarios:
            if skip_model:
                break
            run_config = _build_run_config(
                base_config,
                model=model,
                scenario=scenario,
                results_root=args.results_root,
                block_hf_thinking=args.block_hf_thinking,
            )
            output_root = _pipeline_runs_root(run_config)
            while len(_run_dirs(output_root)) < args.runs_per_model:
                existing_before_count = len(_run_dirs(output_root))
                run_index = existing_before_count + 1
                current += 1
                print(
                    f"\n[BATCH] {current}/{total} model={model} "
                    f"scenario={scenario} run={run_index}/{args.runs_per_model} "
                    f"lira_cli_jar={_lira_cli_jar_for_scenario(scenario)} "
                    f"llm_seed={base_seed + run_index - 1 if base_seed is not None else 'unset'}"
                )
                if base_seed is not None:
                    run_config["llm_seed"] = base_seed + run_index - 1
                before_runs = _run_dirs(output_root)
                pipeline_exception = None
                try:
                    exit_code = _run_pipeline(run_config)
                except Exception as exc:
                    pipeline_exception = exc
                    exit_code = 1
                    print(f"[BATCH] pipeline exception: {type(exc).__name__}: {exc}")

                new_runs = sorted(_run_dirs(output_root) - before_runs, key=lambda path: path.stat().st_mtime)
                discarded_run = False
                discarded_for_thinking = False
                for new_run in new_runs:
                    if args.block_hf_thinking:
                        thinking_hit = _scan_run_for_hf_thinking(new_run)
                        if thinking_hit:
                            log_path, line_number, signal = thinking_hit
                            _discard_run(
                                new_run,
                                reason=(
                                    "Hugging Face thinking detected "
                                    f"log={log_path} line={line_number} signal={signal}"
                                ),
                            )
                            discarded_run = True
                            discarded_for_thinking = True
                if discarded_run:
                    if discarded_for_thinking:
                        thinking_streak += 1
                        print(f"[BATCH] thinking_streak model={model}: {thinking_streak}/2")
                        if thinking_streak >= 2:
                            print(
                                "[BATCH] SKIP_MODEL: two consecutive generation/repair "
                                f"thinking runs for model={model}; moving to next model."
                            )
                            skip_model = True
                            break
                    continue
                thinking_streak = 0
                if pipeline_exception is not None:
                    if _is_gateway_timeout_text(str(pipeline_exception)):
                        print("[BATCH] Gateway Timeout detected; run kept for rerun_failed_runs.py.")
                        failures.append((model, scenario, run_index, exit_code))
                        if not new_runs and args.keep_going:
                            print("[BATCH] no RUN_* directory was created; moving to next scenario to avoid retrying forever.")
                            break
                        if not args.keep_going:
                            return exit_code
                        continue
                    raise pipeline_exception
                if exit_code != 0:
                    failures.append((model, scenario, run_index, exit_code))
                    print(f"[BATCH] FAILED exit_code={exit_code}")
                    if not new_runs and args.keep_going:
                        print("[BATCH] no RUN_* directory was created; moving to next scenario to avoid retrying forever.")
                        break
                    if not args.keep_going:
                        print("[BATCH] stopping after first failure; use --keep-going to continue.")
                        return exit_code
            existing_after_count = len(_run_dirs(output_root))
            if existing_after_count == args.runs_per_model:
                print(
                    f"[BATCH] target reached model={model} scenario={scenario} "
                    f"runs={existing_after_count}/{args.runs_per_model}"
                )
            elif existing_after_count > args.runs_per_model:
                print(
                    f"[BATCH] target already exceeded model={model} scenario={scenario} "
                    f"runs={existing_after_count}/{args.runs_per_model}; no existing runs deleted."
                )

    if failures:
        print("\n[BATCH] completed with failures:")
        for model, scenario, run_index, exit_code in failures:
            print(f"  - model={model} scenario={scenario} run={run_index} exit_code={exit_code}")
        return 1

    print("\n[BATCH] all runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
