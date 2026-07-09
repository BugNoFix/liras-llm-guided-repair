#!/usr/bin/env python3
"""Run one pipeline execution for each NL specification with the right LIRAS CLI."""

import argparse
import copy
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline_runner import PROJECT_ROOT, _load_json, _run_pipeline


DEFAULT_RUNS = [
    ("NL_Specification_1.txt", "liras-cli.jar"),
    ("NL_Specification_3.txt", "liras-cli-layout2.jar"),
]

AVAILABLE_RUNS = [
    ("NL_Specification_1.txt", "liras-cli.jar"),
    ("NL_Specification_2.txt", "liras-cli.jar"),
    ("NL_Specification_3.txt", "liras-cli-layout2.jar"),
]

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-35B-A3B",
    "google/gemma-4-31B-it",
    "openai/gpt-oss-20b",
    "zai-org/GLM-5.2",
]


def _select_runs(raw_specs: Optional[List[str]]) -> List[Tuple[str, str]]:
    if not raw_specs:
        return DEFAULT_RUNS

    selected: List[Tuple[str, str]] = []
    available = {scenario: (scenario, jar) for scenario, jar in AVAILABLE_RUNS}
    available.update({
        scenario.replace("NL_Specification_", "").replace(".txt", ""): (scenario, jar)
        for scenario, jar in AVAILABLE_RUNS
    })

    for raw in raw_specs:
        key = str(raw).strip()
        if key not in available:
            valid = ", ".join(sorted(available))
            raise ValueError(f"Unknown spec '{raw}'. Valid values: {valid}")
        item = available[key]
        if item not in selected:
            selected.append(item)

    return selected


def _model_results_label(model: str) -> str:
    label = model.strip().split("/")[-1]
    return label.replace(":", "-").replace(" ", "_")


def _select_models(base_config: dict, raw_models: Optional[List[str]], all_models: bool) -> List[str]:
    if all_models:
        selected = list(DEFAULT_MODELS)
    elif raw_models:
        selected = [str(model).strip() for model in raw_models if str(model).strip()]
    else:
        selected = [str(base_config.get("generation_model") or "").strip()]

    deduped: List[str] = []
    for model in selected:
        if not model:
            continue
        if model not in deduped:
            deduped.append(model)
    if not deduped:
        raise ValueError("No model selected. Set generation_model in config.json or pass --model/--all-models.")
    return deduped


def _resolve_config_path(raw_path: str) -> Path:
    cfg_path = Path(raw_path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    return cfg_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pipeline once for the enabled NL specifications using config.json parameters."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failed specification instead of running the remaining selected specs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of full rounds to run. Each round runs all enabled specs, then increments llm_seed by 1.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        help="Run only this spec. Use 1, 2, 3, or the full scenario filename. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Run this generation/repair model. Can be repeated. Defaults to generation_model from config.json.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all configured default generation/repair models.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    try:
        selected_runs = _select_runs(args.spec)
    except ValueError as exc:
        parser.error(str(exc))

    base_config = _load_json(_resolve_config_path(args.config))
    try:
        selected_models = _select_models(base_config, args.model, args.all_models)
    except ValueError as exc:
        parser.error(str(exc))
    base_seed = base_config.get("llm_seed")
    if base_seed is not None:
        base_seed = int(base_seed)
    exit_codes: list[int] = []

    for run_index in range(args.runs):
        run_seed = base_seed + run_index if base_seed is not None else None
        print("#" * 80)
        print(f"[NLSPEC_PIPELINE] ROUND {run_index + 1}/{args.runs} llm_seed={run_seed}")
        print("#" * 80)

        for model_index, model in enumerate(selected_models, start=1):
            for spec_index, (scenario, lira_cli_jar) in enumerate(selected_runs, start=1):
                config = copy.deepcopy(base_config)
                config["scenario"] = scenario
                config["lira_cli_jar"] = lira_cli_jar
                config["llm_seed"] = run_seed
                config["generation_model"] = model
                config["repair_model"] = model
                config["results_dir"] = str(Path("Runs") / _model_results_label(model))

                print("=" * 80)
                print(
                    f"[NLSPEC_PIPELINE] round={run_index + 1}/{args.runs} "
                    f"model={model_index}/{len(selected_models)} {model} "
                    f"spec={spec_index}/{len(selected_runs)} scenario={scenario} "
                    f"lira_cli_jar={lira_cli_jar} llm_seed={run_seed}"
                )
                print("=" * 80)

                exit_code = _run_pipeline(config)
                exit_codes.append(exit_code)
                print(
                    f"[NLSPEC_PIPELINE] DONE round={run_index + 1}/{args.runs} "
                    f"model={model} scenario={scenario} llm_seed={run_seed} exit_code={exit_code}"
                )

                if exit_code != 0 and args.stop_on_failure:
                    return exit_code

    return 1 if any(code != 0 for code in exit_codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
