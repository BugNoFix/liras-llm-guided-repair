#!/usr/bin/env python3

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent


def _default_key_path() -> Path:
    key_path = PROJECT_ROOT / "keys" / "key.json"
    if key_path.exists():
        return key_path
    return PROJECT_ROOT / "key.json"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_service_account_key(config: dict) -> Optional[str]:
    cfg_key_path = (
        config.get("service_account_key_path")
        or config.get("service_account_key")
        or config.get("credentials_path")
    )
    if isinstance(cfg_key_path, str) and cfg_key_path.strip():
        candidate = Path(cfg_key_path).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate)
        if not candidate.exists():
            raise FileNotFoundError(f"service account key file not found: {candidate}")
        return str(candidate)

    key_path = _default_key_path()
    if key_path.exists():
        return str(key_path)

    return None


def _resolve_project_id(config: dict, service_account_key: Optional[str]) -> Optional[str]:
    cfg_project_id = config.get("project_id")
    if isinstance(cfg_project_id, str) and cfg_project_id.strip():
        return cfg_project_id.strip()

    if service_account_key:
        try:
            key_data = _load_json(Path(service_account_key))
            pid = key_data.get("project_id")
            if isinstance(pid, str) and pid.strip():
                return pid.strip()
        except Exception:
            pass

    for env_key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GOOGLE_PROJECT_ID"):
        env_val = os.environ.get(env_key)
        if env_val and env_val.strip():
            return env_val.strip()

    return None


def _default_system_prompts(sp_dir: Path) -> list[str]:
    # Only the generation-stage system prompts (exclude repair prompts).
    prompts = []
    for p in sorted(sp_dir.glob("SystemPrompt*.txt")):
        name = p.name
        if "Repair" in name:
            continue
        prompts.append(name)
    for p in sorted(sp_dir.glob("SP*.txt")):
        name = p.name
        if name in prompts:
            continue
        prompts.append(name)
    return prompts


def _default_scenarios(scenarios_dir: Path) -> list[str]:
    scenarios = [p.name for p in sorted(scenarios_dir.glob("UserScenario_*.txt"))]
    if scenarios:
        return scenarios
    return [p.name for p in sorted(scenarios_dir.glob("Scenario_*.txt"))]


def _validate_template_config(config: dict, *, generation_only: bool) -> None:
    required_keys = [
        "generation_model",
        "shots",
        "system_prompt",
        "scenario",
    ]
    if not generation_only:
        required_keys.extend(["compiler_jar", "repair_model", "max_iterations"])
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"config.json missing required keys: {missing}")

    if not generation_only:
        if not isinstance(config.get("max_iterations"), int):
            raise ValueError("'max_iterations' must be an integer")

    if not isinstance(config.get("generation_model"), str) or not config["generation_model"].strip():
        raise ValueError("'generation_model' must be a non-empty string")
    if not generation_only:
        if not isinstance(config.get("repair_model"), str) or not config["repair_model"].strip():
            raise ValueError("'repair_model' must be a non-empty string")

    if not isinstance(config.get("shots"), (int, list)):
        raise ValueError("'shots' must be an integer or a list")

    if not generation_only:
        if "repair_shots" in config and config["repair_shots"] is not None:
            if not isinstance(config["repair_shots"], (int, list)):
                raise ValueError("'repair_shots' must be an integer or a list")

    for name in ("generation_temperature", "repair_temperature"):
        if name in config and config[name] is not None:
            if not isinstance(config[name], (int, float)):
                raise ValueError(f"'{name}' must be a number")
            if float(config[name]) < 0.0:
                raise ValueError(f"'{name}' must be >= 0.0")

    if "compiler_timeout" in config and config["compiler_timeout"] is not None:
        if not isinstance(config["compiler_timeout"], (int, float)):
            raise ValueError("'compiler_timeout' must be a number")
        if float(config["compiler_timeout"]) <= 0:
            raise ValueError("'compiler_timeout' must be > 0")


def _parse_shots_arg(value: str) -> Optional[list[int]]:
    """Parse comma-separated shot counts into a list of ints.

    Returns None when the argument is empty/whitespace, meaning "use template shots".
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    shots = []
    for part in parts:
        try:
            shots.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid shot count '{part}'. Use comma-separated integers.") from exc
    return shots


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full DSL pipeline (generate → compile → repair) for every "
            "UserScenario/SystemPrompt pair. Uses config.json as a template, overriding "
            "system_prompt + scenario for each run."
        )
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to template config.json (default: config.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of pairs to run (0 = no limit)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print planned runs and exit without executing",
    )
    parser.add_argument(
        "--shots",
        default="",
        help=(
            "Comma-separated shot counts to run (e.g., 0,1,2). "
            "Leave empty to use shots from config.json."
        ),
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Only generate DSL (skip compile/repair) for each pair",
    )
    parser.add_argument(
        "--disable-generation",
        action="store_true",
        help="Skip generation and load DSL from cache for each pair",
    )
    parser.add_argument(
        "--compiler-timeout",
        type=int,
        default=0,
        help="Compiler timeout in seconds (0 = use config.json default of 60)",
    )
    parser.add_argument(
        "--inter-run-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between runs to allow system recovery (default: 1.0)",
    )

    args = parser.parse_args()

    if args.compiler_timeout < 0:
        parser.error("--compiler-timeout must be >= 0")
    if args.inter_run_delay < 0:
        parser.error("--inter-run-delay must be >= 0")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    template = _load_json(cfg_path)
    _validate_template_config(template, generation_only=args.generation_only)

    shots_override = _parse_shots_arg(args.shots)

    sp_dir = PROJECT_ROOT / "SPs"
    scenarios_dir = PROJECT_ROOT / "Scenarios"

    system_prompts = _default_system_prompts(sp_dir)
    scenarios = _default_scenarios(scenarios_dir)

    if not system_prompts:
        raise RuntimeError(f"No system prompts found under: {sp_dir}")
    if not scenarios:
        raise RuntimeError(f"No user scenarios found under: {scenarios_dir}")

    planned_pairs: list[tuple[str, str, Optional[int]]] = []
    shots_values = shots_override if shots_override is not None else [None]
    for scenario in scenarios:
        for sp in system_prompts:
            for shots in shots_values:
                planned_pairs.append((scenario, sp, shots))

    if args.limit and args.limit > 0:
        planned_pairs = planned_pairs[: args.limit]

    if args.list_only:
        for i, (scenario, sp, shots) in enumerate(planned_pairs, start=1):
            shots_label = shots if shots is not None else template.get("shots")
            print(
                f"[{i}/{len(planned_pairs)}] scenario={scenario}  system_prompt={sp}  shots={shots_label}"
            )
        return 0

    service_account_key = _resolve_service_account_key(template)
    project_id = _resolve_project_id(template, service_account_key)
    if not project_id:
        raise RuntimeError(
            "Google Cloud Project ID not found. Set 'project_id' in config.json, "
            "export GOOGLE_CLOUD_PROJECT, or include project_id in key.json."
        )

    location = template.get("location", "global")
    generation_temperature = float(template.get("generation_temperature", 1.0))
    repair_temperature = float(template.get("repair_temperature", 0.2))

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(planned_pairs)
    
    # Print batch summary
    print(f"\n{'='*70}")
    print(f"[BATCH] Starting {total} run(s)")
    print(f"  Batch ID:        {batch_id}")
    print(f"  Generation:      {'disabled (loading from cache)' if args.disable_generation else 'enabled'}")
    print(f"  Compiler timeout: {args.compiler_timeout if args.compiler_timeout > 0 else 'default (60s)'}")
    print(f"  Inter-run delay: {args.inter_run_delay}s")
    print(f"{'='*70}\n")

    from dsl_generator import DSLGenerator

    for idx, (scenario, sp, shots) in enumerate(planned_pairs, start=1):
        shots_label = shots if shots is not None else template.get("shots")
        print(
            f"[{idx}/{total}] START batch={batch_id}  scenario={scenario}  "
            f"system_prompt={sp}  shots={shots_label}"
        )

        # Fresh generator per run to avoid any cross-run chat/history leakage.
        generator = DSLGenerator(
            project_id,
            location=location,
            service_account_key=service_account_key,
            generation_temperature=generation_temperature,
            repair_temperature=repair_temperature,
        )

        run_config = deepcopy(template)
        run_config["scenario"] = scenario
        run_config["system_prompt"] = sp
        if shots is not None:
            run_config["shots"] = shots
        
        # Override compiler timeout if provided
        if args.compiler_timeout > 0:
            run_config["compiler_timeout"] = args.compiler_timeout
        
        if args.generation_only:
            run_config["generation_only"] = True
        if args.disable_generation:
            run_config["use_generated_dsl_cache"] = True
            run_config.setdefault("generated_dsl_source", "dsl_folder")

        try:
            generator.run_automated_session(run_config)
        except Exception as e:
            # Generator already tries to persist crash metadata; this is a last-resort guard.
            print(f"[{idx}/{total}] ERROR scenario={scenario} system_prompt={sp}: {type(e).__name__}: {e}")
            continue

        print(
            f"[{idx}/{total}] DONE  batch={batch_id}  scenario={scenario}  "
            f"system_prompt={sp}  shots={shots_label}"
        )
        
        # Add inter-run delay to allow system recovery (unless it's the last run)
        if idx < total and args.inter_run_delay > 0:
            print(f"[BATCH] Waiting {args.inter_run_delay}s before next run...\n")
            time.sleep(args.inter_run_delay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
