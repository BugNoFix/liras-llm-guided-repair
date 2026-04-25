#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
            candidate = PROJECT_ROOT / candidate
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

    try:
        import subprocess

        completed = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False,
        )
        candidate = (completed.stdout or "").strip()
        if candidate and candidate != "(unset)":
            return candidate
    except Exception:
        pass

    return None


def _resolve_provider_api_key(provider: str, config: dict) -> Optional[str]:
    provider = (provider or "gemini").strip().lower()

    if provider == "groq":
        cfg_key = config.get("groq_api_key")
        if isinstance(cfg_key, str) and cfg_key.strip():
            return cfg_key.strip()
        env_val = os.environ.get("GROQ_API_KEY")
        return env_val.strip() if isinstance(env_val, str) and env_val.strip() else None

    if provider == "mistral":
        cfg_key = config.get("mistral_api_key")
        if isinstance(cfg_key, str) and cfg_key.strip():
            return cfg_key.strip()
        env_val = os.environ.get("MISTRAL_API_KEY")
        return env_val.strip() if isinstance(env_val, str) and env_val.strip() else None

    if provider == "openrouter":
        cfg_key = config.get("openrouter_api_key")
        if isinstance(cfg_key, str) and cfg_key.strip():
            return cfg_key.strip()
        env_val = os.environ.get("OPENROUTER_API_KEY")
        return env_val.strip() if isinstance(env_val, str) and env_val.strip() else None

    if provider == "huggingface":
        cfg_key = config.get("huggingface_api_key")
        if isinstance(cfg_key, str) and cfg_key.strip():
            return cfg_key.strip()
        env_val = os.environ.get("HUGGINGFACE_API_KEY")
        if isinstance(env_val, str) and env_val.strip():
            return env_val.strip()
        env_val = os.environ.get("HF_TOKEN")
        return env_val.strip() if isinstance(env_val, str) and env_val.strip() else None

    return None


def _validate_template_config(config: dict) -> None:
    required_keys = ["generation_model", "shots", "system_prompt", "scenario"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"config.json missing required keys: {missing}")

    if not isinstance(config.get("generation_model"), str) or not config["generation_model"].strip():
        raise ValueError("'generation_model' must be a non-empty string")
    if not isinstance(config.get("system_prompt"), str) or not config["system_prompt"].strip():
        raise ValueError("'system_prompt' must be a non-empty string")
    if not isinstance(config.get("scenario"), str) or not config["scenario"].strip():
        raise ValueError("'scenario' must be a non-empty string")
    if not isinstance(config.get("shots"), (int, list)):
        raise ValueError("'shots' must be an integer or a list")

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


def _with_shot_suffix(results_dir: Optional[str], shots_override: Optional[list[int]]) -> Optional[str]:
    """Append a shot suffix to results_dir when shot override is explicitly provided."""
    if not results_dir or not shots_override:
        return results_dir
    # Keep folder names compact and deterministic, e.g. Runs/X_Shot1 or Runs/X_Shot1-2.
    suffix = "-".join(str(s) for s in shots_override)
    return f"{results_dir}_Shot{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DSL pipeline for every scenario using the system prompt, models, and "
            "other settings already defined in config.json."
        )
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to template config.json (default: config.json)",
    )
    parser.add_argument(
        "--shots",
        default="",
        help="Optional comma-separated shot counts to override config.json shots (e.g. 0,1,2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of scenarios to run (0 = no limit)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print planned runs and exit without executing",
    )
    parser.add_argument(
        "--scenario-glob",
        default="*.txt",
        help="Glob used to discover scenarios under Scenarios/ (default: *.txt)",
    )
    parser.add_argument(
        "--inter-run-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between runs (default: 1.0)",
    )
    parser.add_argument(
        "--flash",
        action="store_true",
        default=False,
        help="Use the flash-optimized generator (dsl_generator_flash.py)",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.inter_run_delay < 0:
        parser.error("--inter-run-delay must be >= 0")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    template = _load_json(cfg_path)
    _validate_template_config(template)

    provider = str(template.get("provider", "gemini")).strip().lower()
    if provider not in ("gemini", "groq", "mistral", "openrouter", "huggingface"):
        raise ValueError("Unsupported provider in config. Use gemini, groq, mistral, openrouter, or huggingface")

    if args.flash and provider != "gemini":
        raise ValueError("--flash is supported only with provider=gemini")

    shots_override = _parse_shots_arg(args.shots)

    scenarios_dir = PROJECT_ROOT / "Scenarios"
    scenarios = [p.name for p in sorted(scenarios_dir.glob(args.scenario_glob))]
    if not scenarios:
        raise RuntimeError(f"No scenarios found under: {scenarios_dir}")

    if args.limit and args.limit > 0:
        scenarios = scenarios[: args.limit]

    if args.list_only:
        for i, scenario in enumerate(scenarios, start=1):
            print(f"[{i}/{len(scenarios)}] scenario={scenario} system_prompt={template.get('system_prompt')}")
        return 0

    provider_api_key = _resolve_provider_api_key(provider, template)
    service_account_key = None
    project_id = None
    if provider == "gemini":
        service_account_key = _resolve_service_account_key(template)
        project_id = _resolve_project_id(template, service_account_key)
        if not project_id:
            raise RuntimeError(
                "Missing authentication for provider=gemini. "
                "Provide Vertex project credentials. If you only ran 'gcloud auth login', also run "
                "'gcloud auth application-default login' and set a project with 'gcloud config set project ...'."
            )
    else:
        if not provider_api_key:
            raise RuntimeError(
                f"Missing API key for provider={provider}. "
                f"Set the corresponding key in config.json or export the provider env variable."
            )

    location = template.get("location", "global")
    generation_temperature = float(template.get("generation_temperature", 1.0))
    repair_temperature = float(template.get("repair_temperature", 0.2))
    shots_value = shots_override if shots_override is not None else template.get("shots")

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(scenarios)

    print(f"\n{'='*70}")
    print(f"[BATCH] Starting {total} run(s)")
    print(f"  Batch ID:        {batch_id}")
    print(f"  Provider:        {provider}")
    print(f"  Auth mode:       {'Vertex AI' if provider == 'gemini' else 'API key'}")
    print(f"  System prompt:   {template.get('system_prompt')}")
    print(f"  Generation:      {'flash' if args.flash else 'standard'}")
    print(f"  Shots:           {shots_value}")
    print(f"  Inter-run delay: {args.inter_run_delay}s")
    print(f"{'='*70}\n")

    if args.flash:
        from dsl_generator_flash import DSLGenerator
        print("[BATCH] Using flash-optimized generator")
    else:
        from dsl_generator import DSLGenerator

    for idx, scenario in enumerate(scenarios, start=1):
        print(f"[{idx}/{total}] START batch={batch_id} scenario={scenario}")

        generator = DSLGenerator(
            project_id=project_id,
            location=location,
            service_account_key=service_account_key,
            generation_temperature=generation_temperature,
            repair_temperature=repair_temperature,
            repair_max_output_tokens=int(template.get("repair_max_output_tokens", 16384)),
            provider=provider,
            api_key=provider_api_key,
        )

        run_config = deepcopy(template)
        run_config["scenario"] = scenario
        if shots_override is not None:
            run_config["shots"] = shots_override if len(shots_override) != 1 else shots_override[0]
        else:
            run_config["shots"] = shots_value
        run_config["results_dir"] = _with_shot_suffix(
            str(run_config.get("results_dir")) if run_config.get("results_dir") else None,
            shots_override,
        )

        if run_config.get("results_dir"):
            print(f"[{idx}/{total}] results_dir={run_config['results_dir']}")

        try:
            generator.run_automated_session(run_config)
        except Exception as e:
            print(f"[{idx}/{total}] ERROR scenario={scenario}: {type(e).__name__}: {e}")
            continue

        print(f"[{idx}/{total}] DONE  batch={batch_id} scenario={scenario}")

        if idx < total and args.inter_run_delay > 0:
            print(f"[BATCH] Waiting {args.inter_run_delay}s before next run...\n")
            time.sleep(args.inter_run_delay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())