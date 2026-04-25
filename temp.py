#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

# Prompt4-only batch requested
PROMPT4_PATH = "Generative/NewSp6.txt"
REPAIR_PROMPT4_PATH = "Repair/NewSPR6.txt"

# Provider/model matrix. Add/remove entries as needed.
COMBOS = [
            #("openrouter", "nvidia/nemotron-3-nano-30b-a3b:free", "Runs/nemotron-3-nano-30b-a3b"),
    #("mistral", "mistral-large-2512", "Runs/mistral_large_2512"),
            #("mistral", "mistral-medium-2508", "Runs/mistral_medium_2508"),
            #("mistral", "mistral-small-2603", "Runs/mistral_small_2603"),
            #("huggingface", "openai/gpt-oss-20b:groq", "Runs/groq_gpt_oss_20b"),
            #("huggingface", "openai/gpt-oss-120b:groq", "Runs/gpt_oss_120b"),
            # Non completo ma fatto scenari orilevamte("huggingface", "meta-llama/Llama-3.3-70B-Instruct:groq", "Runs/Llama-3.3-70B"),
            # non finito ma andava male ("huggingface", "meta-llama/Llama-3.1-8B-Instruct:novita", "Runs/llama-3.1-8b-Instruct"),
            #("huggingface", "deepseek-ai/DeepSeek-R1:novita", "Runs/DeepSeek-R1"),
            #("huggingface", "deepseek-ai/DeepSeek-V3:novita", "Runs/DeepSeek-V3"),
            # non finito ma andava male ("huggingface", "Qwen/Qwen3-8B-Base:featherless-ai", "Runs/Qwen3-8B-Base"),
            # non finito ma andava male ("huggingface", "Qwen/Qwen3-4B-Thinking-2507:nscale", "Runs/Qwen3-4B-Thinking-2507"),
    #("huggingface", "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai", "Runs/Mistral-7B-Instruct-v0.2"),
            #("gemini", "gemini-3.1-pro-preview", "Runs/gemini_3.1_pro_preview"),
            #("gemini", "gemini-3.1-flash-lite-preview", "Runs/gemini_3.1_flash_lite_preview"),
            #("gemini", "gemma-4-26b-a4b-it-maas", "Runs/gemma-4-26b-a4b-it-maas"),
]

# Keep one value or add more shot levels if needed, e.g. [0, 1, 2]
SHOTS = [0]

SUCCESS_STATUSES = {
    "success",
    "success_no_output",
    "generation_only",
    "max_iterations_reached",
}


def _scenario_stem(scenario_filename: str) -> str:
    return scenario_filename.replace(".txt", "")


def _results_dir_with_shot_suffix(results_dir: str, shot: int) -> str:
    suffix = f"_Shot{shot}"
    if results_dir.endswith(suffix):
        return results_dir
    return f"{results_dir}{suffix}"


def _latest_run_metadata(results_dir: Path, scenario_filename: str, system_prompt: str) -> Path | None:
    run_root = results_dir / _scenario_stem(scenario_filename) / system_prompt.replace(".txt", "")
    if not run_root.exists():
        return None
    candidates = sorted(run_root.glob("RUN_*/run_metadata.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _find_run_metadata_path(cfg: dict) -> Path | None:
    base_results_dir = ROOT / cfg["results_dir"]
    run_meta_path = _latest_run_metadata(
        results_dir=base_results_dir,
        scenario_filename=cfg["scenario"],
        system_prompt=cfg["system_prompt"],
    )
    # Backward compatibility with older runs created through --shots override.
    if run_meta_path is None and cfg.get("shots") is not None:
        shot_results_dir = ROOT / _results_dir_with_shot_suffix(
            str(cfg["results_dir"]),
            int(cfg["shots"]),
        )
        run_meta_path = _latest_run_metadata(
            results_dir=shot_results_dir,
            scenario_filename=cfg["scenario"],
            system_prompt=cfg["system_prompt"],
        )
    return run_meta_path


def _evaluate_cfg_from_metadata(cfg: dict) -> tuple[str, str]:
    run_meta_path = _find_run_metadata_path(cfg)
    if run_meta_path is None:
        return "failed", "missing_run_metadata"

    try:
        meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return "failed", f"invalid_run_metadata:{type(e).__name__}"

    status = str(meta.get("status") or "unknown")
    if status not in SUCCESS_STATUSES:
        be = meta.get("breaking_error") or {}
        err_msg = be.get("message") if isinstance(be, dict) else None
        if err_msg:
            return "failed", f"status={status}; error={err_msg}"
        return "failed", f"status={status}"

    return "ok", f"status={status}"


def _extract_last_checkpoint_from_metadata(meta: dict, meta_path: Path) -> tuple[Path | None, int | None]:
    iterations = meta.get("iterations")
    if isinstance(iterations, list):
        for it in sorted(
            (row for row in iterations if isinstance(row, dict)),
            key=lambda row: int(row.get("iteration", -1)),
            reverse=True,
        ):
            dsl_path_raw = it.get("dsl_path")
            if not isinstance(dsl_path_raw, str) or not dsl_path_raw.strip():
                continue
            dsl_path = Path(dsl_path_raw)
            if not dsl_path.is_absolute():
                dsl_path = meta_path.parent / dsl_path
            if dsl_path.exists():
                try:
                    iter_num = int(it.get("iteration"))
                except Exception:
                    iter_num = None
                return dsl_path, iter_num

    run_dir_raw = meta.get("run_dir")
    if isinstance(run_dir_raw, str) and run_dir_raw.strip():
        run_dir = Path(run_dir_raw)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        dsl_dir = run_dir / "dsl"
        if dsl_dir.exists():
            dsl_files = sorted(dsl_dir.glob("ITER*_*.LIRAs"), key=lambda p: p.stat().st_mtime)
            if dsl_files:
                last = dsl_files[-1]
                iter_num = None
                stem = last.stem
                if stem.startswith("ITER"):
                    num_part = stem.split("_", 1)[0].replace("ITER", "")
                    try:
                        iter_num = int(num_part)
                    except Exception:
                        iter_num = None
                return last, iter_num

    return None, None


def _build_resume_cfg_for_crash(cfg: dict) -> tuple[dict, str | None]:
    run_meta_path = _find_run_metadata_path(cfg)
    if run_meta_path is None:
        return cfg, None

    try:
        meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return cfg, None

    status = str(meta.get("status") or "").strip().lower()
    if status != "crashed":
        return cfg, None

    checkpoint_path, last_iter = _extract_last_checkpoint_from_metadata(meta, run_meta_path)
    if checkpoint_path is None:
        return cfg, "resume skipped: no DSL checkpoint found"

    raw_max_iter = cfg.get("max_iterations", meta.get("max_iterations", 1))
    try:
        max_iterations = int(raw_max_iter)
    except Exception:
        max_iterations = 1
    if max_iterations < 1:
        max_iterations = 1

    if last_iter is None:
        remaining = max_iterations
    else:
        remaining = max_iterations - (last_iter + 1)

    if remaining < 1:
        return cfg, "resume skipped: no remaining iterations budget"

    resumed_cfg = dict(cfg)
    resumed_cfg["use_generated_dsl_cache"] = True
    resumed_cfg["generated_dsl_source"] = "generated_cache"
    resumed_cfg["generated_dsl_path"] = str(checkpoint_path)
    resumed_cfg["max_iterations"] = remaining

    note = (
        f"resume enabled: status=crashed from {checkpoint_path.name}; "
        f"remaining_iterations={remaining}"
    )
    return resumed_cfg, note


def _build_jobs() -> list[dict]:
    scenarios = [p.name for p in sorted((ROOT / "Scenarios").glob("*.txt"))]
    if not scenarios:
        raise RuntimeError("No scenario files found in Scenarios/ with pattern *.txt")

    jobs: list[dict] = []
    for provider, model, outdir in COMBOS:
        for shot in SHOTS:
            for scenario in scenarios:
                cfg = dict(BASE_CONFIG)
                cfg["provider"] = provider
                cfg["generation_model"] = model
                cfg["repair_model"] = model
                cfg["system_prompt"] = PROMPT4_PATH
                cfg["repair_prompt"] = REPAIR_PROMPT4_PATH
                cfg["scenario"] = scenario
                cfg["shots"] = shot
                cfg["results_dir"] = _results_dir_with_shot_suffix(outdir, int(shot))
                jobs.append(cfg)
    return jobs


def _config_key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def _load_retry_rows(retry_file: Path) -> list[dict]:
    rows: list[dict] = []
    with open(retry_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _save_retry_rows(retry_file: Path, rows: list[dict]) -> None:
    retry_file.parent.mkdir(parents=True, exist_ok=True)
    with open(retry_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _remove_first_by_config(rows: list[dict], cfg: dict) -> bool:
    target = _config_key(cfg)
    for i, row in enumerate(rows):
        row_cfg = row.get("config") if isinstance(row, dict) else None
        if isinstance(row_cfg, dict) and _config_key(row_cfg) == target:
            rows.pop(i)
            return True
    return False


def _update_first_failure_by_config(rows: list[dict], cfg: dict, reason: str) -> bool:
    target = _config_key(cfg)
    for row in rows:
        row_cfg = row.get("config") if isinstance(row, dict) else None
        if isinstance(row_cfg, dict) and _config_key(row_cfg) == target:
            row["failed_at"] = datetime.now().isoformat()
            row["reason"] = reason
            return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Prompt4 across provider/model/scenario matrix and maintain a retry queue for failed configs."
        )
    )
    parser.add_argument(
        "--retry-file",
        default="",
        help="Path to retry queue JSONL. If set, runs only queued configs and removes successful entries.",
    )
    parser.add_argument(
        "--retry-dry-clean",
        action="store_true",
        help=(
            "Retry mode only: prune successful entries from retry queue by checking existing run metadata "
            "without executing model calls."
        ),
    )
    parser.add_argument(
        "--resume-crashed",
        action="store_true",
        help=(
            "Retry mode only: if latest run status is 'crashed', resume from the last saved "
            "ITER*.LIRAs checkpoint instead of regenerating from scratch."
        ),
    )
    return parser.parse_args()


def _run_one_job(cfg: dict, *, resume_crashed: bool = False) -> tuple[str, str]:
    run_cfg = dict(cfg)
    resume_note = None
    if resume_crashed:
        run_cfg, resume_note = _build_resume_cfg_for_crash(run_cfg)
        if resume_note:
            print(f"[RESUME] {resume_note}", flush=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(run_cfg, tf, indent=2)
        tf.flush()
        cfg_path = Path(tf.name)

    cmd = [
        "python3",
        "Utils/run_all_scenarios.py",
        "--config",
        str(cfg_path),
        "--scenario-glob",
        run_cfg["scenario"],
        "--inter-run-delay",
        "0",
    ]

    proc = subprocess.run(cmd)

    try:
        cfg_path.unlink(missing_ok=True)
    except Exception:
        pass

    if proc.returncode != 0:
        return "failed", f"runner_exit_code={proc.returncode}"
    return _evaluate_cfg_from_metadata(run_cfg)


def main() -> int:
    args = _parse_args()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    retry_mode = bool(args.retry_file.strip())
    retry_path = None
    retry_rows: list[dict] = []

    if retry_mode:
        retry_path = Path(args.retry_file).expanduser()
        if not retry_path.is_absolute():
            retry_path = ROOT / retry_path
        if not retry_path.exists():
            raise FileNotFoundError(f"Retry queue not found: {retry_path}")

        retry_rows = _load_retry_rows(retry_path)
        jobs = []
        for row in retry_rows:
            cfg = row.get("config") if isinstance(row, dict) else None
            if isinstance(cfg, dict):
                jobs.append(cfg)

        if args.retry_dry_clean:
            print(f"[DRY-CLEAN] Checking {len(jobs)} queued configs from metadata only", flush=True)
            kept_rows: list[dict] = []
            removed = 0
            unresolved = 0

            for row in retry_rows:
                cfg = row.get("config") if isinstance(row, dict) else None
                if not isinstance(cfg, dict):
                    unresolved += 1
                    kept_rows.append(row)
                    continue

                result, detail = _evaluate_cfg_from_metadata(cfg)
                if result == "ok":
                    removed += 1
                    continue

                unresolved += 1
                row["failed_at"] = datetime.now().isoformat()
                row["reason"] = detail
                kept_rows.append(row)

            _save_retry_rows(retry_path, kept_rows)
            print("\n===== DRY-CLEAN SUMMARY =====", flush=True)
            print(f"Queue file: {retry_path}", flush=True)
            print(f"Removed (already successful): {removed}", flush=True)
            print(f"Kept (still unresolved):      {unresolved}", flush=True)
            print(f"Remaining in queue:           {len(kept_rows)}", flush=True)
            return 0
    else:
        jobs = _build_jobs()

    print(
        f"[BATCH] Starting {len(jobs)} jobs | prompt={PROMPT4_PATH} | "
        f"mode={'retry' if retry_mode else 'matrix'}",
        flush=True,
    )

    failures: list[dict] = []
    successes = 0

    for i, cfg in enumerate(jobs, start=1):
        print(
            f"[{i}/{len(jobs)}] START provider={cfg['provider']} model={cfg['generation_model']} "
            f"scenario={cfg['scenario']} shot={cfg['shots']}",
            flush=True,
        )

        result, detail = _run_one_job(cfg, resume_crashed=retry_mode and args.resume_crashed)
        if result == "ok":
            successes += 1
            print(f"[{i}/{len(jobs)}] DONE {detail}", flush=True)

            # In retry mode, successful entries are removed immediately from queue.
            if retry_mode and retry_path is not None:
                removed = _remove_first_by_config(retry_rows, cfg)
                if removed:
                    _save_retry_rows(retry_path, retry_rows)
            continue

        print(f"[{i}/{len(jobs)}] FAIL {detail}", flush=True)
        failures.append(
            {
                "batch_id": batch_id,
                "failed_at": datetime.now().isoformat(),
                "reason": detail,
                "config": cfg,
            }
        )

        # Keep failed configs in retry queue and refresh failure reason/timestamp.
        if retry_mode and retry_path is not None:
            updated = _update_first_failure_by_config(retry_rows, cfg, detail)
            if not updated:
                retry_rows.append(
                    {
                        "batch_id": batch_id,
                        "failed_at": datetime.now().isoformat(),
                        "reason": detail,
                        "config": cfg,
                    }
                )
            _save_retry_rows(retry_path, retry_rows)

    if retry_mode and retry_path is not None:
        retry_file = retry_path
        _save_retry_rows(retry_file, retry_rows)
    else:
        retry_dir = ROOT / "Runs" / "retry_queues"
        retry_dir.mkdir(parents=True, exist_ok=True)
        retry_file = retry_dir / f"prompt4_retry_queue_{batch_id}.jsonl"
        _save_retry_rows(retry_file, failures)

    print("\n===== SUMMARY =====", flush=True)
    print(f"Total:   {len(jobs)}", flush=True)
    print(f"Success: {successes}", flush=True)
    print(f"Failed:  {len(failures)}", flush=True)
    if retry_mode:
        print(f"Remaining in queue: {len(retry_rows)}", flush=True)
    print(f"Retry queue: {retry_file}", flush=True)

    if failures:
        print("\nRelaunch failed configs later with this helper snippet:", flush=True)
        print(
            "python3 - <<'PY'\n"
            "import json, tempfile, subprocess\n"
            f"from pathlib import Path\nq=Path(r'{retry_file}')\n"
            "for line in q.read_text(encoding='utf-8').splitlines():\n"
            "    if not line.strip():\n"
            "        continue\n"
            "    cfg=json.loads(line)['config']\n"
            "    tf=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,encoding='utf-8')\n"
            "    json.dump(cfg, tf, indent=2); tf.close()\n"
            "    subprocess.run(['python3','Utils/run_all_scenarios.py','--config',tf.name,'--scenario-glob',cfg['scenario']])\n"
            "PY",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())