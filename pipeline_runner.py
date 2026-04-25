#!/usr/bin/env python3

import argparse
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_path(raw: str, *, must_exist: bool = True) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if must_exist and not candidate.exists():
        raise FileNotFoundError(str(candidate))
    return candidate


def _resolve_executable(raw: str) -> str:
    """Resolve executable from absolute/relative path or PATH command name."""
    value = (raw or "").strip()
    if not value:
        raise ValueError("Empty executable name")

    # Absolute paths or values containing path separators are treated as file paths.
    if "/" in value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if not candidate.exists():
            raise FileNotFoundError(str(candidate))
        return str(candidate)

    # Plain command names are resolved through PATH.
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(
            f"Executable '{value}' not found on PATH"
        )
    return resolved


def _resolve_verifyta_command(config: dict) -> str:
    """Resolve verifyta executable with explicit config and sensible fallbacks."""
    configured = config.get("verifyta_bin")
    if isinstance(configured, str) and configured.strip():
        return _resolve_executable(configured.strip())

    # Preferred fallback when a local verifyta binary/script is shipped in the repo.
    local_verifyta = PROJECT_ROOT / "verifyta"
    if local_verifyta.exists():
        return str(local_verifyta)

    resolved = shutil.which("verifyta")
    if resolved:
        return resolved

    raise FileNotFoundError(
        "verifyta executable not found. Set 'verifyta_bin' in config.json "
        "(absolute path, relative path, or command name on PATH)."
    )


def _validate_pipeline_config(config: dict) -> None:
    required_liras = ("generation_model", "shots", "system_prompt", "scenario")
    missing_liras = [k for k in required_liras if k not in config]
    if missing_liras:
        raise ValueError(f"config.json missing required LIRAS keys: {missing_liras}")

    enable_xml_export = bool(config.get("enable_xml_export", True))
    if enable_xml_export:
        jar = config.get("lira_cli_jar")
        if not isinstance(jar, str) or not jar.strip():
            raise ValueError("'lira_cli_jar' is required when enable_xml_export=true")

    if "verifyta_bin" in config and config.get("verifyta_bin") is not None:
        val = config.get("verifyta_bin")
        if not isinstance(val, str):
            raise ValueError("'verifyta_bin' must be a string when provided")


def _read_run_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    return _load_json(metadata_path)


def _write_run_metadata(metadata_path: Path, metadata: dict) -> None:
    if not metadata_path.exists():
        return
    _save_json(metadata_path, metadata)


def _update_pipeline_metadata(metadata_path: Path, updates: dict) -> None:
    metadata = _read_run_metadata(metadata_path)
    metadata.update(updates)
    _write_run_metadata(metadata_path, metadata)


def _find_success_liras_path(run_metadata: dict, run_dir: Path) -> Optional[Path]:
    summary = run_metadata.get("summary") or {}
    final_success = summary.get("final_success_dsl_path")
    if isinstance(final_success, str) and final_success.strip():
        candidate = Path(final_success)
        if candidate.exists():
            return candidate

    dsl_dir = run_dir / "dsl"
    if not dsl_dir.exists():
        return None
    success_files = sorted(dsl_dir.glob("SUCCESS_*.LIRAs"))
    if success_files:
        return success_files[-1]
    return None


def _record_pipeline_error(
    *,
    metadata_path: Path,
    step: str,
    command: list[str],
    exit_code: int,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "pipeline_final_state": "failed",
        "failed_step": step,
        "last_command": " ".join(command),
        "last_exit_code": int(exit_code),
        "last_stdout_path": str(stdout_path) if stdout_path else None,
        "last_stderr_path": str(stderr_path) if stderr_path else None,
    }
    if extra:
        payload.update(extra)
    _update_pipeline_metadata(metadata_path, payload)
    print(f"[PIPELINE_ERROR] step={step} exit_code={exit_code}")


def _run_lira_cli_to_xml(
    *,
    config: dict,
    run_dir: Path,
    metadata_path: Path,
    liras_path: Path,
) -> Optional[Path]:
    if not bool(config.get("enable_xml_export", True)):
        _update_pipeline_metadata(
            metadata_path,
            {
                "xml_status": "skipped",
                "failed_step": "none",
            },
        )
        return None

    lira_cli_jar = _resolve_path(str(config["lira_cli_jar"]))
    xml_dir = run_dir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    default_xml_name = f"{liras_path.stem}.xml"
    xml_name = str(config.get("xml_output_name") or default_xml_name)
    xml_path = xml_dir / xml_name

    command = ["java", "-jar", str(lira_cli_jar), str(liras_path), str(xml_path)]
    timeout_sec = int(config.get("lira_cli_timeout", 120))
    stdout_path = xml_dir / "lira_cli.stdout.txt"
    stderr_path = xml_dir / "lira_cli.stderr.txt"

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("java not found on PATH\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="lira_cli",
            command=command,
            exit_code=127,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            extra={"xml_status": "failed"},
        )
        return None
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text((exc.stdout or ""), encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") + f"\nTimed out after {timeout_sec}s\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="lira_cli",
            command=command,
            exit_code=124,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            extra={"xml_status": "failed"},
        )
        return None

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    if completed.returncode != 0:
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="lira_cli",
            command=command,
            exit_code=completed.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            extra={"xml_status": "failed"},
        )
        return None

    if not xml_path.exists():
        stderr_text = stderr_path.read_text(encoding="utf-8")
        stderr_path.write_text(stderr_text + "\nXML output file was not generated\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="lira_cli",
            command=command,
            exit_code=2,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            extra={"xml_status": "failed"},
        )
        return None

    _update_pipeline_metadata(
        metadata_path,
        {
            "xml_status": "ok",
            "xml_path": str(xml_path),
            "compiled_xml_path": str(xml_path),
            "last_command": " ".join(command),
            "last_exit_code": int(completed.returncode),
            "last_stdout_path": str(stdout_path),
            "last_stderr_path": str(stderr_path),
            "failed_step": "none",
        },
    )
    return xml_path


def _run_verifyta(
    *,
    config: dict,
    run_dir: Path,
    metadata_path: Path,
    compiled_xml_path: Path,
) -> bool:
    if not bool(config.get("enable_uppaal", False)):
        _update_pipeline_metadata(
            metadata_path,
            {
                "verifyta_status": "skipped",
                "pipeline_final_state": "xml_ready",
                "failed_step": "none",
            },
        )
        return True

    uppaal_dir = run_dir / "uppaal"
    uppaal_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = uppaal_dir / "verifyta.stdout.txt"
    stderr_path = uppaal_dir / "verifyta.stderr.txt"
    timeout_sec = int(config.get("verifyta_timeout", 120))

    try:
        verifyta_bin = _resolve_verifyta_command(config)
    except FileNotFoundError as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc) + "\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="verifyta",
            command=["verifyta", str(compiled_xml_path)],
            exit_code=127,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            extra={"verifyta_status": "failed"},
        )
        return False

    command = [verifyta_bin, str(compiled_xml_path)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("verifyta binary not found\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="verifyta",
            command=command,
            exit_code=127,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            extra={"verifyta_status": "failed"},
        )
        return False
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text((exc.stdout or ""), encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") + f"\nTimed out after {timeout_sec}s\n", encoding="utf-8")
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="verifyta",
            command=command,
            exit_code=124,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            extra={"verifyta_status": "failed"},
        )
        return False

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    if completed.returncode != 0:
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="verifyta",
            command=command,
            exit_code=completed.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            extra={"verifyta_status": "failed"},
        )
        return False

    _update_pipeline_metadata(
        metadata_path,
        {
            "verifyta_status": "ok",
            "verifyta_output_path": str(stdout_path),
            "last_command": " ".join(command),
            "last_exit_code": int(completed.returncode),
            "last_stdout_path": str(stdout_path),
            "last_stderr_path": str(stderr_path),
            "pipeline_final_state": "verifyta_done",
            "failed_step": "none",
        },
    )
    return True


def _run_pipeline(config: dict) -> int:
    from dsl_generator import build_generator_from_config

    _validate_pipeline_config(config)
    generator = build_generator_from_config(config)

    generator.run_automated_session(config)

    if not generator.run_dir or not generator.run_metadata_path:
        print("[PIPELINE_ERROR] step=dsl_generator exit_code=1")
        return 1

    run_dir = Path(generator.run_dir)
    metadata_path = Path(generator.run_metadata_path)
    run_metadata = _read_run_metadata(metadata_path)

    success_liras_path = _find_success_liras_path(run_metadata, run_dir)
    if success_liras_path is None:
        _record_pipeline_error(
            metadata_path=metadata_path,
            step="dsl_generator",
            command=["dsl_generator.run_automated_session"],
            exit_code=1,
            extra={"pipeline_final_state": "failed"},
        )
        return 1

    _update_pipeline_metadata(
        metadata_path,
        {
            "pipeline_runner_version": "v1",
            "pipeline_final_state": "liras_ready",
            "failed_step": "none",
            "selected_liras_path": str(success_liras_path),
        },
    )

    compiled_xml_path = _run_lira_cli_to_xml(
        config=config,
        run_dir=run_dir,
        metadata_path=metadata_path,
        liras_path=success_liras_path,
    )
    if bool(config.get("enable_xml_export", True)) and compiled_xml_path is None:
        return 1

    if compiled_xml_path is None:
        _update_pipeline_metadata(
            metadata_path,
            {
                "pipeline_final_state": "liras_ready",
                "failed_step": "none",
            },
        )
        return 0

    _update_pipeline_metadata(
        metadata_path,
        {
            "pipeline_final_state": "xml_ready",
            "failed_step": "none",
        },
    )

    verifyta_ok = _run_verifyta(
        config=config,
        run_dir=run_dir,
        metadata_path=metadata_path,
        compiled_xml_path=compiled_xml_path,
    )
    if not verifyta_ok:
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: LIRAS generation -> XML compilation -> verifyta verification."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to pipeline config.json (default: config.json)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    config = _load_json(cfg_path)
    return _run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
