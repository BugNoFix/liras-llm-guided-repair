#!/usr/bin/env python3
"""Adapt fixed UPPAAL source queries to a generated XML model via LLM."""

from __future__ import annotations

import re
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dsl_generator import DSLGenerator

PROJECT_ROOT = Path(__file__).resolve().parent
HF_QUERY_MAX_OUTPUT_TOKEN_LIMITS = {
    "zai-org/glm-5.2": 16384,
    "glm-5.2": 16384,
}


@dataclass
class QueryAdaptationResult:
    status: str  # ok | skipped | failed
    adapted_query_path: Optional[Path] = None
    source_query_path: Optional[Path] = None
    source_queries_copy_path: Optional[Path] = None
    raw_response_path: Optional[Path] = None
    attempts_path: Optional[Path] = None
    xml_context_path: Optional[Path] = None
    error: Optional[str] = None
    failure_type: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_details: Optional[dict] = None


def scenario_stem(scenario_file: str) -> str:
    name = Path(scenario_file).name
    if name.endswith(".txt"):
        return name[:-4]
    return Path(name).stem


def _resolve_under_root(raw: str, root: Path, *, must_exist: bool = True) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if must_exist and not candidate.exists():
        raise FileNotFoundError(str(candidate))
    return candidate


def resolve_source_query_path(config: dict, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve scenario-specific source queries file under query_source_root."""
    stem = scenario_stem(str(config["scenario"]))
    source_root = config.get("query_source_root", "Queries")
    base = _resolve_under_root(str(source_root), project_root, must_exist=True)

    candidates = [
        base / f"{stem}.j2",
        base / f"{stem}.txt",
        base / f"{stem}.md",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path

    searched = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Source queries file not found for scenario '{config['scenario']}'.\n"
        f"Searched:\n - {searched}"
    )


def resolve_query_prompt_path(
    config: dict,
    key: str,
    *,
    generator: "DSLGenerator",
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a query prompt template path (system or user) under SPs/."""
    raw = config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"'{key}' must be a non-empty string when query adaptation is enabled")

    candidate = Path(raw.strip()).expanduser()
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                project_root / candidate,
                generator.sp_path / candidate,
                generator.sp_path / "Queries" / candidate.name,
            ]
        )

    for path in candidates:
        j2 = path.with_suffix(".j2")
        for option in (j2, path):
            if option.exists() and option.is_file():
                return option

    searched = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Query prompt not found for '{key}': {raw}\nSearched:\n - {searched}")


def validate_query_config(config: dict, project_root: Path = PROJECT_ROOT) -> None:
    if not bool(config.get("enable_query_adaptation", False)):
        return

    required = ("query_model", "query_system_prompt", "query_user_prompt", "query_source_root", "scenario")
    missing = [k for k in required if k not in config or config.get(k) in (None, "")]
    if missing:
        raise ValueError(f"enable_query_adaptation=true requires config keys: {missing}")

    model = config.get("query_model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("'query_model' must be a non-empty string")

    provider = str(config.get("query_provider") or "").strip().lower()
    if provider not in ("gemini", "groq", "mistral", "openrouter", "huggingface"):
        raise ValueError("'query_provider' must be 'gemini', 'groq', 'mistral', 'openrouter' or 'huggingface'")

    empty_query_adaptation_max_retries = config.get("empty_query_adaptation_max_retries", 0)
    if not isinstance(empty_query_adaptation_max_retries, int):
        raise ValueError("'empty_query_adaptation_max_retries' must be an integer")
    if empty_query_adaptation_max_retries < 0:
        raise ValueError("'empty_query_adaptation_max_retries' must be >= 0")

    resolve_source_query_path(config, project_root)


def _bounded_query_max_output_tokens(
    *,
    provider_name: str,
    model_name: str,
    max_output_tokens: Optional[int],
) -> Optional[int]:
    if max_output_tokens is None:
        return None

    normalized_provider = (provider_name or "").strip().lower()
    normalized_model = (model_name or "").strip().lower()
    if normalized_provider != "huggingface":
        return max_output_tokens

    limit = HF_QUERY_MAX_OUTPUT_TOKEN_LIMITS.get(normalized_model)
    if limit is None and normalized_model.endswith("/glm-5.2"):
        limit = 16384
    if limit is None:
        return max_output_tokens

    if max_output_tokens > limit:
        print(
            "[QUERY_ADAPTER] Capping query_max_output_tokens "
            f"{max_output_tokens} -> {limit} for {model_name} on Hugging Face."
        )
        return limit
    return max_output_tokens


def _load_source_queries(path: Path, generator: "DSLGenerator") -> str:
    if path.suffix == ".j2":
        return generator._load_prompt(path)
    return generator.load_file(path)


def _build_user_prompt(
    *,
    generator: "DSLGenerator",
    config: dict,
    scenario_name: str,
    source_queries: str,
    xml_code: str,
    project_root: Path = PROJECT_ROOT,
) -> str:
    template_path = resolve_query_prompt_path(
        config,
        "query_user_prompt",
        generator=generator,
        project_root=project_root,
    )
    return generator._render_jinja(
        template_path,
        {
            "scenario_name": scenario_name,
            "source_queries": source_queries,
            "xml_code": xml_code,
        },
    )


def _clean_query_response(response_text: str) -> str:
    if not response_text:
        return ""

    text = response_text.strip()
    text = re.sub(r"^```(?:uppaal|q|query|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.replace("```", "").strip()

    preamble_markers = ("here are", "adapted quer", "below is", "output:")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        lowered = line.strip().lower()
        if lowered and not any(marker in lowered for marker in preamble_markers):
            return "\n".join(lines[idx:]).strip()

    return text


def _resolve_queries_dir(generator: "DSLGenerator", run_dir: Path) -> Path:
    """Resolve the per-run queries output directory under results_dir."""
    queries_dir = getattr(generator, "run_queries_dir", None)
    if queries_dir is not None:
        queries_dir = Path(queries_dir)
    else:
        queries_dir = run_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    return queries_dir


def _compact_xml_for_prompt(xml_code: str, *, max_chars: int = 24000) -> str:
    """Extract a compact XML summary for LLM query adaptation."""
    import re

    parts: list[str] = []

    decl_match = re.search(r"<declaration>(.*?)</declaration>", xml_code, flags=re.DOTALL)
    if decl_match:
        decl = decl_match.group(1).strip()
        if len(decl) > 12000:
            decl = decl[:12000] + "\n... [declaration truncated] ..."
        parts.append("### GLOBAL DECLARATION\n" + decl)

    system_match = re.search(r"<system>(.*?)</system>", xml_code, flags=re.DOTALL)
    if system_match:
        parts.append("### SYSTEM INSTANCES\n" + system_match.group(1).strip())

    template_names = re.findall(r"<template>\s*<name>([^<]+)</name>", xml_code)
    if template_names:
        parts.append("### TEMPLATE NAMES\n" + "\n".join(f"- {name}" for name in template_names))

    pattern_locations: list[str] = []
    for tmpl in re.finditer(r"<template>.*?</template>", xml_code, flags=re.DOTALL):
        block = tmpl.group(0)
        name_match = re.search(r"<name>([^<]+)</name>", block)
        if not name_match:
            continue
        tmpl_name = name_match.group(1)
        if not any(
            token in tmpl_name
            for token in ("_orchestrator", "_robot", "_employee", "_def", "Pattern")
        ):
            continue
        loc_names = re.findall(r"<location[^>]*>.*?<name[^>]*>([^<]+)</name>", block, flags=re.DOTALL)
        if loc_names:
            pattern_locations.append(f"{tmpl_name}: " + ", ".join(loc_names))

    if pattern_locations:
        parts.append("### PATTERN LOCATIONS\n" + "\n".join(pattern_locations))

    embedded_queries = re.findall(
        r"<query>\s*<formula>(.*?)</formula>(?:\s*<comment>(.*?)</comment>)?",
        xml_code,
        flags=re.DOTALL,
    )
    if embedded_queries:
        query_lines = []
        for formula, comment in embedded_queries:
            formula = formula.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").strip()
            if comment and comment.strip():
                query_lines.append(f"// {comment.strip()}\n{formula}")
            else:
                query_lines.append(formula)
        parts.append("### EMBEDDED XML QUERIES (reference naming in this model)\n" + "\n\n".join(query_lines))

    compact = "\n\n".join(parts).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars] + "\n... [xml context truncated] ..."
    return compact


def generate_adapted_queries(
    *,
    config: dict,
    generator: "DSLGenerator",
    run_dir: Path,
    compiled_xml_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> QueryAdaptationResult:
    """Generate adapted UPPAAL queries for the compiled XML and save run artifacts."""
    if not bool(config.get("enable_query_adaptation", False)):
        return QueryAdaptationResult(status="skipped")

    queries_dir = _resolve_queries_dir(generator, run_dir)

    adapted_path = queries_dir / "adapted.q"
    source_copy_path = queries_dir / "source_queries.txt"
    raw_response_path = queries_dir / "query_adapter_response.txt"
    attempts_path = queries_dir / "query_adapter_attempts.jsonl"
    xml_context_path = queries_dir / "xml_context.txt"
    attempts_path.unlink(missing_ok=True)

    try:
        validate_query_config(config, project_root)
        source_path = resolve_source_query_path(config, project_root)
        source_queries = _load_source_queries(source_path, generator)
        source_copy_path.write_text(source_queries, encoding="utf-8")

        xml_code = compiled_xml_path.read_text(encoding="utf-8")
        xml_context = _compact_xml_for_prompt(xml_code)
        xml_context_path.write_text(xml_context, encoding="utf-8")
        scenario_name = scenario_stem(str(config["scenario"]))

        system_prompt_path = resolve_query_prompt_path(
            config,
            "query_system_prompt",
            generator=generator,
            project_root=project_root,
        )
        system_prompt = generator._load_prompt(system_prompt_path)
        user_prompt = _build_user_prompt(
            generator=generator,
            config=config,
            scenario_name=scenario_name,
            source_queries=source_queries,
            xml_code=xml_context,
            project_root=project_root,
        )

        model_name = str(config["query_model"]).strip()
        provider_name = str(config.get("query_provider") or "").strip().lower()
        temperature = float(config.get("query_temperature", 0.2))
        max_output_tokens = config.get("query_max_output_tokens")
        if max_output_tokens is not None:
            max_output_tokens = int(max_output_tokens)
        max_output_tokens = _bounded_query_max_output_tokens(
            provider_name=provider_name,
            model_name=model_name,
            max_output_tokens=max_output_tokens,
        )
        max_empty_retries = int(config.get("empty_query_adaptation_max_retries", 0) or 0)

        raw_response = ""
        adapted_queries = ""
        query_attempts = []
        for retry_index in range(max_empty_retries + 1):
            attempt_number = retry_index + 1
            attempt_for_log = attempt_number if max_empty_retries else None
            if max_empty_retries:
                print(
                    f"[QUERY_ADAPTER] Adapting queries with {model_name} "
                    f"for scenario={scenario_name} "
                    f"(attempt {attempt_number}/{max_empty_retries + 1})..."
                )
            else:
                print(f"[QUERY_ADAPTER] Adapting queries with {model_name} for scenario={scenario_name}...")

            raw_response = generator.call_stateless_llm(
                kind="query_adaptation",
                model_name=model_name,
                system_instruction=system_prompt,
                user_message=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                provider=provider_name,
                attempt=attempt_for_log,
            )
            adapted_queries = _clean_query_response(raw_response)
            accepted_response = bool(adapted_queries.strip())
            attempt_record = {
                "timestamp": datetime.now().isoformat(),
                "attempt": attempt_number,
                "raw_response_chars": len(raw_response or ""),
                "adapted_query_chars": len(adapted_queries or ""),
                "empty_response": not bool((raw_response or "").strip()),
                "accepted": accepted_response,
                "model": model_name,
                "provider": provider_name,
            }
            query_attempts.append(attempt_record)
            with open(attempts_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")

            if accepted_response:
                break
            if retry_index < max_empty_retries:
                if (raw_response or "").strip():
                    print(
                        "[QUERY_ADAPTER] Query model returned no usable queries; "
                        f"retrying same request ({attempt_number + 1}/{max_empty_retries + 1})."
                    )
                else:
                    print(
                        "[QUERY_ADAPTER] Query model returned an empty response; "
                        f"retrying same request ({attempt_number + 1}/{max_empty_retries + 1})."
                    )

        raw_response_path.write_text(raw_response or "", encoding="utf-8")

        if not adapted_queries.strip():
            return QueryAdaptationResult(
                status="failed",
                source_query_path=source_path,
                source_queries_copy_path=source_copy_path,
                raw_response_path=raw_response_path,
                attempts_path=attempts_path,
                error="LLM returned empty adapted queries",
                failure_type="empty_query_adaptation",
                failure_reason="EmptyQueryAdaptation",
                failure_details={
                    "operation": "llm_query_adaptation",
                    "error_type": "EmptyQueryAdaptation",
                    "error_message": "Query adaptation model returned no usable UPPAAL queries.",
                    "model": model_name,
                    "provider": provider_name,
                    "raw_response_path": str(raw_response_path),
                    "attempts_path": str(attempts_path),
                    "attempts": query_attempts,
                    "empty_query_adaptation_max_retries": max_empty_retries,
                    "raw_response_chars": len(raw_response or ""),
                    "adapted_query_chars": len(adapted_queries or ""),
                },
            )

        adapted_path.write_text(
            adapted_queries + ("\n" if not adapted_queries.endswith("\n") else ""),
            encoding="utf-8",
        )
        print(f"[QUERY_ADAPTER] Saved adapted queries to {adapted_path}")

        return QueryAdaptationResult(
            status="ok",
            adapted_query_path=adapted_path,
            source_query_path=source_path,
            source_queries_copy_path=source_copy_path,
            raw_response_path=raw_response_path,
            attempts_path=attempts_path,
            xml_context_path=xml_context_path,
        )
    except Exception as exc:
        message = str(exc)
        return QueryAdaptationResult(
            status="failed",
            adapted_query_path=adapted_path if adapted_path.exists() else None,
            source_queries_copy_path=source_copy_path if source_copy_path.exists() else None,
            raw_response_path=raw_response_path if raw_response_path.exists() else None,
            attempts_path=attempts_path if attempts_path.exists() else None,
            error=message,
            failure_type="configuration" if isinstance(exc, (FileNotFoundError, ValueError)) else "execution",
            failure_reason=f"Query adaptation failed: {message}",
            failure_details={
                "operation": "query_adaptation",
                "error_type": type(exc).__name__,
                "error_message": message,
            },
        )
