#!/usr/bin/env python3
"""Probe Hugging Face Router params that may disable reasoning for Qwen 9B."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


CANDIDATES: list[dict[str, Any]] = [
    {"name": "baseline", "params": {}},
    {"name": "reasoning_effort_none", "params": {"reasoning_effort": "none"}},
    {"name": "reasoning_effort_low", "params": {"reasoning_effort": "low"}},
    {
        "name": "chat_template_enable_thinking_false",
        "params": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    },
    {
        "name": "chat_template_thinking_false",
        "params": {"extra_body": {"chat_template_kwargs": {"thinking": False}}},
    },
    {"name": "extra_body_enable_thinking_false", "params": {"extra_body": {"enable_thinking": False}}},
    {"name": "extra_body_disable_reasoning_true", "params": {"extra_body": {"disable_reasoning": True}}},
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _api_key(config: dict[str, Any]) -> str:
    raw = config.get("huggingface_api_key")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    raw = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if raw:
        return raw
    raise RuntimeError("Hugging Face API key missing. Set huggingface_api_key, HUGGINGFACE_API_KEY, or HF_TOKEN.")


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(), depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(), depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value), depth=depth + 1)
        except Exception:
            pass
    return repr(value)


def _short(value: Any, limit: int = 240) -> str:
    text = str(value).replace("\n", "\\n")
    return text[:limit] + "..." if len(text) > limit else text


def _thinking_signal(response_text: str, payload: Any) -> str | None:
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
    token_keys = {"reasoning_tokens", "thinking_tokens"}

    def present(value: Any) -> bool:
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
                key_norm = str(key).lower()
                item_path = f"{path}.{key}"
                if key_norm in token_keys and present(item):
                    return f"{item_path}={_short(item)}"
                if key_norm in thinking_keys and present(item):
                    return f"{item_path}={_short(item)}"
                found = walk(item, item_path)
                if found:
                    return found
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found = walk(item, f"{path}[{index}]")
                if found:
                    return found
        return None

    return walk(payload)


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _call_candidate(
    client: OpenAI,
    *,
    model: str,
    candidate: dict[str, Any],
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    params = {
        "model": model,
        "messages": [{"role": "user", "content": "Return exactly OK and nothing else."}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    params.update(copy.deepcopy(candidate["params"]))

    try:
        response = client.chat.completions.create(**params)
        payload = _json_safe(response)
        text = ""
        try:
            text = payload["choices"][0]["message"].get("content") or ""
        except Exception:
            pass
        signal = _thinking_signal(text, payload)
        return {
            "timestamp": datetime.now().isoformat(),
            "candidate": candidate["name"],
            "accepted": True,
            "thinking_signal": signal,
            "response_text": text,
            "request_params": _json_safe(params),
            "response_obj": payload,
        }
    except Exception as exc:
        return {
            "timestamp": datetime.now().isoformat(),
            "candidate": candidate["name"],
            "accepted": False,
            "status_code": _status_code(exc),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "request_params": _json_safe(params),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe reasoning-control params for Qwen 9B on Hugging Face Router.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="Results/hf_qwen9b_reasoning_probe.jsonl")
    args = parser.parse_args()

    if OpenAI is None:
        raise RuntimeError("The openai package is required.")

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = _read_json(config_path)

    try:
        api_key = _api_key(config)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 2

    client = OpenAI(api_key=api_key, base_url="https://router.huggingface.co/v1")

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_without_thinking: list[dict[str, Any]] = []
    with open(output_path, "a", encoding="utf-8") as f:
        for candidate in CANDIDATES:
            result = _call_candidate(
                client,
                model=args.model,
                candidate=candidate,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            status = "accepted" if result.get("accepted") else f"rejected:{result.get('status_code')}"
            signal = result.get("thinking_signal") or "no-thinking-signal"
            print(f"{candidate['name']}: {status}; {signal}")
            if result.get("accepted") and not result.get("thinking_signal") and candidate["name"] != "baseline":
                accepted_without_thinking.append(result)

    if accepted_without_thinking:
        print(f"BEST={accepted_without_thinking[0]['candidate']}")
        print(f"OUTPUT={output_path}")
        return 0

    print("BEST=NONE")
    print(f"OUTPUT={output_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
