#!/usr/bin/env python3
"""Print models exposed by the Hugging Face OpenAI-compatible inference router."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_api_key(config: dict[str, Any]) -> str:
    for key in ("huggingface_api_key", "hf_token", "api_key"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for env_key in ("HUGGINGFACE_API_KEY", "HF_TOKEN"):
        value = os.environ.get(env_key)
        if value and value.strip():
            return value.strip()

    raise RuntimeError(
        "Hugging Face API key not found. Set 'huggingface_api_key' in config.json "
        "or export HUGGINGFACE_API_KEY/HF_TOKEN."
    )


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if isinstance(model, dict):
        return model
    return dict(vars(model))


def _model_label(model: dict[str, Any]) -> str:
    model_id = str(model.get("id") or model.get("model") or "")
    owned_by = model.get("owned_by")
    provider = model.get("provider")
    suffix_parts = [str(value) for value in (provider, owned_by) if value]
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"{model_id}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List models available through Hugging Face's OpenAI-compatible inference router."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json for huggingface_api_key (default: config.json).",
    )
    parser.add_argument(
        "--base-url",
        default="https://router.huggingface.co/v1",
        help="OpenAI-compatible base URL (default: Hugging Face router).",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Case-insensitive substring filter applied to the model JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full model objects as JSON instead of one model per line.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = _load_config(config_path)

    client = OpenAI(api_key=_resolve_api_key(config), base_url=args.base_url)
    models_page = client.models.list()
    models = [_model_to_dict(model) for model in models_page.data]

    if args.filter:
        needle = args.filter.lower()
        models = [
            model
            for model in models
            if needle in json.dumps(model, ensure_ascii=False, sort_keys=True).lower()
        ]

    models.sort(key=lambda item: str(item.get("id") or item.get("model") or "").lower())

    if args.json:
        print(json.dumps(models, indent=2, ensure_ascii=False))
    else:
        for model in models:
            print(_model_label(model))
        print(f"\nTotal models: {len(models)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
