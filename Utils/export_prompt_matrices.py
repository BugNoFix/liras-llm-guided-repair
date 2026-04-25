#!/usr/bin/env python3

"""Export scenario x model-shot matrices for Prompt4 and Prompt5.

Reads Report/Tables/RunSummaries/runs_summary.csv and writes a markdown file
with two tables:
- Prompt4 matrix
- Prompt5 matrix

Cell format: "<iterations> (<status>)" based on latest run for that key.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_CSV = PROJECT_ROOT / "Report" / "Tables" / "RunSummaries" / "runs_summary.csv"
DEFAULT_OUT_MD = PROJECT_ROOT / "Report" / "Tables" / "RunSummaries" / "prompt_model_shot_tables.md"


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_dt(value: str) -> datetime:
    text = (value or "").strip()
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return datetime.min


def _short_model(model: str) -> str:
    m = (model or "").strip().lower()
    if "2.5-flash" in m:
        return "flash25"
    if "3.1-pro" in m:
        return "pro31"
    return m.replace("gemini-", "") or "unknown"


def _prompt_tag(prompt: str) -> str:
    p = (prompt or "").strip()
    if p.endswith("NewSp4.txt"):
        return "prompt4"
    if p.endswith("NewSp5.txt"):
        return "prompt5"
    return "other"


def _is_stateless(value: str) -> bool:
    return str(value or "").strip().lower() == "true"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        safe = [str(v).replace("|", "\\|").replace("\n", " ") for v in row]
        out.append("| " + " | ".join(safe) + " |")
    return "\n".join(out) + "\n"


def _build_matrix(rows: list[dict], prompt_tag: str, *, stateless_mode: str = "all") -> str:
    # Keep only runs for requested prompt.
    candidates = [r for r in rows if _prompt_tag(r.get("system_prompt", "")) == prompt_tag]
    if stateless_mode == "only_true":
        candidates = [r for r in candidates if _is_stateless(r.get("repair.repair_stateless", ""))]
    elif stateless_mode == "only_false":
        candidates = [r for r in candidates if not _is_stateless(r.get("repair.repair_stateless", ""))]

    # Latest run per (scenario, model_short, shot)
    latest: dict[tuple[str, str, str], dict] = {}
    for r in candidates:
        scenario = (r.get("scenario") or "").strip()
        model_short = _short_model(r.get("generation_model", ""))
        shot = str(r.get("shots") or "").strip()
        key = (scenario, model_short, shot)
        prev = latest.get(key)
        if prev is None or _parse_dt(r.get("run_started_at", "")) > _parse_dt(prev.get("run_started_at", "")):
            latest[key] = r

    scenarios = sorted({k[0] for k in latest.keys()})
    headers = [
        "scenario",
        "flash25_s0",
        "flash25_s1",
        "flash25_s2",
        "pro31_s0",
        "pro31_s1",
        "pro31_s2",
    ]

    table_rows: list[list[str]] = []
    for s in scenarios:
        row = [s]
        for model in ("flash25", "pro31"):
            for shot in ("0", "1", "2"):
                cell = "-"
                rec = latest.get((s, model, shot))
                if rec is not None:
                    it = str(rec.get("derived.iteration_count") or "")
                    st = str(rec.get("status") or "")
                    cell = f"{it} ({st})" if it else st
                row.append(cell)
        table_rows.append(row)

    return _md_table(headers, table_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Prompt4/Prompt5 scenario x model-shot markdown tables")
    ap.add_argument("--runs-csv", default=str(DEFAULT_RUNS_CSV), help="Path to runs_summary.csv")
    ap.add_argument("--out", default=str(DEFAULT_OUT_MD), help="Output markdown path")
    args = ap.parse_args()

    runs_csv = Path(args.runs_csv)
    if not runs_csv.is_absolute():
        runs_csv = PROJECT_ROOT / runs_csv
    out_md = Path(args.out)
    if not out_md.is_absolute():
        out_md = PROJECT_ROOT / out_md

    rows = _read_rows(runs_csv)

    lines: list[str] = []
    lines.append("# Prompt Model Shot Tables")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("Cell format: iterations (status), using latest run for each scenario/model/shot.")
    lines.append("")
    lines.append("## Prompt4")
    lines.append("")
    lines.append(_build_matrix(rows, "prompt4", stateless_mode="only_false"))
    lines.append("## Prompt5")
    lines.append("")
    lines.append(_build_matrix(rows, "prompt5", stateless_mode="only_false"))
    lines.append("## Prompt4 Stateless")
    lines.append("")
    lines.append(_build_matrix(rows, "prompt4", stateless_mode="only_true"))
    lines.append("## Prompt5 Stateless")
    lines.append("")
    lines.append(_build_matrix(rows, "prompt5", stateless_mode="only_true"))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())