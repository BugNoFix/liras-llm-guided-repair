#!/usr/bin/env python3
"""Render every table*.csv in Report/Tables/ as a publication-quality PNG image.

Usage
-----
    python Utils/render_tables.py                       # defaults
    python Utils/render_tables.py --indir Report/Tables --outdir Report/Tables/Images

Each CSV is read with pandas and drawn via matplotlib, producing a tight
table image with alternating row shading, bold headers, and automatic
column-width sizing.  Wide tables (e.g. table05 with 24 raw columns) are
reformatted to show only the most informative columns.

Outputs land next to the CSVs by default (Report/Tables/Images/).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN   = PROJECT_ROOT / "Report" / "Tables"
DEFAULT_OUT  = PROJECT_ROOT / "Report" / "Tables" / "Images"

# ── Colour scheme ──────────────────────────────────────────────────────────
HEADER_BG    = "#2c3e50"
HEADER_FG    = "#ffffff"
ROW_EVEN     = "#f7f9fb"
ROW_ODD      = "#ffffff"
EDGE_COLOUR  = "#dee2e6"
SUMMARY_BG   = "#eaf0f6"           # used for "ALL" / totals row

# ── Per-table display configuration ────────────────────────────────────────
# Keys that match the CSV stem.  Values specify:
#   title   – human-readable table caption
#   drop    – columns to drop before rendering
#   rename  – {old: new} column renames
#   fmt     – {col: python_format_spec}  applied to numeric cells
#   pct     – columns to render as "xx.x %"
#   int_cols– columns to render as integer (no decimals)
#   note    – optional footnote string

_PCT  = lambda cols: {c: "{:.1f}%" for c in cols}
_INT  = lambda cols: {c: "{:.0f}"  for c in cols}

TABLE_CONFIG: dict[str, dict] = {
    "table00_study_summary": dict(
        title="Table 0 — Study Summary",
        rename={
            "total_runs": "Runs",
            "total_compile_iterations": "Iterations",
            "configurations": "Configs",
            "scenarios": "Scenarios",
            "generative_system_prompts": "Gen. SPs",
            "successful_runs": "Successes",
            "failed_runs": "Failures",
            "overall_success_rate": "Success %",
        },
        fmt={"Success %": "{:.1%}"},
    ),
    "table01_config_scorecard": dict(
        title="Table 1 — Configuration Scorecard",
        rename={
            "config_id": "Config",
            "model": "Model",
            "cfg_system_prompt": "Repair SP",
            "few_shots": "Few-Shot Ex.",
            "runs": "Runs",
            "successes": "Succ.",
            "success_rate": "Rate",
            "median_first_success_iter": "Med. Iter",
            "mean_first_success_iter": "Mean Iter",
            "median_best_error_score": "Med. Best Err",
            "median_final_error_score": "Med. Final Err",
            "median_duration_s": "Med. Dur (s)",
            "median_tokens": "Med. Tokens",
        },
        fmt={
            "Rate": "{:.1%}",
            "Med. Iter": "{:.1f}",
            "Mean Iter": "{:.1f}",
            "Med. Best Err": "{:.0f}",
            "Med. Final Err": "{:.0f}",
            "Med. Dur (s)": "{:.1f}",
            "Med. Tokens": "{:.0f}",
        },
    ),
    "table02_prompt_scenario_matrix": dict(
        title="Table 2 — Prompt × Scenario matrix",
        rename={
            "system_prompt": "System Prompt",
            "scenario": "Scenario",
            "runs": "Runs",
            "successes": "Succ.",
            "success_rate": "Rate",
            "mean_final_error_score": "Mean Final Err",
            "mean_initial_error_score": "Mean Init. Err",
        },
        fmt={
            "Rate": "{:.1%}",
            "Mean Final Err": "{:.1f}",
            "Mean Init. Err": "{:.1f}",
        },
    ),
    "table03_time_to_success": dict(
        title="Table 3 — Time-to-Success",
        rename={
            "config_id": "Config",
            "model": "Model",
            "cfg_system_prompt": "Repair SP",
            "few_shots": "Few-Shot Ex.",
            "successful_runs": "Succ. Runs",
            "median_iters_to_success": "Med. Iter",
            "mean_iters_to_success": "Mean Iter",
            "max_iters_to_success": "Max Iter",
            "median_duration_s": "Med. Dur (s)",
        },
        fmt={
            "Med. Iter": "{:.1f}",
            "Mean Iter": "{:.2f}",
            "Max Iter": "{:.0f}",
            "Med. Dur (s)": "{:.1f}",
        },
    ),
    "table04_parameter_effects": dict(
        title="Table 4 — Main-Effect Parameter Isolation",
        rename={
            "factor": "Factor",
            "level": "Level",
            "runs": "Runs",
            "success_rate": "Rate",
            "median_first_success_iter": "Med. Iter",
            "median_best_error_score": "Med. Best Err",
            "median_duration_s": "Med. Dur (s)",
            "median_tokens": "Med. Tokens",
        },
        fmt={
            "Rate": "{:.1%}",
            "Med. Iter": "{:.1f}",
            "Med. Best Err": "{:.0f}",
            "Med. Dur (s)": "{:.1f}",
            "Med. Tokens": "{:.0f}",
        },
    ),
    "table05_error_frequency": dict(
        title="Table 5 — Error-Category Frequency (failed runs)",
        # Drop raw-count columns; keep only config_id, failed_runs, and *_pct
        note="Values are % of failed runs exhibiting each error category.",
    ),
    "table06_failure_by_prompt_scenario": dict(
        title="Table 6 — Failure Analysis by Prompt × Scenario",
        rename={
            "system_prompt": "System Prompt",
            "scenario": "Scenario",
            "failed_runs": "Failed",
            "mean_final_error_score": "Mean Final Err",
            "mean_initial_error_score": "Mean Init. Err",
            "dominant_error_mode": "Dominant Error",
            "dominant_error_pct": "Dom. %",
        },
        fmt={
            "Mean Final Err": "{:.1f}",
            "Mean Init. Err": "{:.1f}",
            "Dom. %": "{:.1f}%",
        },
    ),
    "table07_status_breakdown": dict(
        title="Table 7 — Run Status Breakdown",
        rename={
            "status": "Status",
            "count": "Count",
            "fraction": "Fraction",
        },
        fmt={"Fraction": "{:.1%}"},
    ),
}


# ── Table 05 special handling ──────────────────────────────────────────────

def _reshape_table05(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the pct columns (plus config_id and failed_runs).

    Renames them to short human-readable labels.
    """
    keep = ["config_id", "failed_runs"]
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    keep += pct_cols
    df = df[[c for c in keep if c in df.columns]].copy()

    rename = {
        "config_id": "Config",
        "failed_runs": "Failed",
    }
    for c in pct_cols:
        short = (
            c.replace("err_", "")
            .replace("_pct", "")
            .replace("semantic_", "sem:")
            .replace("_", " ")
            .title()
        )
        rename[c] = short

    df = df.rename(columns=rename)

    # Format pct values
    for col in df.columns:
        if col not in ("Config", "Failed"):
            df[col] = df[col].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

    return df


# ── Core rendering ─────────────────────────────────────────────────────────

def _format_cell(val, fmt_spec: str | None) -> str:
    """Apply a format spec to a value, handling NaN gracefully."""
    if pd.isna(val):
        return ""
    if fmt_spec is None:
        # Auto-format: round floats, leave strings as-is
        if isinstance(val, float):
            if val == int(val):
                return str(int(val))
            return f"{val:.2f}"
        return str(val)
    try:
        return fmt_spec.format(val)
    except (ValueError, TypeError):
        return str(val)


def _render_table(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    note: str | None = None,
):
    """Draw *df* as a styled table image and save to *out_path*."""
    n_rows, n_cols = df.shape

    # ── Sizing heuristic ──
    col_widths: list[float] = []
    for col in df.columns:
        max_len = max(len(str(col)), df[col].astype(str).str.len().max())
        col_widths.append(max(max_len * 0.115, 0.6))

    fig_w = max(sum(col_widths) + 0.6, 6.0)
    row_h = 0.38
    fig_h = (n_rows + 2) * row_h + (0.6 if note else 0.2) + 0.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Title
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14, loc="left")

    # Build table
    cell_text = df.values.tolist()
    col_labels = list(df.columns)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
        colWidths=[w / sum(col_widths) for w in col_widths],
    )
    tbl.auto_set_font_size(False)

    font_size = 9 if n_cols <= 10 else (7.5 if n_cols <= 16 else 6.5)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.6)

    # ── Style cells ──
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE_COLOUR)
        cell.set_linewidth(0.5)

        if row == 0:
            # Header
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight="bold")
        else:
            # Data row
            text = cell.get_text().get_text()
            is_summary = text.strip().upper() in ("ALL", "TOTAL")
            if is_summary:
                cell.set_facecolor(SUMMARY_BG)
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor(ROW_EVEN if row % 2 == 0 else ROW_ODD)

    # Footnote
    if note:
        fig.text(
            0.02, 0.01, note,
            fontsize=7, fontstyle="italic", color="#555555",
            verticalalignment="bottom",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_csv(csv_path: Path, out_dir: Path):
    """Read a single CSV and render it as a PNG."""
    stem = csv_path.stem
    cfg = TABLE_CONFIG.get(stem, {})

    df = pd.read_csv(csv_path)

    # Special reshape for table05
    if stem == "table05_error_frequency":
        df = _reshape_table05(df)
    else:
        # General formatting pipeline
        drop = cfg.get("drop", [])
        if drop:
            df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

        rename = cfg.get("rename", {})
        if rename:
            df = df.rename(columns=rename)

        fmt = cfg.get("fmt", {})
        for col in df.columns:
            if col in fmt:
                df[col] = df[col].apply(lambda v, f=fmt[col]: _format_cell(v, f))

    title = cfg.get("title", stem.replace("_", " ").title())
    note = cfg.get("note")

    out_path = out_dir / f"{stem}.png"
    _render_table(df, title, out_path, note=note)
    print(f"  OK   {out_path.name:45s}  ({df.shape[0]} rows x {df.shape[1]} cols)")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Render Report/Tables/*.csv as PNG images.")
    ap.add_argument("--indir",  default=str(DEFAULT_IN),  help="Directory containing table*.csv files")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT), help="Output directory for PNG images")
    args = ap.parse_args()

    in_dir  = Path(args.indir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_dir.glob("table*.csv"))
    if not csvs:
        print(f"No table*.csv files found in {in_dir}")
        return 1

    print(f"Rendering {len(csvs)} tables from {in_dir}\n")
    for csv_path in csvs:
        render_csv(csv_path, out_dir)

    print(f"\nExported {len(csvs)} table images to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
