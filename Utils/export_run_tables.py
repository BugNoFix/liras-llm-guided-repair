#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def _load(csv_path: Path):
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    numeric_cols = [
        "iteration",
        "shots_count",
        "derived_success",
        "derived_first_success_iteration",
        "derived_run_duration_seconds",
        "derived_best_error_score",
        "derived_monotonicity_ratio",
        "summary_prompt_tokens_est_total",
        "summary_response_tokens_est_total",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    run_df = (
        df.sort_values(["run_id", "iteration"], na_position="last")
        .groupby("run_id", as_index=False)
        .first()
    )

    if "derived_success" in run_df.columns:
        run_df["success_bool"] = run_df["derived_success"].fillna(0).astype(float) > 0
    elif "status" in run_df.columns:
        run_df["success_bool"] = run_df["status"].astype(str).eq("success")
    else:
        run_df["success_bool"] = False

    return df, run_df


def _save_table(df, csv_path: Path, tex_path: Path | None = None) -> None:
    df.to_csv(csv_path, index=False)
    if tex_path is not None:
        try:
            tex = df.to_latex(index=False, float_format="%.3f")
            tex_path.write_text(tex, encoding="utf-8")
        except Exception:
            pass


def _table_overall(run_df):
    import pandas as pd

    total = len(run_df)
    success_rate = float(run_df["success_bool"].mean()) if total else 0.0

    row = {
        "runs": total,
        "success_rate": success_rate,
        "median_duration_s": run_df["derived_run_duration_seconds"].median(),
        "median_first_success_iter": run_df["derived_first_success_iteration"].median(),
        "median_best_error_score": run_df["derived_best_error_score"].median(),
        "median_monotonicity": run_df["derived_monotonicity_ratio"].median(),
        "median_prompt_tokens": run_df["summary_prompt_tokens_est_total"].median(),
        "median_response_tokens": run_df["summary_response_tokens_est_total"].median(),
    }
    return pd.DataFrame([row])


def _table_prompt_shots(run_df):
    grouped = (
        run_df.groupby(["system_prompt", "shots_count"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            success_rate=("success_bool", "mean"),
            median_duration_s=("derived_run_duration_seconds", "median"),
            median_first_success_iter=("derived_first_success_iteration", "median"),
            median_best_error_score=("derived_best_error_score", "median"),
            median_monotonicity=("derived_monotonicity_ratio", "median"),
        )
        .sort_values(["system_prompt", "shots_count"])
    )
    return grouped


def _table_scenario_prompt(run_df):
    grouped = (
        run_df.groupby(["scenario", "system_prompt"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            success_rate=("success_bool", "mean"),
            median_duration_s=("derived_run_duration_seconds", "median"),
            median_best_error_score=("derived_best_error_score", "median"),
        )
        .sort_values(["scenario", "system_prompt"])
    )
    return grouped


def _table_status_breakdown(run_df):
    import pandas as pd

    status_counts = run_df["status"].fillna("unknown").value_counts().rename_axis("status").reset_index(name="count")
    status_counts["fraction"] = status_counts["count"] / status_counts["count"].sum()
    return pd.DataFrame(status_counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export publication tables from Report/run_history.csv")
    parser.add_argument("--csv", default="Report/run_history.csv", help="Input run history CSV")
    parser.add_argument("--outdir", default="Report/Tables", help="Output directory for tables")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, run_df = _load(csv_path)

    t1 = _table_overall(run_df)
    t2 = _table_prompt_shots(run_df)
    t3 = _table_scenario_prompt(run_df)
    t4 = _table_status_breakdown(run_df)

    _save_table(t1, out_dir / "table01_overall_metrics.csv", out_dir / "table01_overall_metrics.tex")
    _save_table(t2, out_dir / "table02_prompt_shots.csv", out_dir / "table02_prompt_shots.tex")
    _save_table(t3, out_dir / "table03_scenario_prompt.csv", out_dir / "table03_scenario_prompt.tex")
    _save_table(t4, out_dir / "table04_status_breakdown.csv", out_dir / "table04_status_breakdown.tex")

    print(f"Exported tables to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
