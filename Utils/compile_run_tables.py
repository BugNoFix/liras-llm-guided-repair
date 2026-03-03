#!/usr/bin/env python3
"""Export publication-ready tables from the combined run-history CSV.

Tables produced
---------------
Table 0 – Study Summary
    Total runs, total iterations, configs, scenarios, system prompts.

Table 1 – Configuration Scorecard  (model × system_prompt × few_shots)
    Success rate, median iterations-to-success, median best error score,
    median duration, median token cost.

Table 2 – Generative Prompt × Scenario Matrix
    Success rate and mean final error score per (system_prompt, scenario) cell.
    Justifies whether failures stem from the initial DSL generation quality.

Table 3 – Time-to-Success by Configuration
    For successful runs only: median / mean / max first-success iteration,
    grouped by config.

Table 4 – Parameter Effect Isolation
    Main-effect comparison for each independent variable (model, prompt, few_shots)
    averaged across all other factors.

Table 5 – Error Category Frequency (Failed Runs)
    For unsuccessful runs: frequency distribution of compiler error types in the
    final iteration, grouped by config.

Table 6 – Failure Root-Cause by Prompt × Scenario
    For unsuccessful runs: dominant error category breakdown per
    (system_prompt, scenario) cell — demonstrates failures are linked to
    generated-DSL quality driven by the initial prompt/scenario.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load(csv_paths: list[Path]):
    import pandas as pd

    if not csv_paths:
        raise ValueError("No CSV files provided")

    chunks = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        dfi = pd.read_csv(csv_path)
        if dfi.empty:
            continue
        chunks.append(dfi)

    if not chunks:
        raise ValueError("All input CSV files are empty")

    df = pd.concat(chunks, ignore_index=True)

    # Ensure numeric types
    numeric_cols = [
        "iteration", "shots_count", "derived_success",
        "derived_first_success_iteration", "derived_run_duration_seconds",
        "derived_best_error_score", "derived_final_error_score",
        "derived_initial_error_score", "derived_monotonicity_ratio",
        "summary_prompt_tokens_est_total", "summary_response_tokens_est_total",
        "compiler_error_score", "compiler_error_lines",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Error category columns
    err_cols = [c for c in df.columns if c.startswith("err_") and not c.endswith("_json")]
    for col in err_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Build run-level dataframe (one row per run)
    run_key_col = "run_key" if "run_key" in df.columns else "run_id"
    if "config_id" in df.columns:
        uid_cols = ["config_id", run_key_col]
    else:
        uid_cols = [run_key_col]

    run_df = (
        df.sort_values(uid_cols + ["iteration"], na_position="last")
        .groupby(uid_cols, as_index=False)
        .first()
    )

    # Success boolean
    if "derived_success" in run_df.columns:
        run_df["success_bool"] = run_df["derived_success"].fillna(0).astype(float) > 0
    elif "status" in run_df.columns:
        run_df["success_bool"] = run_df["status"].astype(str).eq("success")
    else:
        run_df["success_bool"] = False

    # Total tokens
    run_df["tokens_total"] = (
        run_df.get("summary_prompt_tokens_est_total", 0).fillna(0)
        + run_df.get("summary_response_tokens_est_total", 0).fillna(0)
    )

    # Derive unified 'few_shots' (Yes/No) if not already present (backward compat)
    for frame in (df, run_df):
        if "few_shots" not in frame.columns:
            if "jshots" in frame.columns:
                frame["few_shots"] = frame["jshots"].map({"yes": "Yes", "no": "No"}).fillna("No")
            elif "repair_shots" in frame.columns:
                frame["few_shots"] = frame["repair_shots"].apply(
                    lambda v: "Yes" if pd.to_numeric(v, errors="coerce") > 0 else "No"
                )
        else:
            frame["few_shots"] = frame["few_shots"].astype(str).str.strip()

    return df, run_df


def _save_table(df, csv_path: Path, tex_path: Path | None = None) -> None:
    df.to_csv(csv_path, index=False)
    if tex_path is not None:
        try:
            tex = df.to_latex(index=False, float_format="%.3f")
            tex_path.write_text(tex, encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Table 0 – Study Summary
# ---------------------------------------------------------------------------
def _table_study_summary(df, run_df):
    import pandas as pd

    total_runs = len(run_df)
    total_iterations = int(df["iteration"].notna().sum())
    configs = int(run_df["config_id"].nunique()) if "config_id" in run_df.columns else 1
    scenarios = int(run_df["scenario"].nunique()) if "scenario" in run_df.columns else 0
    gen_prompts = int(run_df["system_prompt"].nunique()) if "system_prompt" in run_df.columns else 0
    success_count = int(run_df["success_bool"].sum())
    failure_count = total_runs - success_count
    overall_success_rate = float(run_df["success_bool"].mean()) if total_runs else 0.0

    row = {
        "total_runs": total_runs,
        "total_compile_iterations": total_iterations,
        "configurations": configs,
        "scenarios": scenarios,
        "generative_system_prompts": gen_prompts,
        "successful_runs": success_count,
        "failed_runs": failure_count,
        "overall_success_rate": round(overall_success_rate, 4),
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Table 1 – Configuration Scorecard
# ---------------------------------------------------------------------------
def _table_config_scorecard(run_df):
    import pandas as pd
    import numpy as np

    group_cols = [c for c in ["config_id", "model", "cfg_system_prompt", "few_shots"]
                  if c in run_df.columns]
    if not group_cols:
        group_cols = ["config_id"] if "config_id" in run_df.columns else []
    if not group_cols:
        return pd.DataFrame()

    g = (
        run_df.groupby(group_cols, as_index=False)
        .agg(
            runs=("success_bool", "count"),
            successes=("success_bool", "sum"),
            success_rate=("success_bool", "mean"),
            median_first_success_iter=("derived_first_success_iteration", "median"),
            mean_first_success_iter=("derived_first_success_iteration", "mean"),
            median_best_error_score=("derived_best_error_score", "median"),
            median_final_error_score=("derived_final_error_score", "median"),
            median_duration_s=("derived_run_duration_seconds", "median"),
            median_tokens=("tokens_total", "median"),
        )
        .sort_values(group_cols)
    )
    # Round
    for c in ["success_rate", "mean_first_success_iter"]:
        if c in g.columns:
            g[c] = g[c].round(3)
    return g


# ---------------------------------------------------------------------------
# Table 2 – Generative Prompt × Scenario Success Matrix
# ---------------------------------------------------------------------------
def _table_prompt_scenario_matrix(run_df):
    import pandas as pd

    group_cols = [c for c in ["system_prompt", "scenario"] if c in run_df.columns]
    if len(group_cols) < 2:
        return pd.DataFrame()

    g = (
        run_df.groupby(group_cols, as_index=False)
        .agg(
            runs=("success_bool", "count"),
            successes=("success_bool", "sum"),
            success_rate=("success_bool", "mean"),
            mean_final_error_score=("derived_final_error_score", "mean"),
            mean_initial_error_score=("derived_initial_error_score", "mean"),
        )
        .sort_values(group_cols)
    )
    for c in ["success_rate", "mean_final_error_score", "mean_initial_error_score"]:
        if c in g.columns:
            g[c] = g[c].round(3)
    return g


# ---------------------------------------------------------------------------
# Table 3 – Time-to-Success by Configuration
# ---------------------------------------------------------------------------
def _table_time_to_success(run_df):
    import pandas as pd

    succ = run_df[run_df["success_bool"]].copy()
    if succ.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["config_id", "model", "cfg_system_prompt", "few_shots"]
                  if c in succ.columns]
    if not group_cols:
        return pd.DataFrame()

    g = (
        succ.groupby(group_cols, as_index=False)
        .agg(
            successful_runs=("derived_first_success_iteration", "count"),
            median_iters_to_success=("derived_first_success_iteration", "median"),
            mean_iters_to_success=("derived_first_success_iteration", "mean"),
            max_iters_to_success=("derived_first_success_iteration", "max"),
            median_duration_s=("derived_run_duration_seconds", "median"),
        )
        .sort_values(group_cols)
    )
    for c in ["mean_iters_to_success"]:
        if c in g.columns:
            g[c] = g[c].round(2)
    return g


# ---------------------------------------------------------------------------
# Table 4 – Parameter Effect Isolation (main effects)
# ---------------------------------------------------------------------------
def _table_parameter_effects(run_df):
    import pandas as pd

    factors = {
        "model": "model",
        "cfg_system_prompt": "repair_system_prompt",
        "few_shots": "few_shots",
    }

    rows = []
    for col, label in factors.items():
        if col not in run_df.columns:
            continue
        for val in sorted(run_df[col].dropna().unique(), key=str):
            sub = run_df[run_df[col] == val]
            rows.append({
                "factor": label,
                "level": str(val),
                "runs": len(sub),
                "success_rate": round(float(sub["success_bool"].mean()), 3),
                "median_first_success_iter": sub["derived_first_success_iteration"].median(),
                "median_best_error_score": sub["derived_best_error_score"].median(),
                "median_duration_s": sub["derived_run_duration_seconds"].median(),
                "median_tokens": sub["tokens_total"].median(),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 5 – Error Category Frequency Distribution (Failed Runs)
# ---------------------------------------------------------------------------
def _table_error_frequency(df, run_df):
    import pandas as pd

    err_cols = [c for c in df.columns if c.startswith("err_") and not c.endswith("_json")]
    if not err_cols:
        return pd.DataFrame()

    # Get final iteration of each failed run
    failed_keys = set(
        run_df[~run_df["success_bool"]]["run_key"].tolist()
    )
    if not failed_keys:
        return pd.DataFrame()

    # For each failed run, take the last iteration
    failed_iters = df[df["run_key"].isin(failed_keys)].copy()
    final = (
        failed_iters.sort_values(["run_key", "iteration"])
        .groupby("run_key", as_index=False)
        .last()
    )

    if "config_id" not in final.columns:
        return pd.DataFrame()

    rows = []
    for cfg in sorted(final["config_id"].dropna().unique(), key=str):
        sub = final[final["config_id"] == cfg]
        row = {"config_id": cfg, "failed_runs": len(sub)}
        for ec in err_cols:
            # Count how many runs have >= 1 error of this type
            row[ec + "_runs"] = int((sub[ec] > 0).sum())
            row[ec + "_pct"] = round(float((sub[ec] > 0).mean()) * 100, 1) if len(sub) else 0
        rows.append(row)

    # Also add an "ALL" row
    sub = final
    row = {"config_id": "ALL", "failed_runs": len(sub)}
    for ec in err_cols:
        row[ec + "_runs"] = int((sub[ec] > 0).sum())
        row[ec + "_pct"] = round(float((sub[ec] > 0).mean()) * 100, 1) if len(sub) else 0
    rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 6 – Failure Root-Cause by Prompt × Scenario
# ---------------------------------------------------------------------------
def _table_failure_by_prompt_scenario(run_df):
    import pandas as pd

    failed = run_df[~run_df["success_bool"]].copy()
    if failed.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["system_prompt", "scenario"] if c in failed.columns]
    if len(group_cols) < 2:
        return pd.DataFrame()

    if "final_dominant_error" not in failed.columns:
        return pd.DataFrame()

    g = (
        failed.groupby(group_cols, as_index=False)
        .agg(
            failed_runs=("success_bool", "count"),
            mean_final_error_score=("derived_final_error_score", "mean"),
            mean_initial_error_score=("derived_initial_error_score", "mean"),
            dominant_error_mode=(
                "final_dominant_error",
                lambda s: s.mode().iloc[0] if not s.mode().empty else None,
            ),
            dominant_error_pct=(
                "final_dominant_error",
                lambda s: round(
                    float(s.value_counts().iloc[0]) / len(s) * 100, 1
                ) if len(s) > 0 else 0,
            ),
        )
        .sort_values(group_cols)
    )
    for c in ["mean_final_error_score", "mean_initial_error_score"]:
        if c in g.columns:
            g[c] = g[c].round(1)
    return g


# ---------------------------------------------------------------------------
# Table 7 – Status Breakdown
# ---------------------------------------------------------------------------
def _table_status_breakdown(run_df):
    import pandas as pd

    status_counts = (
        run_df["status"].fillna("unknown")
        .value_counts()
        .rename_axis("status")
        .reset_index(name="count")
    )
    status_counts["fraction"] = (
        status_counts["count"] / status_counts["count"].sum()
    ).round(3)
    return pd.DataFrame(status_counts)


# ===========================================================================
# main
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export research-paper tables from the combined run-history CSV"
    )
    parser.add_argument(
        "--csv",
        default="Report/Histories/combined_run_histories.csv",
        help="Combined run history CSV (output of compile_run_histories.py)",
    )
    parser.add_argument(
        "--csv-glob",
        default="",
        help="Alternative: glob for per-config CSVs (e.g. 'Report/Histories/c[0-9]*.csv')",
    )
    parser.add_argument(
        "--outdir", default="Report/Tables",
        help="Output directory for tables",
    )
    args = parser.parse_args()

    csv_glob = str(args.csv_glob or "").strip()
    if csv_glob:
        csv_paths = sorted(Path().glob(csv_glob))
        if not csv_paths:
            raise FileNotFoundError(f"No files matched --csv-glob: {args.csv_glob}")
    else:
        csv_paths = [Path(args.csv)]

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, run_df = _load(csv_paths)

    tables = [
        ("table00_study_summary", _table_study_summary(df, run_df)),
        ("table01_config_scorecard", _table_config_scorecard(run_df)),
        ("table02_prompt_scenario_matrix", _table_prompt_scenario_matrix(run_df)),
        ("table03_time_to_success", _table_time_to_success(run_df)),
        ("table04_parameter_effects", _table_parameter_effects(run_df)),
        ("table05_error_frequency", _table_error_frequency(df, run_df)),
        ("table06_failure_by_prompt_scenario", _table_failure_by_prompt_scenario(run_df)),
        ("table07_status_breakdown", _table_status_breakdown(run_df)),
    ]

    for name, tbl in tables:
        if tbl is None or tbl.empty:
            print(f"  SKIP {name} (no data)")
            continue
        _save_table(tbl, out_dir / f"{name}.csv", out_dir / f"{name}.tex")
        print(f"  OK   {name}.csv  ({len(tbl)} rows, {len(tbl.columns)} cols)")

    print(f"\nExported {sum(1 for _, t in tables if t is not None and not t.empty)} tables to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
