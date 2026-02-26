#!/usr/bin/env python3

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
import re


def _to_bool(series):
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _wilson_ci(p: float, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _style():
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="talk")
    except Exception:
        pass


def _config_from_filename(path: Path) -> str:
    return path.stem


def _config_sort_key(label: str):
    m = re.fullmatch(r"c(\d+)", str(label).strip(), flags=re.IGNORECASE)
    if m:
        return (0, int(m.group(1)))
    return (1, str(label))


def _load_data_from_paths(csv_paths: list[Path]):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'pandas'. Install with: pip install pandas matplotlib seaborn"
        ) from exc

    if not csv_paths:
        raise ValueError("No CSV files provided")

    chunks = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        dfi = pd.read_csv(csv_path)
        dfi["source_history_file"] = str(csv_path)
        dfi["source_config"] = _config_from_filename(csv_path)
        chunks.append(dfi)

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        raise ValueError("Input run_history CSVs have no rows")

    numeric_cols = [
        "iteration",
        "shots_count",
        "derived_success",
        "derived_first_success_iteration",
        "derived_run_duration_seconds",
        "summary_prompt_tokens_est_total",
        "summary_response_tokens_est_total",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "run_id" not in df.columns and "run_key" not in df.columns:
        raise ValueError("Expected 'run_id' column in run_history.csv")

    run_key_col = "run_key" if "run_key" in df.columns else "run_id"
    df["run_uid"] = (
        df["source_history_file"].astype(str)
        + "::"
        + df[run_key_col].astype(str)
    )

    run_df = (
        df.sort_values(["run_uid", "iteration"], na_position="last")
        .groupby("run_uid", as_index=False)
        .first()
    )

    if "derived_success" in run_df.columns:
        run_df["success_bool"] = run_df["derived_success"].fillna(0).astype(float) > 0
    elif "status" in run_df.columns:
        run_df["success_bool"] = run_df["status"].astype(str).eq("success")
    else:
        run_df["success_bool"] = False

    if "is_valid" in df.columns:
        df["is_valid_bool"] = _to_bool(df["is_valid"])
    else:
        df["is_valid_bool"] = False

    init_valid = (
        df[df["iteration"].fillna(-1) == 0]
        .groupby("run_uid", as_index=False)["is_valid_bool"]
        .max()
        .rename(columns={"is_valid_bool": "initial_compile_success"})
    )
    run_df = run_df.merge(init_valid, on="run_uid", how="left")
    run_df["initial_compile_success"] = run_df["initial_compile_success"].fillna(False)

    run_df["tokens_total"] = (
        run_df.get("summary_prompt_tokens_est_total", 0).fillna(0)
        + run_df.get("summary_response_tokens_est_total", 0).fillna(0)
    )

    return df, run_df


def _load_paths_from_manifest(manifest_path: Path):
    import pandas as pd

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    mdf = pd.read_csv(manifest_path)
    if "history_csv" not in mdf.columns:
        raise ValueError("Manifest missing required column: history_csv")

    paths = [Path(p) for p in mdf["history_csv"].dropna().astype(str).tolist()]
    label_map = None
    if "label" in mdf.columns:
        pairs = mdf[["history_csv", "label"]].dropna()
        label_map = {str(Path(h)): str(l) for h, l in pairs.values.tolist()}
    return paths, label_map


def _attach_config_labels(run_df, config_cols: list[str], file_label_map: dict[str, str] | None = None):
    import pandas as pd

    if file_label_map:
        run_df = run_df.copy()
        run_df["config_details"] = run_df["source_history_file"].map(
            lambda p: file_label_map.get(str(Path(p)), str(Path(p)))
        )
        unique_details = sorted(run_df["config_details"].dropna().unique().tolist())
        run_df["config_label"] = run_df["config_details"]
        mapping = pd.DataFrame([{"config_label": d, "config_details": d} for d in unique_details])
        return run_df, mapping

    if "source_config" in run_df.columns and run_df["source_config"].notna().any():
        run_df = run_df.copy()
        run_df["config_details"] = run_df["source_config"].astype(str)
        unique_details = sorted(run_df["config_details"].dropna().unique().tolist(), key=_config_sort_key)
        run_df["config_label"] = run_df["config_details"]
        mapping = pd.DataFrame([{"config_label": d, "config_details": d} for d in unique_details])
        return run_df, mapping

    available = [c for c in config_cols if c in run_df.columns]
    if not available:
        run_df = run_df.copy()
        run_df["config_details"] = "default"
        run_df["config_label"] = "C1"
        mapping = pd.DataFrame([{"config_label": "C1", "config_details": "default"}])
        return run_df, mapping

    def mk(row):
        parts = []
        for c in available:
            v = row.get(c)
            parts.append(f"{c}={v}")
        return " | ".join(parts)

    run_df = run_df.copy()
    run_df["config_details"] = run_df.apply(mk, axis=1)

    unique_details = sorted(run_df["config_details"].dropna().unique().tolist())
    label_map = {d: f"c{i+1}" for i, d in enumerate(unique_details)}
    run_df["config_label"] = run_df["config_details"].map(label_map)

    mapping = pd.DataFrame(
        [{"config_label": label_map[d], "config_details": d} for d in unique_details]
    )
    return run_df, mapping


def _figure_viability_funnel(run_df, out_dir: Path):
    import matplotlib.pyplot as plt
    import pandas as pd

    required = {"config_label", "initial_compile_success", "success_bool", "run_uid"}
    if not required.issubset(set(run_df.columns)):
        return

    g = run_df.groupby("config_label", as_index=False).agg(
        total_runs=("run_uid", "count"),
        initial_compile_ok=("initial_compile_success", "sum"),
        final_success=("success_bool", "sum"),
    )

    long_df = pd.DataFrame(
        {
            "config_label": g["config_label"].repeat(3).values,
            "stage": ["Total", "Initial compile", "Final success"] * len(g),
            "value": [v for row in g[["total_runs", "initial_compile_ok", "final_success"]].values for v in row],
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    configs = sorted(long_df["config_label"].unique(), key=_config_sort_key)
    stages = ["Total", "Initial compile", "Final success"]
    width = 0.8 / max(1, len(configs))

    x = list(range(len(stages)))
    for i, cfg in enumerate(configs):
        vals = [
            long_df[(long_df["config_label"] == cfg) & (long_df["stage"] == s)]["value"].iloc[0]
            for s in stages
        ]
        pos = [p - 0.4 + width / 2 + i * width for p in x]
        bars = ax.bar(pos, vals, width=width, label=cfg)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Number of Runs")
    ax.set_title("Figure 1: Viability Funnel by Pipeline Configuration")
    ax.legend(title="Config")
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_viability_funnel.png", dpi=220)
    plt.close(fig)


def _figure_success_rate_ci(run_df, out_dir: Path):
    import matplotlib.pyplot as plt

    required = {"config_label", "success_bool", "run_uid"}
    if not required.issubset(set(run_df.columns)):
        return

    g = run_df.groupby("config_label", as_index=False).agg(
        n=("run_uid", "count"),
        successes=("success_bool", "sum"),
    )
    g = g.sort_values("config_label", key=lambda s: s.map(_config_sort_key))
    g["rate"] = g["successes"] / g["n"]

    ci = g.apply(lambda r: _wilson_ci(float(r["rate"]), int(r["n"])), axis=1)
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]

    x = list(range(len(g)))
    y = g["rate"].values
    err_low = y - g["ci_low"].values
    err_high = g["ci_high"].values - y

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=[err_low, err_high], fmt="none", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(g["config_label"].tolist())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title("Figure 2: Success Rate with 95% CI by Configuration")
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_success_rate_with_ci.png", dpi=220)
    plt.close(fig)


def _figure_iterations_success_ecdf(run_df, out_dir: Path):
    import matplotlib.pyplot as plt

    required = {"config_label", "success_bool", "derived_first_success_iteration", "run_uid"}
    if not required.issubset(set(run_df.columns)):
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for cfg in sorted(run_df["config_label"].dropna().unique(), key=_config_sort_key):
        subset = run_df[(run_df["config_label"] == cfg) & (run_df["success_bool"])].copy()
        s = subset["derived_first_success_iteration"].dropna().sort_values()
        total_cfg = int((run_df["config_label"] == cfg).sum())
        succ_cfg = int((run_df[(run_df["config_label"] == cfg)]["success_bool"]).sum())

        if s.empty:
            continue

        y = (s.rank(method="first") / len(s)).values
        ax.step(s.values, y, where="post", label=f"{cfg} ({succ_cfg}/{total_cfg} success)")

    ax.set_xlabel("Iteration of First Success")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 3: Iteration-to-Success ECDF by Configuration")
    ax.legend(title="Config")
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_iterations_to_success_ecdf.png", dpi=220)
    plt.close(fig)


def _figure_cost_to_success(run_df, out_dir: Path):
    import matplotlib.pyplot as plt

    required = {"config_label", "success_bool", "tokens_total", "derived_run_duration_seconds"}
    if not required.issubset(set(run_df.columns)):
        return

    succ = run_df[run_df["success_bool"]].dropna(subset=["tokens_total", "derived_run_duration_seconds"])
    if succ.empty:
        return

    cfgs = sorted(succ["config_label"].unique(), key=_config_sort_key)
    token_data = [succ[succ["config_label"] == c]["tokens_total"].values for c in cfgs]
    dur_data = [succ[succ["config_label"] == c]["derived_run_duration_seconds"].values for c in cfgs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    ax1, ax2 = axes[0, 0], axes[0, 1]

    ax1.boxplot(token_data, tick_labels=cfgs)
    ax1.set_title("Token Cost (successful runs)")
    ax1.set_ylabel("Estimated total tokens")

    ax2.boxplot(dur_data, tick_labels=cfgs)
    ax2.set_title("Wall-clock cost (successful runs)")
    ax2.set_ylabel("Run duration (seconds)")

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Figure 4: Cost-to-Success by Configuration")
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_cost_to_success.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate viability figures from one or many run_history CSV files")
    parser.add_argument("--csv", default="Report/Histories/c1.csv", help="Single input run history CSV")
    parser.add_argument(
        "--csv-glob",
        default="Report/Histories/c*.csv",
        help="Glob for multiple run_history CSV files (e.g., 'Report/Histories/c*.csv')",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest CSV from run_all_configurations.py (uses history_csv + label)",
    )
    parser.add_argument("--outdir", default="Report/Figures", help="Output directory for PNG figures")
    parser.add_argument(
        "--config-cols",
        default="repair_prompt,repair_model,repair_temperature,generation_model,generation_temperature,max_iterations",
        help=(
            "Comma-separated columns defining a pipeline configuration. "
            "Scenario/system prompt/shot are intentionally excluded."
        ),
    )
    args = parser.parse_args()

    file_label_map = None
    if args.manifest and str(args.manifest).strip():
        csv_paths, file_label_map = _load_paths_from_manifest(Path(str(args.manifest).strip()))
        if not csv_paths:
            raise FileNotFoundError(f"No history CSVs found in manifest: {args.manifest}")
    elif args.csv_glob and str(args.csv_glob).strip():
        csv_paths = sorted(Path().glob(str(args.csv_glob).strip()))
        if not csv_paths:
            raise FileNotFoundError(f"No files matched --csv-glob: {args.csv_glob}")
    else:
        csv_paths = [Path(args.csv)]

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _style()
    _, run_df = _load_data_from_paths(csv_paths)

    config_cols = [c.strip() for c in str(args.config_cols).split(",") if c.strip()]
    run_df, mapping = _attach_config_labels(run_df, config_cols, file_label_map=file_label_map)
    mapping.to_csv(out_dir / "config_mapping.csv", index=False)

    _figure_viability_funnel(run_df, out_dir)
    _figure_success_rate_ci(run_df, out_dir)
    _figure_iterations_success_ecdf(run_df, out_dir)
    _figure_cost_to_success(run_df, out_dir)

    print(f"Generated figures in: {out_dir}")
    print(f"Compared input files: {len(csv_paths)}")
    print(f"Configuration mapping: {out_dir / 'config_mapping.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
