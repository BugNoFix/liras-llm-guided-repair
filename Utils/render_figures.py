#!/usr/bin/env python3
"""Generate publication figures for the cross-comparative LLM-guided DSL repair study.

Reads  combined_run_histories.csv  (produced by compile_run_histories.py)
and exports eight PNG figures to Report/Figures/.

Figures
-------
1  Success Rate per Configuration (95 % Wilson CI)
2  Main-Effect Forest Plot
3  Factor Interaction: Model x Few-Shot x Repair Prompt
4  Prompt x Scenario Success-Rate Heatmap
5  Iteration-to-Success ECDF
6  Error-Score Convergence by Configuration
7  Error-Category Prevalence Heatmap (failed runs)
8  Scenario Difficulty Profile
"""

from __future__ import annotations

import argparse
import re
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "Report" / "Histories" / "combined_run_histories.csv"
DEFAULT_OUT = PROJECT_ROOT / "Report" / "Figures"

# Per-config colour palette (tab10-derived, C1-C8)
_PAL = dict(
    zip(
        [f"C{i}" for i in range(1, 9)],
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ],
    )
)

# Error categories tracked per iteration in the combined CSV
ERR_CATS = [
    "err_syntax_structure",
    "err_token_recognition",
    "err_mismatched_input",
    "err_extraneous_input",
    "err_missing_token",
    "err_semantic_agent",
    "err_semantic_resource",
    "err_semantic_target",
    "err_semantic_dup_agent",
    "err_semantic_ordering",
    "err_other",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wilson(p: float, n: int, z: float = 1.96):
    """Wilson score 95 % confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 0.0
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return max(0.0, c - m), min(1.0, c + m)


def _csort(label):
    """Sort key so C1 < C2 < ... < C10."""
    m = re.fullmatch(r"c(\d+)", str(label).strip(), re.I)
    return (0, int(m.group(1))) if m else (1, str(label))


def _short_model(name: str) -> str:
    """gemini-2.5-flash -> 2.5-flash"""
    return str(name).replace("gemini-", "")


def _short_cat(col: str) -> str:
    """err_semantic_agent -> sem:agent"""
    return col.replace("err_", "").replace("semantic_", "sem:")


def _style():
    if _HAS_SNS:
        sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def _save(fig, out_dir: Path, name: str):
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK   {path.name}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load(csv_path: Path):
    """Load the combined iteration-level CSV.

    Returns
    -------
    iter_df : DataFrame   – one row per (run, iteration)
    run_df  : DataFrame   – one row per run  (first iteration, derived_* cols)
    last_df : DataFrame   – one row per run  (last iteration, for error analysis)
    """
    df = pd.read_csv(csv_path)

    # Coerce numerics
    for c in [
        "iteration",
        "compiler_error_score",
        "derived_success",
        "derived_first_success_iteration",
        "derived_run_duration_seconds",
        "derived_initial_error_score",
        "derived_final_error_score",
        "derived_best_error_score",
        "derived_iteration_count",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ERR_CATS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["config_label"] = df["config_id"].str.upper()

    # Derive 'few_shots' (Yes/No) if not already present (backward compat)
    if "few_shots" not in df.columns:
        if "jshots" in df.columns:
            df["few_shots"] = df["jshots"].map({"yes": "Yes", "no": "No"}).fillna("No")
        elif "repair_shots" in df.columns:
            df["few_shots"] = df["repair_shots"].apply(
                lambda v: "Yes" if pd.to_numeric(v, errors="coerce") > 0 else "No"
            )
    else:
        df["few_shots"] = df["few_shots"].astype(str).str.strip()

    # Run-level: first iteration row (derived_* are broadcast on every row)
    run_df = (
        df.sort_values(["config_id", "run_id", "iteration"])
        .drop_duplicates(subset=["run_id"], keep="first")
        .copy()
    )
    run_df["success_bool"] = run_df["status"].astype(str).eq("success")

    # Last-iteration view (for final error-category analysis)
    last_df = (
        df.sort_values(["config_id", "run_id", "iteration"])
        .drop_duplicates(subset=["run_id"], keep="last")
        .copy()
    )
    last_df["success_bool"] = last_df["status"].astype(str).eq("success")

    return df, run_df, last_df


# ---------------------------------------------------------------------------
# Figure 1 – Success Rate per Configuration (95 % Wilson CI)
# ---------------------------------------------------------------------------


def fig01_success_rate_ci(run_df: pd.DataFrame, out_dir: Path):
    g = (
        run_df.groupby("config_label", as_index=False)
        .agg(n=("run_id", "count"), succ=("success_bool", "sum"))
    )
    g = g.sort_values("config_label", key=lambda s: s.map(_csort))
    g["rate"] = g["succ"] / g["n"]
    ci = g.apply(lambda r: _wilson(r["rate"], r["n"]), axis=1)
    g["lo"], g["hi"] = zip(*ci)

    x = np.arange(len(g))
    y = g["rate"].values
    cols = [_PAL.get(l, "#aaa") for l in g["config_label"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, y, color=cols, edgecolor="black", linewidth=0.4)
    yerr_lo = np.clip(y - g["lo"].values, 0, None)
    yerr_hi = np.clip(g["hi"].values - y, 0, None)
    ax.errorbar(
        x, y,
        yerr=[yerr_lo, yerr_hi],
        fmt="none", capsize=5, color="black", linewidth=1.2,
    )
    for b, v in zip(bars, y):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.03,
            f"{v:.0%}", ha="center", fontsize=10, fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(g["config_label"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Configuration")
    ax.set_title("Fig. 1 \u2014 Success Rate per Configuration (95 % Wilson CI)")
    _save(fig, out_dir, "fig01_success_rate_ci")


# ---------------------------------------------------------------------------
# Figure 2 – Main-Effect Forest Plot
# ---------------------------------------------------------------------------


def fig02_main_effect_forest(run_df: pd.DataFrame, out_dir: Path):
    factors = [
        ("model", "Model"),
        ("cfg_system_prompt", "Repair Prompt"),
        ("few_shots", "Few-Shot Examples"),
    ]
    rows = []
    for col, label in factors:
        if col not in run_df.columns:
            continue
        for level in sorted(run_df[col].dropna().unique(), key=str):
            grp = run_df[run_df[col] == level]
            n = len(grp)
            p = grp["success_bool"].mean()
            lo, hi = _wilson(p, n)
            display = _short_model(level) if col == "model" else str(level)
            rows.append(
                dict(factor=label, level=display, rate=p, lo=lo, hi=hi, n=n)
            )

    ef = pd.DataFrame(rows)
    overall = run_df["success_bool"].mean()
    factor_cols = {
        "Model": "#1f77b4",
        "Repair Prompt": "#ff7f0e",
        "Few-Shot Examples": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(ef) + 1.2))
    y_pos = np.arange(len(ef))[::-1]

    for i, (_, r) in enumerate(ef.iterrows()):
        c = factor_cols.get(r["factor"], "grey")
        ax.errorbar(
            r["rate"], y_pos[i],
            xerr=[[max(0, r["rate"] - r["lo"])], [max(0, r["hi"] - r["rate"])]],
            fmt="o", color=c, capsize=4, markersize=8, linewidth=1.5,
        )
        ax.annotate(
            f'{r["rate"]:.0%}', (r["rate"], y_pos[i]),
            textcoords="offset points", xytext=(14, 0), fontsize=9, va="center",
        )

    ax.axvline(
        overall, ls="--", color="grey", alpha=0.5,
        label=f"Overall ({overall:.0%})",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f'{r["factor"]}: {r["level"]}' for _, r in ef.iterrows()], fontsize=10
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Success Rate")
    ax.set_title("Fig. 2 \u2014 Main-Effect Forest Plot")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out_dir, "fig02_main_effect_forest")


# ---------------------------------------------------------------------------
# Figure 3 – Factor Interaction Plot (2-panel)
# ---------------------------------------------------------------------------


def fig03_factor_interaction(run_df: pd.DataFrame, out_dir: Path):
    needed = {"model", "cfg_system_prompt", "few_shots", "success_bool"}
    if not needed.issubset(run_df.columns):
        return

    sps = sorted(run_df["cfg_system_prompt"].dropna().unique())
    models = sorted(run_df["model"].dropna().unique())
    js_levels = sorted(run_df["few_shots"].dropna().unique(), key=str)

    fig, axes = plt.subplots(1, len(sps), figsize=(6 * len(sps), 5), sharey=True)
    if len(sps) == 1:
        axes = [axes]

    styles = {
        "no": ("--", "s"),
        "yes": ("-", "o"),
        "0": ("--", "s"),
        "3": ("-", "o"),
    }

    for ax, sp in zip(axes, sps):
        sub = run_df[run_df["cfg_system_prompt"] == sp]
        for js in js_levels:
            rates = []
            for mdl in models:
                grp = sub[(sub["model"] == mdl) & (sub["few_shots"] == js)]
                rates.append(grp["success_bool"].mean() if len(grp) else np.nan)
            ls, mk = styles.get(str(js), ("-", "o"))
            ax.plot(
                range(len(models)), rates,
                marker=mk, linestyle=ls, linewidth=2, markersize=8,
                label=f"Few-Shot: {js}",
            )
            for xi, r in enumerate(rates):
                if not np.isnan(r):
                    ax.annotate(
                        f"{r:.0%}", (xi, r),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9,
                    )
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([_short_model(m) for m in models], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(sp, fontsize=11)
        ax.set_xlabel("Model")
        if ax is axes[0]:
            ax.set_ylabel("Success Rate")
            ax.legend(fontsize=9)

    fig.suptitle(
        "Fig. 3 \u2014 Factor Interaction: Model \u00d7 Few-Shot Examples \u00d7 Repair Prompt",
        fontsize=13, y=1.02,
    )
    _save(fig, out_dir, "fig03_factor_interaction")


# ---------------------------------------------------------------------------
# Figure 4 – Prompt x Scenario Success-Rate Heatmap
# ---------------------------------------------------------------------------


def fig04_prompt_scenario_heatmap(run_df: pd.DataFrame, out_dir: Path):
    if not {"system_prompt", "scenario", "success_bool"}.issubset(run_df.columns):
        return

    pivot = (
        run_df.pivot_table(
            index="system_prompt", columns="scenario",
            values="success_bool", aggfunc="mean",
        )
        .sort_index()
    )
    # Shorten column labels for readability
    pivot.columns = [c.replace("Scenario_", "S").replace(".txt", "") for c in pivot.columns]
    pivot.index = [i.replace(".txt", "") for i in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    if _HAS_SNS:
        sns.heatmap(
            pivot, annot=True, fmt=".0%", cmap="RdYlGn",
            vmin=0, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Success Rate"},
        )
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(
                    j, i, f"{pivot.values[i, j]:.0%}",
                    ha="center", va="center", fontsize=10,
                )
        plt.colorbar(im, ax=ax, label="Success Rate")

    ax.set_title("Fig. 4 \u2014 Prompt \u00d7 Scenario Success Rate")
    ax.set_ylabel("Generative System Prompt")
    ax.set_xlabel("Scenario")
    _save(fig, out_dir, "fig04_prompt_scenario_heatmap")


# ---------------------------------------------------------------------------
# Figure 5 – Iterations-to-Success (Box + Strip)
# ---------------------------------------------------------------------------


def fig05_iterations_box_strip(run_df: pd.DataFrame, out_dir: Path):
    """Box-plot of first-success iteration per config, overlaid with
    individual data points (strip / swarm).  Median is marked with a
    line; the annotation shows n_success / n_total.
    """
    col = "derived_first_success_iteration"
    if col not in run_df.columns:
        return

    cfgs = sorted(run_df["config_label"].dropna().unique(), key=_csort)
    # Only include Pro-model configs (C5–C8) — Flash configs have too few
    # successes for a meaningful iteration distribution.
    cfgs = [c for c in cfgs if c in ("C5", "C6", "C7", "C8")]
    succ = run_df[run_df["success_bool"]].copy()
    if succ.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Collect data & colours per config
    data, colours, labels = [], [], []
    for cfg in cfgs:
        s = succ.loc[succ["config_label"] == cfg, col].dropna()
        total = int((run_df["config_label"] == cfg).sum())
        if s.empty:
            continue
        data.append(s.values)
        colours.append(_PAL.get(cfg, "#888888"))
        labels.append(f"{cfg}\n({len(s)}/{total})")

    positions = np.arange(len(data))

    # Box-plots (no outlier markers — the strip shows them)
    bp = ax.boxplot(
        data, positions=positions, widths=0.45,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    # Strip (jittered dots)
    rng = np.random.default_rng(42)
    for i, (vals, c) in enumerate(zip(data, colours)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            positions[i] + jitter, vals,
            color=c, edgecolors="black", linewidths=0.4,
            s=30, alpha=0.75, zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Iteration of First Success")
    ax.set_xlabel("Configuration (success / total)")
    ax.set_title("Fig. 5 \u2014 Iterations to Success (Pro-model Configs)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(0.5, run_df[col].max() + 0.5 if run_df[col].notna().any() else 10.5)
    _save(fig, out_dir, "fig05_iterations_box_strip")


# ---------------------------------------------------------------------------
# Figure 6 – Error-Score Convergence by Configuration
# ---------------------------------------------------------------------------


def fig06_error_convergence(iter_df: pd.DataFrame, out_dir: Path):
    """Median compiler error score at each repair iteration, per config.

    Note: sample size decreases at later iterations because successful
    runs stop early.  The curves therefore reflect the trajectory of
    the *remaining* (harder) runs.
    """
    if "compiler_error_score" not in iter_df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for cfg in sorted(iter_df["config_label"].dropna().unique(), key=_csort):
        sub = iter_df[iter_df["config_label"] == cfg]
        med = sub.groupby("iteration")["compiler_error_score"].median()
        ax.plot(
            med.index, med.values, marker=".", linewidth=1.8,
            label=cfg, color=_PAL.get(cfg),
        )

    ax.set_xlabel("Repair Iteration")
    ax.set_ylabel("Median Compiler Error Score")
    ax.set_title("Fig. 6 \u2014 Error-Score Convergence by Configuration")
    ax.legend(title="Config", fontsize=8, title_fontsize=9, ncol=2)
    ax.set_xticks(range(0, 11))
    _save(fig, out_dir, "fig06_error_convergence")


# ---------------------------------------------------------------------------
# Figure 7 – Error-Category Prevalence Heatmap (failed runs)
# ---------------------------------------------------------------------------


def fig07_error_categories(last_df: pd.DataFrame, out_dir: Path):
    """Heatmap showing what fraction of failed runs exhibit each error
    category in their *final* iteration, broken down by config."""
    failed = last_df[last_df["status"].astype(str).ne("success")].copy()
    if failed.empty:
        return
    present = [c for c in ERR_CATS if c in failed.columns]
    if not present:
        return

    cfgs = sorted(failed["config_label"].dropna().unique(), key=_csort)
    rows, labels = [], []
    for cfg in list(cfgs) + ["ALL"]:
        sub = failed if cfg == "ALL" else failed[failed["config_label"] == cfg]
        n = len(sub)
        if n == 0:
            continue
        rows.append({_short_cat(c): (sub[c] > 0).sum() / n for c in present})
        labels.append(f"{cfg} (n={n})")

    mat = pd.DataFrame(rows, index=labels)

    fig, ax = plt.subplots(figsize=(13, 0.6 * len(mat) + 2))
    if _HAS_SNS:
        sns.heatmap(
            mat, annot=True, fmt=".0%", cmap="YlOrRd",
            vmin=0, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Prevalence"},
        )
    else:
        im = ax.imshow(mat.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j, i, f"{mat.values[i, j]:.0%}",
                    ha="center", va="center", fontsize=9,
                )
        plt.colorbar(im, ax=ax, label="Prevalence")

    ax.set_title("Fig. 7 \u2014 Error-Category Prevalence among Failed Runs")
    ax.set_xlabel("Error Category")
    ax.set_ylabel("Configuration")
    _save(fig, out_dir, "fig07_error_categories")


# ---------------------------------------------------------------------------
# Figure 8 – Scenario Difficulty Profile
# ---------------------------------------------------------------------------


def fig08_scenario_difficulty(run_df: pd.DataFrame, out_dir: Path):
    """Three-panel bar chart profiling each scenario's difficulty."""
    if "scenario" not in run_df.columns:
        return

    scenarios = sorted(run_df["scenario"].dropna().unique())
    sc_short = [s.replace("Scenario_", "S").replace(".txt", "") for s in scenarios]
    x = np.arange(len(scenarios))

    fail_rate, mean_err, mean_init_err = [], [], []
    for sc in scenarios:
        sub = run_df[run_df["scenario"] == sc]
        fail_rate.append(1 - sub["success_bool"].mean())
        fs = sub[~sub["success_bool"]]
        mean_err.append(fs["derived_final_error_score"].mean() if len(fs) else 0)
        mean_init_err.append(sub["derived_initial_error_score"].mean() if len(sub) else 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A – Failure rate
    axes[0].bar(x, fail_rate, color="#d62728", edgecolor="black", linewidth=0.4)
    for xi, v in zip(x, fail_rate):
        axes[0].text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    axes[0].set_title("Failure Rate")
    axes[0].set_ylim(0, max(fail_rate) * 1.2 if fail_rate else 1)
    axes[0].set_ylabel("Rate")

    # Panel B – Mean final error score (failed runs)
    axes[1].bar(x, mean_err, color="#ff7f0e", edgecolor="black", linewidth=0.4)
    for xi, v in zip(x, mean_err):
        axes[1].text(xi, v + 10, f"{v:.0f}", ha="center", fontsize=9)
    axes[1].set_title("Mean Final Error Score (failed)")
    axes[1].set_ylabel("Error Score")

    # Panel C – Mean initial error score (all runs)
    axes[2].bar(x, mean_init_err, color="#1f77b4", edgecolor="black", linewidth=0.4)
    for xi, v in zip(x, mean_init_err):
        axes[2].text(xi, v + 10, f"{v:.0f}", ha="center", fontsize=9)
    axes[2].set_title("Mean Initial Error Score (all runs)")
    axes[2].set_ylabel("Error Score")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(sc_short, fontsize=10)
        ax.set_xlabel("Scenario")

    fig.suptitle("Fig. 8 \u2014 Scenario Difficulty Profile", fontsize=13, y=1.02)
    _save(fig, out_dir, "fig08_scenario_difficulty")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate publication figures for the cross-comparative study."
    )
    ap.add_argument(
        "--csv", default=str(DEFAULT_CSV),
        help="Path to combined_run_histories.csv",
    )
    ap.add_argument(
        "--outdir", default=str(DEFAULT_OUT),
        help="Output directory for PNG figures",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    iter_df, run_df, last_df = _load(csv_path)
    print(
        f"Loaded {len(iter_df):,} iterations, {len(run_df):,} runs, "
        f"{run_df['config_label'].nunique()} configs\n"
    )

    fig01_success_rate_ci(run_df, out_dir)
    fig02_main_effect_forest(run_df, out_dir)
    fig03_factor_interaction(run_df, out_dir)
    fig04_prompt_scenario_heatmap(run_df, out_dir)
    fig05_iterations_box_strip(run_df, out_dir)
    fig06_error_convergence(iter_df, out_dir)
    fig07_error_categories(last_df, out_dir)
    fig08_scenario_difficulty(run_df, out_dir)

    print(f"\nExported 8 figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
