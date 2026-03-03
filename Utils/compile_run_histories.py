#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS_CSV = PROJECT_ROOT / "Report" / "configs.csv"


def _config_from_filename(path: Path) -> str:
    return path.stem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine multiple run_history CSV files (one per pipeline config) into a comparative dataset"
    )
    parser.add_argument(
        "--input-glob",
        default="Report/Histories/c[0-9]*.csv",
        help="Glob for input run_history files (default matches c1.csv through c99.csv)",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest CSV from run_all_configurations.py (uses history_csv column)",
    )
    parser.add_argument(
        "--configs",
        default=str(DEFAULT_CONFIGS_CSV),
        help="Path to configs.csv mapping config_id -> model, system_prompt, jshots, repair_shots",
    )
    parser.add_argument(
        "--outcsv",
        default="Report/Histories/combined_run_histories.csv",
        help="Output combined CSV path",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'pandas'. Install with: pip install pandas matplotlib seaborn"
        ) from exc

    paths: list[Path]
    manifest_path = str(args.manifest or "").strip()
    if manifest_path:
        mpath = Path(manifest_path)
        if not mpath.exists():
            raise FileNotFoundError(f"Manifest not found: {mpath}")
        manifest_df = pd.read_csv(mpath)
        if "history_csv" not in manifest_df.columns:
            raise ValueError("Manifest missing required column: history_csv")
        paths = [Path(p) for p in manifest_df["history_csv"].dropna().astype(str).tolist()]
    else:
        paths = sorted(Path().glob(args.input_glob))
    if not paths:
        if manifest_path:
            raise FileNotFoundError(f"No history_csv entries found in manifest: {manifest_path}")
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    chunks = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["source_history_file"] = str(p)
        # Use config_id column if present; otherwise infer from filename
        if "config_id" not in df.columns or df["config_id"].isna().all():
            df["config_id"] = _config_from_filename(p)
        chunks.append(df)

    if not chunks:
        raise ValueError("All matched files were empty")

    combined = pd.concat(chunks, ignore_index=True)

    # --------------- Enrich with config parameters from configs.csv ---------------
    configs_path = Path(str(args.configs).strip()) if args.configs else None
    if configs_path and configs_path.exists():
        configs_df = pd.read_csv(configs_path)
        if "config_id" in configs_df.columns:
            # Merge config parameters (model, system_prompt, jshots, repair_shots, etc.)
            param_cols = [c for c in configs_df.columns if c != "config_id"]
            # Avoid overwriting existing columns — prefix with cfg_ if collision
            rename_map = {}
            for c in param_cols:
                if c in combined.columns:
                    rename_map[c] = f"cfg_{c}"
            if rename_map:
                configs_df = configs_df.rename(columns=rename_map)
            combined = combined.merge(configs_df, on="config_id", how="left")
            # Derive unified 'few_shots' column (Yes/No) from jshots or repair_shots
            if "jshots" in combined.columns:
                combined["few_shots"] = combined["jshots"].map({"yes": "Yes", "no": "No"}).fillna("No")
            elif "repair_shots" in combined.columns:
                combined["few_shots"] = combined["repair_shots"].apply(lambda v: "Yes" if int(v) > 0 else "No")
            print(f"Enriched with config parameters from: {configs_path}")
        else:
            print(f"WARN: {configs_path} has no 'config_id' column — skipping enrichment")
    else:
        if configs_path:
            print(f"WARN: configs file not found ({configs_path}) — skipping enrichment")

    # --------------- Build unique run identifier ---------------
    run_key_col = "run_key" if "run_key" in combined.columns else "run_id"
    if run_key_col in combined.columns:
        combined["run_uid"] = (
            combined["config_id"].astype(str)
            + "::"
            + combined[run_key_col].astype(str)
        )

    out_path = Path(args.outcsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"Combined files: {len(paths)}")
    print(f"Rows written: {len(combined)}")
    print(f"Unique configs: {combined['config_id'].nunique()}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
