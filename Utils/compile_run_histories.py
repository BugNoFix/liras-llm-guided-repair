#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def _config_from_filename(path: Path) -> str:
    return path.stem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine multiple run_history CSV files (one per pipeline config) into a comparative dataset"
    )
    parser.add_argument(
        "--input-glob",
        default="Report/Histories/c*.csv",
        help="Glob for input run_history files",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest CSV from run_all_configurations.py (uses history_csv column)",
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
        df["source_config"] = _config_from_filename(p)
        chunks.append(df)

    if not chunks:
        raise ValueError("All matched files were empty")

    combined = pd.concat(chunks, ignore_index=True)

    run_key_col = "run_key" if "run_key" in combined.columns else "run_id"
    if run_key_col in combined.columns:
        combined["run_uid"] = (
            combined["source_history_file"].astype(str)
            + "::"
            + combined[run_key_col].astype(str)
        )

    out_path = Path(args.outcsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"Combined files: {len(paths)}")
    print(f"Rows written: {len(combined)}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
