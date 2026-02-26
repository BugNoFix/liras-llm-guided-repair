#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _config_from_filename(path: Path) -> str:
    stem = path.stem
    name = re.sub(r"^run_history[_-]?", "", stem, flags=re.IGNORECASE)
    name = name.strip("_-")
    return name or stem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine multiple run_history CSV files (one per pipeline config) into a comparative dataset"
    )
    parser.add_argument(
        "--input-glob",
        default="Report/Histories/run_history_*.csv",
        help="Glob for input run_history files",
    )
    parser.add_argument(
        "--outcsv",
        default="Report/combined_run_histories.csv",
        help="Output combined CSV path",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'pandas'. Install with: pip install pandas matplotlib seaborn"
        ) from exc

    paths = sorted(Path().glob(args.input_glob))
    if not paths:
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
