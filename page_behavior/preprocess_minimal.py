import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TIME_WINDOW_PATTERN_LEN = 11  # '00:00-00:10'


def find_time_window_columns(columns: List[str]) -> List[str]:
    time_cols: List[str] = []
    for col in columns:
        if isinstance(col, str) and len(col) == TIME_WINDOW_PATTERN_LEN and col.count(":") == 2 and col.count("-") == 1:
            time_cols.append(col)
    return time_cols


def is_cumulative_series(values: np.ndarray) -> bool:
    series = values.astype(float).copy()
    for i in range(series.shape[0]):
        if i == 0:
            if np.isnan(series[i]):
                series[i] = 0.0
        else:
            if np.isnan(series[i]):
                series[i] = series[i - 1]
    diffs = np.diff(series, prepend=series[0])
    non_decreasing_ratio = float(np.mean(diffs >= -1e-9))
    large_jumps = int(np.sum(diffs > 20.0))
    idx = np.arange(series.shape[0])
    try:
        rho, _ = spearmanr(idx, series)
    except Exception:
        rho = np.nan
    median_value = float(np.median(series)) if series.size > 0 else 0.0
    last_value = float(series[-1]) if series.size > 0 else 0.0
    last_to_median = float(last_value / (median_value + 1e-9)) if (median_value > 0) else (float('inf') if last_value > 0 else 1.0)
    return (non_decreasing_ratio >= 0.95) and ( (not np.isnan(rho) and rho >= 0.9) or (large_jumps >= 1) or (last_to_median > 10.0) )


def diff_with_resets(values: np.ndarray) -> Tuple[np.ndarray, int]:
    diffs = np.zeros_like(values, dtype=float)
    reset_count = 0
    for i in range(values.shape[0]):
        if i == 0:
            diffs[i] = max(values[i], 0.0)
        else:
            delta = values[i] - values[i - 1]
            if delta < 0:
                reset_count += 1
                diffs[i] = max(values[i], 0.0)
            else:
                diffs[i] = delta
    diffs[diffs < 0] = 0.0
    return diffs, reset_count


def main():
    parser = argparse.ArgumentParser(description="Minimal preprocess: read raw CSV, fix cumulative rows, output single summary CSV.")
    parser.add_argument("--input", type=str, default=str(Path(__file__).resolve().parents[0] / "all_groups_page_behavior_by_10s.csv"))
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parents[0] / "page_behavior_preprocessed.csv"))
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)

    # Skip first title row; header starts on line 2
    df = pd.read_csv(input_csv, skiprows=1)

    required = ["Group", "speaker", "Type"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    time_cols = find_time_window_columns(list(df.columns))
    if not time_cols:
        raise ValueError("No time window columns detected.")

    # sort time columns by start time
    def parse_start_seconds(col: str) -> int:
        start = col.split("-")[0]
        mm, ss = start.split(":")
        return int(mm) * 60 + int(ss)

    time_cols = sorted(time_cols, key=parse_start_seconds)

    df_time = df[time_cols].apply(pd.to_numeric, errors="coerce")

    # Forward-filled view for detection
    df_ffill = df_time.ffill(axis=1).fillna(0.0)

    processed = np.zeros_like(df_ffill.values, dtype=float)

    for i in range(df_ffill.shape[0]):
        row_series = df_ffill.iloc[i].values.astype(float)
        if is_cumulative_series(row_series):
            diffs, _ = diff_with_resets(row_series)
            values_row = diffs
        else:
            orig = df_time.iloc[i].values.astype(float)
            np.nan_to_num(orig, copy=False, nan=0.0)
            orig[orig < 0] = 0.0
            values_row = orig
        processed[i, :] = values_row

    df_processed = pd.DataFrame(processed, columns=time_cols)
    sum_count = df_processed.sum(axis=1)
    op_freq = sum_count / 30.0

    df_out = pd.DataFrame({
        "Group": df["Group"],
        "speaker": df["speaker"],
        "Type": df["Type"].astype(str).str.strip(),
        "SumCount": sum_count,
        "Op_Freq": op_freq,
    })

    # Keep only valid types if present
    valid = df_out["Type"].isin(["AI", "NoAI"])
    df_out = df_out.loc[valid].reset_index(drop=True)

    df_out.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()


