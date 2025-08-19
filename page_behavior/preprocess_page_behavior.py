import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TIME_WINDOW_PATTERN_LEN = 11  # e.g., '00:00-00:10'


@dataclass
class PreprocessConfig:
    cumulative_non_decreasing_ratio_threshold: float = 0.95
    cumulative_spearman_threshold: float = 0.9
    cumulative_large_jump_threshold: float = 20.0
    low_coverage_windows_threshold: int = 90  # 15 minutes of 10s windows
    outlier_single_window_threshold: float = 99.0
    treat_missing_as_zero: bool = True  # If False, will keep NaN and use valid windows


def find_time_window_columns(columns: List[str]) -> List[str]:
    time_cols: List[str] = []
    for col in columns:
        # Simple heuristic: time window columns look like 'mm:ss-mm:ss' length 11 with two dashes and two colons
        if isinstance(col, str) and len(col) == TIME_WINDOW_PATTERN_LEN and col.count(":") == 2 and col.count("-") == 1:
            time_cols.append(col)
    return time_cols


def is_cumulative_series(values: np.ndarray, cfg: PreprocessConfig) -> Tuple[bool, Dict[str, float]]:
    # Use a copy and forward-fill NaNs for monotonicity checks
    series = values.astype(float).copy()
    # Forward fill NaN using previous known value; start NaNs become 0
    for i in range(series.shape[0]):
        if i == 0:
            if np.isnan(series[i]):
                series[i] = 0.0
        else:
            if np.isnan(series[i]):
                series[i] = series[i - 1]

    diffs = np.diff(series, prepend=series[0])
    non_decreasing_ratio = float(np.mean(diffs >= -1e-9))  # small tolerance
    # Large jumps count
    large_jumps = int(np.sum(diffs > cfg.cumulative_large_jump_threshold))

    # Spearman correlation with time index
    idx = np.arange(series.shape[0])
    try:
        rho, _ = spearmanr(idx, series)
    except Exception:
        rho = np.nan

    # Heuristic: last value vs median window value
    median_value = float(np.median(series)) if series.size > 0 else 0.0
    last_value = float(series[-1]) if series.size > 0 else 0.0
    last_to_median = float(last_value / (median_value + 1e-9)) if (median_value > 0) else (float('inf') if last_value > 0 else 1.0)

    is_cumulative = (
        (non_decreasing_ratio >= cfg.cumulative_non_decreasing_ratio_threshold)
        and (
            (not np.isnan(rho) and rho >= cfg.cumulative_spearman_threshold)
            or (large_jumps >= 1)
            or (last_to_median > 10.0)
        )
    )

    details = {
        "non_decreasing_ratio": non_decreasing_ratio,
        "rho_spearman": float(rho) if not np.isnan(rho) else np.nan,
        "large_jumps": float(large_jumps),
        "last_to_median": last_to_median,
    }
    return is_cumulative, details


def diff_with_resets(values: np.ndarray) -> Tuple[np.ndarray, int]:
    # values assumed forward-filled already (no NaN), cumulative-like
    diffs = np.zeros_like(values, dtype=float)
    reset_count = 0
    for i in range(values.shape[0]):
        if i == 0:
            diffs[i] = max(values[i], 0.0)
        else:
            delta = values[i] - values[i - 1]
            if delta < 0:
                # reset point
                reset_count += 1
                diffs[i] = max(values[i], 0.0)
            else:
                diffs[i] = delta
    diffs[diffs < 0] = 0.0
    return diffs, reset_count


def preprocess(input_csv: Path, output_dir: Path, cfg: PreprocessConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip the first line (title/comment), header is at line 2
    df = pd.read_csv(input_csv, skiprows=1)

    # Basic column normalization
    expected_id_cols = ["Group", "speaker", "Type"]
    missing = [c for c in expected_id_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    time_cols = find_time_window_columns(list(df.columns))
    if not time_cols:
        raise ValueError("No time window columns detected.")

    # Ensure time columns are ordered chronologically by parsing start time (first mm:ss)
    def parse_start_seconds(col: str) -> int:
        start = col.split("-")[0]
        mm, ss = start.split(":")
        return int(mm) * 60 + int(ss)

    time_cols = sorted(time_cols, key=parse_start_seconds)

    # Coerce to numeric
    df_time = df[time_cols].apply(pd.to_numeric, errors="coerce")

    # Validate Type values
    df["Type"] = df["Type"].astype(str).str.strip()
    valid_mask = df["Type"].isin(["AI", "NoAI"])
    if (~valid_mask).any():
        # Keep only valid rows
        df = df.loc[valid_mask].reset_index(drop=True)
        df_time = df_time.loc[valid_mask].reset_index(drop=True)

    # Compute valid windows before filling
    valid_windows = df_time.notna().sum(axis=1).astype(int)
    valid_minutes = (valid_windows / 6.0).round(3)

    n_missing_filled = df_time.isna().sum(axis=1).astype(int)

    # For detection, create a forward-filled view
    df_ffill = df_time.copy()
    df_ffill = df_ffill.ffill(axis=1).fillna(0.0)

    # Initialize outputs
    processed_values = np.zeros_like(df_ffill.values, dtype=float)
    flag_cumulative_fixed = np.zeros(df_ffill.shape[0], dtype=bool)
    n_resets_arr = np.zeros(df_ffill.shape[0], dtype=int)
    flag_outlier = np.zeros(df_ffill.shape[0], dtype=bool)

    cumulative_detect_summaries: List[Dict[str, float]] = []

    # Row-wise processing
    for i in range(df_ffill.shape[0]):
        row_series = df_ffill.iloc[i].values.astype(float)
        is_cum, details = is_cumulative_series(row_series, cfg)
        cumulative_detect_summaries.append(details)

        if is_cum:
            diffs, reset_count = diff_with_resets(row_series)
            values_row = diffs
            flag_cumulative_fixed[i] = True
            n_resets_arr[i] = int(reset_count)
        else:
            # Treat as per-window counts
            # Use original numeric (coerced) with NaN -> 0 if configured
            orig = df_time.iloc[i].values.astype(float)
            if cfg.treat_missing_as_zero:
                np.nan_to_num(orig, copy=False, nan=0.0)
            else:
                # Keep NaN; for processed_values, replace NaN with 0 but coverage tracked via valid_windows
                orig = np.where(np.isnan(orig), 0.0, orig)
            # Clamp negatives to 0 just in case
            orig[orig < 0] = 0.0
            values_row = orig

        processed_values[i, :] = values_row
        if np.nanmax(values_row) > cfg.outlier_single_window_threshold:
            flag_outlier[i] = True

    # Build processed DataFrame
    df_processed = pd.DataFrame(processed_values, columns=time_cols)

    # Flags
    flag_all_zero = (df_processed.sum(axis=1) <= 1e-9)
    flag_low_coverage = valid_windows < cfg.low_coverage_windows_threshold

    # Summaries
    sum_count = df_processed.sum(axis=1)

    # Attach identifiers and flags
    df_out = pd.concat(
        [
            df[expected_id_cols].reset_index(drop=True),
            df_processed,
            pd.DataFrame(
                {
                    "flag_cumulative_fixed": flag_cumulative_fixed,
                    "n_resets": n_resets_arr,
                    "flag_all_zero": flag_all_zero,
                    "flag_low_coverage": flag_low_coverage,
                    "flag_outlier": flag_outlier,
                    "n_missing_filled": n_missing_filled,
                    "valid_windows": valid_windows,
                    "valid_minutes": valid_minutes,
                }
            ),
        ],
        axis=1,
    )

    # Summary (row-level)
    df_summary = pd.DataFrame(
        {
            "Group": df["Group"].values,
            "speaker": df["speaker"].values,
            "Type": df["Type"].values,
            "SumCount": sum_count.values,
            "valid_windows": valid_windows.values,
            "valid_minutes": valid_minutes.values,
            "flag_cumulative_fixed": flag_cumulative_fixed,
            "n_resets": n_resets_arr,
            "flag_all_zero": flag_all_zero.values,
            "flag_low_coverage": flag_low_coverage.values,
            "flag_outlier": flag_outlier,
            "n_missing_filled": n_missing_filled.values,
        }
    )

    # Write outputs
    out_preprocessed = output_dir / "page_behavior_by_10s_preprocessed.csv"
    out_summary = output_dir / "page_behavior_summary.csv"
    out_log = output_dir / "preprocess_log.json"

    df_out.to_csv(out_preprocessed, index=False)
    df_summary.to_csv(out_summary, index=False)

    # QC log
    qc = {
        "config": asdict(cfg),
        "input_file": str(input_csv),
        "n_rows": int(df.shape[0]),
        "n_time_windows": int(len(time_cols)),
        "n_cumulative_flagged": int(flag_cumulative_fixed.sum()),
        "n_low_coverage": int(flag_low_coverage.sum()),
        "n_all_zero": int(flag_all_zero.sum()),
        "n_outlier_rows": int(flag_outlier.sum()),
        "valid_windows_stats": {
            "min": int(valid_windows.min()) if len(valid_windows) else 0,
            "p50": float(np.percentile(valid_windows, 50)) if len(valid_windows) else 0.0,
            "p95": float(np.percentile(valid_windows, 95)) if len(valid_windows) else 0.0,
            "max": int(valid_windows.max()) if len(valid_windows) else 0,
        },
        "sumcount_stats": {
            "min": float(sum_count.min()) if len(sum_count) else 0.0,
            "p50": float(np.percentile(sum_count, 50)) if len(sum_count) else 0.0,
            "p95": float(np.percentile(sum_count, 95)) if len(sum_count) else 0.0,
            "max": float(sum_count.max()) if len(sum_count) else 0.0,
        },
        "cumulative_detection_examples": cumulative_detect_summaries[:5],
    }

    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_preprocessed}")
    print(f"Wrote: {out_summary}")
    print(f"Wrote: {out_log}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess page behavior logs by 10s windows.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(
            Path(__file__).resolve().parents[0] / "all_groups_page_behavior_by_10s.csv"
        ),
        help="Path to input CSV (raw).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[0]),
        help="Output directory for preprocessed CSVs and logs.",
    )
    args = parser.parse_args()

    cfg = PreprocessConfig()
    preprocess(Path(args.input), Path(args.outdir), cfg)


if __name__ == "__main__":
    main()


