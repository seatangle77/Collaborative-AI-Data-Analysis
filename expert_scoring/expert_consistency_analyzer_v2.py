import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_dir() -> str:
    """Return the directory where this script resides."""
    return os.path.dirname(os.path.abspath(__file__))


def load_expert_csvs(
    e1_filename: str = "E1_scoring.csv",
    e2_filename: str = "E2_socring.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load expert CSVs from the same folder as this script.
    
    Notes on raw schema observed from the provided files:
    - E1 has columns: ['Group', '文本 13', 'Condition', C1..C6, O7, O8, 'Process Score', 'Outcome Score', 'Total Score']
      where 'Group' seems to be a row index (1..24) and '文本 13' holds the label like 'Group 0'.
    - E2 has columns: ['NO.', 'Group', 'Condition', C1..C6, O7, O8, 'Process Score', 'Outcome Score', 'Total Score']
      where 'NO.' seems to be a row index and 'Group' holds the label like 'Group 0'.
    """
    base_dir = get_project_dir()
    e1_path = os.path.join(base_dir, e1_filename)
    e2_path = os.path.join(base_dir, e2_filename)

    if not os.path.exists(e1_path):
        raise FileNotFoundError(f"E1 CSV not found: {e1_path}")
    if not os.path.exists(e2_path):
        raise FileNotFoundError(f"E2 CSV not found: {e2_path}")

    df_e1 = pd.read_csv(e1_path)
    df_e2 = pd.read_csv(e2_path)
    return df_e1, df_e2


def unify_schema(df_e1: pd.DataFrame, df_e2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Unify the schemas of two expert DataFrames for downstream analysis.
    
    - Standardize the group label to column 'GroupLabel'
    - Keep 'Condition' as-is
    - Preserve scoring columns (C1..C6, Process Score, Total Score) - EXCLUDING O7, O8, Outcome Score
    - Ensure numeric columns are numeric (coerce errors to NaN then fill with 0 if any)
    """
    df1 = df_e1.copy()
    df2 = df_e2.copy()

    # E1: drop the numeric row index, rename the text label to GroupLabel
    if '文本 13' in df1.columns:
        df1 = df1.rename(columns={'文本 13': 'GroupLabel'})
        if 'Group' in df1.columns:
            # This 'Group' in E1 appears to be 1..24; not needed for alignment
            df1 = df1.drop(columns=['Group'])
    else:
        # Fallback: if '文本 13' not present, try to use 'Group' as label
        if 'Group' in df1.columns:
            df1 = df1.rename(columns={'Group': 'GroupLabel'})

    # E2: drop NO., rename Group to GroupLabel
    if 'NO.' in df2.columns:
        df2 = df2.drop(columns=['NO.'])
    if 'Group' in df2.columns:
        df2 = df2.rename(columns={'Group': 'GroupLabel'})

    # Ensure 'Condition' exists and is string
    for dfx in (df1, df2):
        if 'Condition' not in dfx.columns:
            raise ValueError("Column 'Condition' is required in both CSVs.")
        dfx['Condition'] = dfx['Condition'].astype(str).str.strip()
        if 'GroupLabel' not in dfx.columns:
            raise ValueError("Column 'GroupLabel' is required after schema unification.")
        dfx['GroupLabel'] = dfx['GroupLabel'].astype(str).str.strip()

    # Identify scoring columns present in both - EXCLUDING O7, O8, Outcome Score, Total Score
    possible_score_cols = [
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
        'Process Score'
    ]
    common_cols = [c for c in possible_score_cols if c in df1.columns and c in df2.columns]

    # Coerce scoring columns to numeric where applicable
    for dfx in (df1, df2):
        for col in common_cols:
            dfx[col] = pd.to_numeric(dfx[col], errors='coerce')

    return df1, df2, common_cols


def describe_scores(df: pd.DataFrame, which: str, score_cols: List[str]) -> pd.DataFrame:
    """
    Return a descriptive statistics DataFrame for selected score columns.
    Includes overall stats and condition-wise grouped means/std where applicable.
    """
    # Overall description
    overall_desc = df[score_cols].describe().T
    overall_desc = overall_desc.loc[:, ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    overall_desc.insert(0, 'Expert', which)
    overall_desc.insert(1, 'Scope', 'Overall')

    # Condition-wise means and std
    by_cond_mean = df.groupby('Condition')[score_cols].mean(numeric_only=True)
    by_cond_std = df.groupby('Condition')[score_cols].std(ddof=1, numeric_only=True)

    rows = [overall_desc]
    for condition_value in by_cond_mean.index:
        tmp = pd.DataFrame(index=score_cols)
        tmp['Expert'] = which
        tmp['Scope'] = f"Condition={condition_value}"
        tmp['count'] = df[df['Condition'] == condition_value].shape[0]
        tmp['mean'] = by_cond_mean.loc[condition_value]
        tmp['std'] = by_cond_std.loc[condition_value]
        # min/percentiles/max per condition (computed robustly)
        cond_subset = df.loc[df['Condition'] == condition_value, score_cols]
        tmp['min'] = cond_subset.min(numeric_only=True)
        tmp['25%'] = cond_subset.quantile(0.25, numeric_only=True)
        tmp['50%'] = cond_subset.quantile(0.50, numeric_only=True)
        tmp['75%'] = cond_subset.quantile(0.75, numeric_only=True)
        tmp['max'] = cond_subset.max(numeric_only=True)
        rows.append(tmp)

    result = pd.concat(rows, axis=0)
    result.index.name = 'Score'
    return result.reset_index()


def align_experts(
    df_e1: pd.DataFrame,
    df_e2: pd.DataFrame,
    score_cols: List[str],
    on_keys: List[str] = None,
) -> pd.DataFrame:
    """
    Align E1 and E2 rows by keys and return a merged DataFrame containing
    paired columns per score, e.g., 'Process Score_E1' and 'Process Score_E2'.
    """
    if on_keys is None:
        on_keys = ['GroupLabel', 'Condition']

    left = df_e1[on_keys + score_cols].copy()
    right = df_e2[on_keys + score_cols].copy()
    left = left.add_suffix('_E1')
    # Keep keys without suffix for merge
    for k in on_keys:
        left.rename(columns={f"{k}_E1": k}, inplace=True)
    right = right.add_suffix('_E2')
    for k in on_keys:
        right.rename(columns={f"{k}_E2": k}, inplace=True)

    merged = pd.merge(left, right, on=on_keys, how='inner', validate='one_to_one')
    return merged


def _anova_ms_two_way_random(scores: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute ANOVA mean squares for two-way random effects with n subjects x k raters.
    scores: array of shape (n, k)
    Returns: MSR (rows/subjects), MSC (columns/raters), MSE (residual)
    """
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array [n_subjects x k_raters]")
    n, k = scores.shape
    if n < 2 or k < 2:
        raise ValueError("Need at least 2 subjects and 2 raters for ICC.")

    grand_mean = np.nanmean(scores)
    subject_means = np.nanmean(scores, axis=1, keepdims=True)
    rater_means = np.nanmean(scores, axis=0, keepdims=True)

    # Replace NaNs if any before sums of squares (should not happen after cleaning)
    scores_filled = np.where(np.isnan(scores), grand_mean, scores)

    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_raters = n * np.sum((rater_means - grand_mean) ** 2)
    ss_total = np.sum((scores_filled - grand_mean) ** 2)
    ss_residual = ss_total - ss_subjects - ss_raters

    msr = ss_subjects / (n - 1)
    msc = ss_raters / (k - 1)
    mse = ss_residual / ((n - 1) * (k - 1))
    return float(msr), float(msc), float(mse)


def icc_2k(scores: np.ndarray) -> float:
    """
    Compute ICC(2,k): Two-way random effects, absolute agreement, average measures.
    Formula (McGraw & Wong, 1996):
    ICC(2,k) = (MSR - MSE) / (MSR + (MSC - MSE) / n)
    """
    n, k = scores.shape
    msr, msc, mse = _anova_ms_two_way_random(scores)
    denominator = (msr + (msc - mse) / n)
    if denominator == 0:
        return np.nan
    return (msr - mse) / denominator


def icc_2k_with_ci(
    scores: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute ICC(2,k) and a bootstrap percentile CI by resampling subjects.
    Returns (icc, ci_low, ci_high).
    """
    rng = np.random.default_rng(random_state)
    # Drop rows with any NaNs to avoid complications in bootstrap
    mask_valid = ~np.any(np.isnan(scores), axis=1)
    scores_valid = scores[mask_valid]
    if scores_valid.shape[0] < 3:
        # Not enough subjects; return NaNs
        return np.nan, np.nan, np.nan

    point = icc_2k(scores_valid)

    n = scores_valid.shape[0]
    boots: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores = scores_valid[idx]
        boots.append(icc_2k(boot_scores))

    boots = np.array(boots)
    alpha = (1 - ci) / 2
    low = np.nanpercentile(boots, 100 * alpha)
    high = np.nanpercentile(boots, 100 * (1 - alpha))
    return float(point), float(low), float(high)


def compute_icc_summary(
    merged: pd.DataFrame,
    summary_cols: List[str],
    by_condition: bool = True,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    expert1_name: str = 'E1',
    expert2_name: str = 'E2',
) -> pd.DataFrame:
    """
    Compute ICC(2,k) for summary scores overall and by condition.
    Returns a DataFrame with columns: ['Score', 'Scope', 'N', 'ICC_2k', 'CI95_L', 'CI95_U'].
    """
    results: List[Dict[str, object]] = []

    # Overall
    for col in summary_cols:
        e1 = merged[f"{col}_E1"].to_numpy()
        e2 = merged[f"{col}_E2"].to_numpy()
        mat = np.vstack([e1, e2]).T
        icc, l, u = icc_2k_with_ci(mat, n_bootstrap=n_bootstrap, random_state=random_state)
        results.append({
            'Score': col,
            'Scope': 'Overall',
            'Expert1': expert1_name,
            'Expert2': expert2_name,
            'ExpertPair': f"{expert1_name}_vs_{expert2_name}",
            'N': int(np.sum(~np.any(np.isnan(mat), axis=1))),
            'ICC_2k': icc,
            'CI95_L': l,
            'CI95_U': u,
        })

    if by_condition and 'Condition' in merged.columns:
        for col in summary_cols:
            for cond, sub in merged.groupby('Condition'):
                e1 = sub[f"{col}_E1"].to_numpy()
                e2 = sub[f"{col}_E2"].to_numpy()
                mat = np.vstack([e1, e2]).T
                icc, l, u = icc_2k_with_ci(mat, n_bootstrap=n_bootstrap, random_state=random_state)
                results.append({
                    'Score': col,
                    'Scope': f"Condition={cond}",
                    'Expert1': expert1_name,
                    'Expert2': expert2_name,
                    'ExpertPair': f"{expert1_name}_vs_{expert2_name}",
                    'N': int(np.sum(~np.any(np.isnan(mat), axis=1))),
                    'ICC_2k': icc,
                    'CI95_L': l,
                    'CI95_U': u,
                })

    return pd.DataFrame(results)


def plot_icc_by_condition(icc_df: pd.DataFrame, out_dir: str) -> str:
    """
    Create a barplot comparing ICC(2,k) by condition for each summary score.
    Saves to out_dir and returns the filepath.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Keep only condition rows
    plot_df = icc_df[icc_df['Scope'].str.startswith('Condition=')].copy()
    if plot_df.empty:
        return ""
    plot_df['Condition'] = plot_df['Scope'].str.replace('Condition=', '', regex=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x='Score', y='ICC_2k', hue='Condition')
    plt.ylim(0, 1)
    plt.axhline(0.7, color='red', linestyle='--', linewidth=1, label='0.70 threshold')
    plt.title('ICC(2,k) by Condition (v2 - No O7/O8/Outcome Score)')
    plt.ylabel('ICC(2,k)')
    plt.legend()
    out_path = os.path.join(out_dir, 'icc_by_condition_summary_scores_v2.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main_step1() -> None:
    """Step 1: Load CSVs, clean/unify schemas, and print basic descriptive stats."""
    df_e1_raw, df_e2_raw = load_expert_csvs()

    print("=== Raw shapes ===")
    print({
        'E1_raw_shape': df_e1_raw.shape,
        'E2_raw_shape': df_e2_raw.shape,
    })
    print()

    df_e1_clean, df_e2_clean, score_cols = unify_schema(df_e1_raw, df_e2_raw)

    # For step 1, we focus on the summary scores (excluding Outcome Score and Total Score)
    summary_cols = [col for col in ['Process Score'] if col in score_cols]
    if len(summary_cols) == 0:
        raise ValueError("No summary score columns found. Expected: 'Process Score'.")

    print("=== Cleaned columns (E1) ===")
    print(df_e1_clean.columns.tolist())
    print("=== Cleaned columns (E2) ===")
    print(df_e2_clean.columns.tolist())
    print()

    print("=== Basic descriptive statistics (Summary Scores - v2) ===")
    desc_e1 = describe_scores(df_e1_clean, which='E1', score_cols=summary_cols)
    desc_e2 = describe_scores(df_e2_clean, which='E2', score_cols=summary_cols)
    desc = pd.concat([desc_e1, desc_e2], axis=0, ignore_index=True)
    print(desc.to_string(index=False))

    # Optionally, write out intermediate cleaned CSVs for transparency
    out_dir = os.path.join(get_project_dir(), 'intermediate')
    os.makedirs(out_dir, exist_ok=True)
    df_e1_clean.to_csv(os.path.join(out_dir, 'E1_clean_v2.csv'), index=False)
    df_e2_clean.to_csv(os.path.join(out_dir, 'E2_clean_v2.csv'), index=False)
    desc.to_csv(os.path.join(out_dir, 'summary_descriptives_v2.csv'), index=False)

    print()
    print(f"Cleaned CSVs and descriptives saved to: {out_dir}")


def main_step2() -> None:
    """Step 2: Compute ICC(2,k) with bootstrap CI and by-condition analysis, plus plot."""
    df_e1_raw, df_e2_raw = load_expert_csvs()
    df_e1_clean, df_e2_clean, score_cols = unify_schema(df_e1_raw, df_e2_raw)

    summary_cols = [col for col in ['Process Score'] if col in score_cols]
    if len(summary_cols) == 0:
        raise ValueError("No summary score columns found for ICC computation.")

    merged = align_experts(df_e1_clean, df_e2_clean, summary_cols, on_keys=['GroupLabel', 'Condition'])

    icc_df = compute_icc_summary(
        merged,
        summary_cols,
        by_condition=True,
        n_bootstrap=3000,
        random_state=123,
        expert1_name='E1',
        expert2_name='E2',
    )

    out_dir = os.path.join(get_project_dir(), 'intermediate')
    os.makedirs(out_dir, exist_ok=True)
    # Re-order rows so that Overall -> AI -> NoAI, and within each, Process only
    scope_order_map = {
        'Overall': 0,
        'Condition=AI': 1,
        'Condition=NoAI': 2,
    }
    score_order_map = {
        'Process Score': 0,
    }
    icc_df['_ScopeOrder'] = icc_df['Scope'].map(scope_order_map).fillna(99)
    icc_df['_ScoreOrder'] = icc_df['Score'].map(score_order_map).fillna(99)
    icc_df_sorted = icc_df.sort_values(by=['_ScopeOrder', '_ScoreOrder'], kind='stable').drop(columns=['_ScopeOrder', '_ScoreOrder'])

    icc_out_csv = os.path.join(out_dir, 'icc_results_v2.csv')
    icc_df_sorted.to_csv(icc_out_csv, index=False)

    plots_dir = os.path.join(get_project_dir(), 'consistency_plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = plot_icc_by_condition(icc_df, plots_dir)

    print("=== ICC(2,k) Results (with 95% bootstrap CI) [Grouped by Scope: Overall -> AI -> NoAI] ===")
    print("=== V2: No O7/O8/Outcome Score ===")
    # Pretty grouped printing
    for scope in ['Overall', 'Condition=AI', 'Condition=NoAI']:
        sub = icc_df_sorted[icc_df_sorted['Scope'] == scope]
        if sub.empty:
            continue
        print(f"\n-- {scope} --")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:0.3f}" if pd.notna(x) else "nan"))
    print()
    print(f"Saved ICC results to: {icc_out_csv}")
    if plot_path:
        print(f"Saved condition-wise ICC plot to: {plot_path}")


if __name__ == "__main__":
    # Run step 1 by default; uncomment next line to also run step 2
    main_step1()
    print("\nProceeding to Step 2: ICC analysis...\n")
    main_step2()
