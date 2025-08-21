import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# -----------------------------
# Utilities: IO and schema unify
# -----------------------------

def get_project_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_expert_csvs(
    e1_filename: str = "E1_scoring.csv",
    e2_filename: str = "E2_socring.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = get_project_dir()
    e1_path = os.path.join(base_dir, e1_filename)
    e2_path = os.path.join(base_dir, e2_filename)
    if not os.path.exists(e1_path):
        raise FileNotFoundError(f"E1 CSV not found: {e1_path}")
    if not os.path.exists(e2_path):
        raise FileNotFoundError(f"E2 CSV not found: {e2_path}")
    return pd.read_csv(e1_path), pd.read_csv(e2_path)


def unify_schema(df_e1: pd.DataFrame, df_e2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df1 = df_e1.copy()
    df2 = df_e2.copy()

    # E1: '文本 13' holds the label like 'Group 0'
    if '文本 13' in df1.columns:
        df1 = df1.rename(columns={'文本 13': 'GroupLabel'})
        if 'Group' in df1.columns:
            df1 = df1.drop(columns=['Group'])
    elif 'Group' in df1.columns:
        df1 = df1.rename(columns={'Group': 'GroupLabel'})

    # E2: drop NO., rename Group
    if 'NO.' in df2.columns:
        df2 = df2.drop(columns=['NO.'])
    if 'Group' in df2.columns:
        df2 = df2.rename(columns={'Group': 'GroupLabel'})

    for dfx in (df1, df2):
        if 'Condition' not in dfx.columns:
            raise ValueError("Column 'Condition' is required")
        if 'GroupLabel' not in dfx.columns:
            raise ValueError("Column 'GroupLabel' is required")
        dfx['Condition'] = dfx['Condition'].astype(str).str.strip()
        dfx['GroupLabel'] = dfx['GroupLabel'].astype(str).str.strip()

    # Identify scoring columns present in both - EXCLUDING O7, O8, Outcome Score
    score_cols = [
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
        'Process Score', 'Total Score'
    ]
    common_cols = [c for c in score_cols if c in df1.columns and c in df2.columns]
    for dfx in (df1, df2):
        for c in common_cols:
            dfx[c] = pd.to_numeric(dfx[c], errors='coerce')

    return df1, df2, common_cols


# -----------------------------
# Step 4: Within-group paired tests (AI vs NoAI) per expert
# -----------------------------

SummaryScores = ['Process Score']  # V2: No Outcome Score, No Total Score

# Smallest effect size of interest (SESOI) per summary score
SESOI: Dict[str, float] = {
    'Process Score': 0.5,
}


def build_paired_vectors(df_exp: pd.DataFrame, score_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return aligned vectors (NoAI, AI) for the given score column, paired by GroupLabel."""
    subset = df_exp[df_exp['Condition'].isin(['AI', 'NoAI'])][['GroupLabel', 'Condition', score_col]].copy()
    pivot = subset.pivot(index='GroupLabel', columns='Condition', values=score_col)
    # Keep only rows with both conditions
    pivot = pivot.dropna(subset=['AI', 'NoAI'], how='any')
    groups = pivot.index.astype(str).tolist()
    return pivot['NoAI'].to_numpy(), pivot['AI'].to_numpy(), groups


def normality_of_diff(noai: np.ndarray, ai: np.ndarray) -> Tuple[float, float]:
    diff = ai - noai
    diff = diff[~np.isnan(diff)]
    if diff.size < 3:
        return np.nan, 1.0
    w, p = stats.shapiro(diff)
    return float(w), float(p)


def paired_tests(noai: np.ndarray, ai: np.ndarray) -> Dict[str, float]:
    diff = ai - noai
    diff = diff[~np.isnan(diff)]
    n = diff.size

    result: Dict[str, float] = {"n": n}
    if n < 2:
        result.update({
            'test': np.nan, 'stat': np.nan, 'p': np.nan,
            'effect': np.nan, 'effect_name': '',
            'mean_diff': np.nan, 'ci_l': np.nan, 'ci_u': np.nan
        })
        return result

    # Normality test
    _, p_norm = normality_of_diff(noai, ai)

    # Prefer Wilcoxon for discrete small samples; use t-test only if p_norm >= .05
    if np.isfinite(p_norm) and p_norm >= 0.05 and n >= 3:
        # Paired t-test
        tstat, pval = stats.ttest_rel(ai, noai, nan_policy='omit')
        mean_diff = float(np.nanmean(diff))
        sd_diff = float(np.nanstd(diff, ddof=1)) if n > 1 else np.nan
        dz = mean_diff / sd_diff if (sd_diff is not None and sd_diff > 0) else np.nan
        # 95% CI for mean difference
        se = sd_diff / np.sqrt(n) if sd_diff is not None else np.nan
        tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
        ci_l = mean_diff - tcrit * se if (se is not None and np.isfinite(tcrit)) else np.nan
        ci_u = mean_diff + tcrit * se if (se is not None and np.isfinite(tcrit)) else np.nan
        result.update({
            'test': 'paired_t', 'stat': float(tstat), 'p': float(pval),
            'effect': float(dz) if np.isfinite(dz) else np.nan, 'effect_name': 'Cohen_dz',
            'mean_diff': mean_diff, 'ci_l': float(ci_l), 'ci_u': float(ci_u)
        })
    else:
        # Wilcoxon signed-rank test (two-sided)
        try:
            wstat, pval = stats.wilcoxon(ai, noai, zero_method='wilcox')
        except ValueError:
            # All differences zero or invalid; fall back to sign test approx
            wstat, pval = 0.0, 1.0
        # Approximate z from two-sided p-value
        # z has direction of mean difference
        if pval > 0:
            z_abs = stats.norm.ppf(1 - pval / 2)
            z = np.sign(np.nanmean(diff)) * z_abs
            r = z / np.sqrt(n)
        else:
            z, r = np.nan, np.nan
        result.update({
            'test': 'wilcoxon', 'stat': float(wstat), 'p': float(pval),
            'effect': float(r) if np.isfinite(r) else np.nan, 'effect_name': 'r_rb_approx',
            'mean_diff': float(np.nanmean(diff)), 'ci_l': np.nan, 'ci_u': np.nan
        })

    return result


def holm_bonferroni(pvals: List[float], alpha: float = 0.05) -> List[float]:
    """Return Holm-Bonferroni adjusted p-values for a list of raw p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        rank = i + 1
        adj_p = (m - i) * pvals[idx]
        adj[idx] = max(adj_p, prev)
        prev = adj[idx]
    # Cap at 1
    adj = np.minimum(adj, 1.0)
    return adj.tolist()


def bootstrap_ci_mean(diff: np.ndarray, n_bootstrap: int = 5000, ci: float = 0.95, seed: int = 123) -> Tuple[float, float]:
    """Bootstrap percentile CI for the mean of paired differences."""
    valid = diff[~np.isnan(diff)]
    n = valid.size
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(valid[idx]))
    alpha = (1 - ci) / 2
    l = np.nanpercentile(boots, 100 * alpha)
    u = np.nanpercentile(boots, 100 * (1 - alpha))
    return float(l), float(u)


def tost_paired(diff: np.ndarray, delta: float) -> Tuple[float, float, float, bool]:
    """Two one-sided tests (TOST) for paired differences against ±delta (SESOI)."""
    valid = diff[~np.isnan(diff)]
    n = valid.size
    if n < 2:
        return np.nan, np.nan, np.nan, False
    mean_diff = float(np.mean(valid))
    sd = float(np.std(valid, ddof=1))
    if sd == 0:
        return np.nan, np.nan, np.nan, False
    se = sd / np.sqrt(n)
    df = n - 1
    # H1: mean_diff > -delta
    t_lower = (mean_diff - (-delta)) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    # H1: mean_diff < +delta
    t_upper = (delta - mean_diff) / se
    p_upper = 1 - stats.t.cdf(t_upper, df)
    p_max = max(p_lower, p_upper)
    equivalent = (p_lower < 0.05) and (p_upper < 0.05)
    return float(p_lower), float(p_upper), float(p_max), bool(equivalent)


def paired_analysis_per_expert(df_exp: pd.DataFrame, expert_name: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for score in SummaryScores:
        if score not in df_exp.columns:
            continue
        noai, ai, groups = build_paired_vectors(df_exp, score)
        res = paired_tests(noai, ai)
        diff = (ai - noai).astype(float)

        # Use existing CI if available; otherwise bootstrap CI for mean difference
        ci_l = res.get('ci_l', np.nan)
        ci_u = res.get('ci_u', np.nan)
        if not (np.isfinite(ci_l) and np.isfinite(ci_u)):
            b_l, b_u = bootstrap_ci_mean(diff, n_bootstrap=5000, ci=0.95, seed=123)
            ci_l, ci_u = b_l, b_u

        # TOST and SESOI-based CI decision
        sesoi = SESOI.get(score, np.nan)
        if np.isfinite(sesoi):
            p_l, p_u, p_tost, equiv = tost_paired(diff, sesoi)
            ci_within = (np.isfinite(ci_l) and np.isfinite(ci_u) and (ci_l >= -sesoi) and (ci_u <= sesoi))
        else:
            p_l = p_u = p_tost = np.nan
            equiv = False
            ci_within = False

        if equiv and ci_within:
            eq_conclusion = 'Equivalent'
        elif equiv or ci_within:
            eq_conclusion = 'Probably equivalent'
        else:
            eq_conclusion = 'Not equivalent'

        rows.append({
            'Expert': expert_name,
            'Score': score,
            'N_pairs': int(res['n']),
            'Test': res['test'],
            'Statistic': res['stat'],
            'P_value_raw': res['p'],
            'Effect': res['effect'],
            'Effect_name': res['effect_name'],
            'Mean_Diff_AI_minus_NoAI': res['mean_diff'],
            'CI95_L': float(ci_l) if np.isfinite(ci_l) else np.nan,
            'CI95_U': float(ci_u) if np.isfinite(ci_u) else np.nan,
            'SESOI': float(sesoi) if np.isfinite(sesoi) else np.nan,
            'CI_within_SESOI': bool(ci_within),
            'TOST_p_lower': float(p_l) if np.isfinite(p_l) else np.nan,
            'TOST_p_upper': float(p_u) if np.isfinite(p_u) else np.nan,
            'TOST_p': float(p_tost) if np.isfinite(p_tost) else np.nan,
            'TOST_equivalent': bool(equiv),
            'Equivalence_Conclusion': eq_conclusion,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out['P_value_adj_Holm'] = holm_bonferroni(out['P_value_raw'].fillna(1.0).tolist())
    return out


def plot_paired_lines(df_exp: pd.DataFrame, expert_name: str, out_dir: str) -> List[str]:
    paths: List[str] = []
    os.makedirs(out_dir, exist_ok=True)
    for score in SummaryScores:
        if score not in df_exp.columns:
            continue
        noai, ai, groups = build_paired_vectors(df_exp, score)
        if len(groups) == 0:
            continue
        plt.figure(figsize=(6, 4))
        x_noai, x_ai = np.zeros_like(noai, dtype=float), np.ones_like(ai, dtype=float)
        for y0, y1 in zip(noai, ai):
            plt.plot([0, 1], [y0, y1], color='gray', alpha=0.5)
        # Add scatter with jitter
        jitter = (np.random.rand(len(noai)) - 0.5) * 0.05
        plt.scatter(x_noai + jitter, noai, label='NoAI', color='#1f77b4')
        plt.scatter(x_ai + jitter, ai, label='AI', color='#d62728')
        plt.xticks([0, 1], ['NoAI', 'AI'])
        plt.title(f"{expert_name} - {score} (Paired AI vs NoAI) - V2")
        plt.ylabel(score)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"paired_lines_{expert_name.replace(' ', '_')}_{score.replace(' ', '_')}_v2.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        paths.append(out_path)
    return paths


def plot_diff_bar(df_exp: pd.DataFrame, expert_name: str, out_dir: str) -> List[str]:
    """Plot per-group bars of diff = AI - NoAI with mean and 95% CI lines."""
    os.makedirs(out_dir, exist_ok=True)
    out_paths: List[str] = []
    for score in SummaryScores:
        if score not in df_exp.columns:
            continue
        noai, ai, groups = build_paired_vectors(df_exp, score)
        if len(groups) == 0:
            continue
        diff = ai - noai
        n = diff.size
        mean_diff = float(np.nanmean(diff))
        sd_diff = float(np.nanstd(diff, ddof=1)) if n > 1 else np.nan
        se = sd_diff / np.sqrt(n) if n > 1 else np.nan
        try:
            tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
        except Exception:
            tcrit = np.nan
        ci_l = mean_diff - (tcrit * se) if np.isfinite(tcrit) else np.nan
        ci_u = mean_diff + (tcrit * se) if np.isfinite(tcrit) else np.nan

        plt.figure(figsize=(8, 4))
        positions = np.arange(len(groups))
        plt.bar(positions, diff, color="#6baed6")
        plt.axhline(0, color='black', linewidth=1)
        # Mean and CI lines
        plt.axhline(mean_diff, color='#e41a1c', linestyle='-', linewidth=1.5, label='Mean diff')
        if np.isfinite(ci_l) and np.isfinite(ci_u):
            plt.axhline(ci_l, color='#e41a1c', linestyle='--', linewidth=1, label='95% CI')
            plt.axhline(ci_u, color='#e41a1c', linestyle='--', linewidth=1)
        plt.xticks(positions, groups, rotation=45, ha='right')
        plt.ylabel(f"AI - NoAI ({score})")
        plt.title(f"{expert_name} - {score} (Difference per Group) - V2")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"diff_bar_{expert_name.replace(' ', '_')}_{score.replace(' ', '_')}_v2.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        out_paths.append(out_path)
    return out_paths


def plot_diff_box(df_exp: pd.DataFrame, expert_name: str, out_dir: str) -> List[str]:
    """Plot boxplot of diffs = AI - NoAI across groups with points and mean line."""
    os.makedirs(out_dir, exist_ok=True)
    out_paths: List[str] = []
    for score in SummaryScores:
        if score not in df_exp.columns:
            continue
        noai, ai, groups = build_paired_vectors(df_exp, score)
        if len(groups) == 0:
            continue
        diff = ai - noai
        mean_diff = float(np.nanmean(diff))

        plt.figure(figsize=(4.8, 4))
        df_plot = pd.DataFrame({'diff': diff})
        sns.boxplot(y='diff', data=df_plot, width=0.3, color='#9ecae1')
        sns.stripplot(y='diff', data=df_plot, color='#3182bd', alpha=0.7, jitter=0.15)
        plt.axhline(0, color='black', linewidth=1)
        plt.axhline(mean_diff, color='#e41a1c', linestyle='-', linewidth=1.5, label='Mean diff')
        plt.ylabel(f"AI - NoAI ({score})")
        plt.title(f"{expert_name} - {score} (Difference Boxplot) - V2")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"diff_box_{expert_name.replace(' ', '_')}_{score.replace(' ', '_')}_v2.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    df_e1_raw, df_e2_raw = load_expert_csvs()
    df_e1, df_e2, _ = unify_schema(df_e1_raw, df_e2_raw)

    # Focus on summary scores only (excluding Outcome Score)
    keep = ['GroupLabel', 'Condition'] + [c for c in SummaryScores if c in df_e1.columns]
    df_e1 = df_e1[keep].copy()
    df_e2 = df_e2[keep].copy()

    res_e1 = paired_analysis_per_expert(df_e1, 'E1')
    res_e2 = paired_analysis_per_expert(df_e2, 'E2')
    res_all = pd.concat([res_e1, res_e2], axis=0, ignore_index=True)

    out_dir = os.path.join(get_project_dir(), 'intermediate')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'within_group_tests_v2.csv')
    res_all.to_csv(out_csv, index=False)

    plots_dir = os.path.join(get_project_dir(), 'consistency_plots')
    paths = []
    paths.extend(plot_paired_lines(df_e1, 'E1', plots_dir))
    paths.extend(plot_paired_lines(df_e2, 'E2', plots_dir))
    # New diff views
    paths.extend(plot_diff_bar(df_e1, 'E1', plots_dir))
    paths.extend(plot_diff_bar(df_e2, 'E2', plots_dir))
    paths.extend(plot_diff_box(df_e1, 'E1', plots_dir))
    paths.extend(plot_diff_box(df_e2, 'E2', plots_dir))

    print("=== Within-group paired tests (AI vs NoAI) - V2 ===")
    print("=== No O7/O8/Outcome Score ===")
    print(res_all.to_string(index=False, float_format=lambda x: f"{x:0.3f}" if pd.notna(x) else "nan"))
    print()
    print(f"Saved results to: {out_csv}")
    for p in paths:
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
