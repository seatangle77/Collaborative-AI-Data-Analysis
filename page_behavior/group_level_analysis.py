import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'


@dataclass
class Config:
    input_csv: Path
    outdir: Path
    ci: float = 0.95
    bootstrap_iters: int = 10000
    random_seed: int = 42


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required_cols = ["Group", "speaker", "Type", "Op_Freq"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    # 仅保留合法类型
    df = df[df["Type"].isin(["AI", "NoAI"])].copy()
    # 规范类型
    df["Group"] = df["Group"].astype(str)
    df["speaker"] = df["speaker"].astype(str)
    df["Type"] = df["Type"].astype(str)
    df["Op_Freq"] = pd.to_numeric(df["Op_Freq"], errors="coerce")
    return df


def build_group_means(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 每组每条件：成员 Op_Freq 平均
    grp = (
        df.groupby(["Group", "Type"], as_index=False)["Op_Freq"].mean()
        .rename(columns={"Op_Freq": "GroupMean"})
    )
    # 透视为宽表
    wide = grp.pivot(index="Group", columns="Type", values="GroupMean").reset_index()
    # 仅保留同时有 AI 与 NoAI 的组
    wide = wide.dropna(subset=["AI", "NoAI"]).copy()
    wide["Diff"] = wide["AI"] - wide["NoAI"]
    return grp, wide


def desc_groups(grp_means: pd.DataFrame) -> pd.DataFrame:
    # 组级分布（每个条件在组间的分布）
    desc = (
        grp_means.groupby("Type")["GroupMean"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    return desc


def t_ci_mean(diff: np.ndarray, ci: float) -> Tuple[float, float]:
    n = diff.size
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        se = sd / np.sqrt(n)
        alpha = 1 - ci
        tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        return mean - tcrit * se, mean + tcrit * se
    return mean, mean


def bootstrap_ci(values: np.ndarray, func, ci: float, iters: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = values.size
    if n == 0:
        return np.nan, np.nan
    boots = np.empty(iters, dtype=float)
    for i in range(iters):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = float(func(sample))
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def normality_test(diff: np.ndarray) -> Dict[str, float]:
    if diff.size < 3:
        return {"W": np.nan, "p": np.nan}
    W, p = stats.shapiro(diff)
    return {"W": float(W), "p": float(p)}


def paired_test(diff: np.ndarray, cfg: Config) -> Dict[str, float]:
    res = normality_test(diff)
    test_type = "t-test" if (not np.isnan(res["p"]) and res["p"] >= 0.05) else "wilcoxon"
    out: Dict[str, float] = {"test": test_type}

    n = diff.size
    mean_diff = float(np.mean(diff)) if n > 0 else np.nan
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    dz = mean_diff / sd_diff if (sd_diff and not np.isnan(sd_diff) and sd_diff > 0) else np.nan

    if test_type == "t-test" and n >= 2:
        tstat, p = stats.ttest_rel(diff, np.zeros_like(diff))
        ci_lo, ci_hi = t_ci_mean(diff, cfg.ci)
        out.update({"stat": float(tstat), "p": float(p), "mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi})
    else:
        # Wilcoxon 符号秩检验（零差需要处理，scipy 会忽略零差）
        try:
            wstat, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            wstat, p = (np.nan, np.nan)
        # 中位数差与 bootstrap 均值差 CI（稳健）
        median_diff = float(np.median(diff)) if n > 0 else np.nan
        ci_lo, ci_hi = bootstrap_ci(diff, np.mean, cfg.ci, cfg.bootstrap_iters, cfg.random_seed)
        out.update({"stat": float(wstat), "p": float(p), "median_diff": median_diff, "ci_lo": ci_lo, "ci_hi": ci_hi})

    # dz 及其 bootstrap CI（对差值向量）
    dz_lo, dz_hi = bootstrap_ci(diff, lambda x: np.mean(x) / (np.std(x, ddof=1) if x.size > 1 else np.nan), cfg.ci, cfg.bootstrap_iters, cfg.random_seed)
    out.update({"dz": dz, "dz_ci_lo": dz_lo, "dz_ci_hi": dz_hi, "n": int(n)})
    return out


def plot_groups_bar(wide: pd.DataFrame, outpath: Path, cfg: Config) -> None:
    # wide has columns: Group, AI, NoAI, Diff
    data = wide[["AI", "NoAI"]].melt(var_name="Type", value_name="GroupMean")
    plt.figure(figsize=(5, 3.2))
    ax = sns.barplot(data=data, x="Type", y="GroupMean", ci=None, estimator=np.mean, color="#88aaff")
    # Add 95% CI manually
    ci = cfg.ci
    stats_df = data.groupby("Type")["GroupMean"].agg(["mean", "std", "count"]).reset_index()
    for i, row in stats_df.iterrows():
        mean = row["mean"]
        sd = row["std"]
        n = int(row["count"])
        if n > 1 and sd > 0:
            se = sd / np.sqrt(n)
            tcrit = stats.t.ppf(1 - (1 - ci) / 2, df=n - 1)
            lo, hi = mean - tcrit * se, mean + tcrit * se
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="#333333", capsize=4, lw=1.2)
    ax.set_ylabel("Group mean Op_Freq (events/min)")
    ax.set_xlabel("")
    ax.set_title("AI vs NoAI (Group means) — mean ± 95% CI")
    sns.despine()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_groups_box(wide: pd.DataFrame, outpath: Path) -> None:
    data = wide[["AI", "NoAI"]].melt(var_name="Type", value_name="GroupMean")
    plt.figure(figsize=(5, 3.2))
    ax = sns.boxplot(data=data, x="Type", y="GroupMean")
    sns.stripplot(data=data, x="Type", y="GroupMean", color="#555", size=2.5, alpha=0.5)
    ax.set_ylabel("Group mean Op_Freq (events/min)")
    ax.set_xlabel("")
    ax.set_title("Distribution of group means (AI vs NoAI)")
    sns.despine()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_paired_lines(wide: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(6, 4))
    for _, row in wide.iterrows():
        plt.plot(["NoAI", "AI"], [row["NoAI"], row["AI"]], marker="o", markersize=3.5, linewidth=1.0, color="#1f77b4", alpha=0.75)
    plt.ylabel("Group mean Op_Freq (events/min)")
    plt.title("Paired lines by group (AI vs NoAI)")
    sns.despine()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_diff_hist(diff: np.ndarray, outpath: Path) -> None:
    plt.figure(figsize=(5, 3.2))
    sns.histplot(diff, kde=True, color="#6aa84f", bins='auto')
    plt.axvline(0, color="#333", linestyle="--", linewidth=1)
    plt.xlabel("Diff = AI - NoAI (events/min)")
    plt.title("Distribution of differences (group level)")
    sns.despine()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_qq(diff: np.ndarray, outpath: Path) -> None:
    plt.figure(figsize=(4.2, 4.2))
    stats.probplot(diff, dist="norm", plot=plt)
    plt.title("QQ-Plot (group differences)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def write_html_report(
    cfg: Config,
    wide: pd.DataFrame,
    desc_df: pd.DataFrame,
    norm_res: Dict[str, float],
    paired_res: Dict[str, float],
    balance_html_sections: Dict[str, str],
) -> None:
    out_html = cfg.outdir / "group_level_analysis_report.html"
    # Render tables to HTML fragments
    desc_html = desc_df.to_html(index=False)
    paired_df = pd.DataFrame([paired_res])
    paired_html = paired_df.to_html(index=False)
    norm_df = pd.DataFrame([norm_res])
    norm_html = norm_df.to_html(index=False)
    # Minimal HTML template (English)
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Group-level Page Behavior Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 16px; }}
    h1,h2 {{ margin: 12px 0; }}
    img {{ max-width: 900px; display: block; margin: 12px 0; }}
    table {{ border-collapse: collapse; }}
    table, th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
  </style>
  </head>
  <body>
    <h1>Group-level Page Behavior Analysis</h1>
    <p>Input: {cfg.input_csv}</p>

    <h2>1) Descriptive statistics (group means)</h2>
    {desc_html}
    <img src=\"groups_bar_op_freq.png\" alt=\"Bar: AI vs NoAI\" />
    <img src=\"groups_box_op_freq.png\" alt=\"Box: AI vs NoAI\" />

    <h2>2) Normality test (Diff = AI - NoAI)</h2>
    {norm_html}
    <img src=\"diff_hist_groups.png\" alt=\"Diff histogram\" />
    <img src=\"qqplot_groups_diff.png\" alt=\"QQ Plot\" />

    <h2>3) Paired test and effect size</h2>
    {paired_html}
    <img src=\"paired_lines_groups.png\" alt=\"Paired lines\" />

    <h2>4) Within-group balance (Gini & CV)</h2>
    <h3>4.1 Gini coefficient</h3>
    {balance_html_sections.get('desc_gini', '')}
    {balance_html_sections.get('norm_gini', '')}
    {balance_html_sections.get('paired_gini', '')}
    <img src=\"balance_gini_bar.png\" alt=\"Gini bar\" />
    <img src=\"balance_gini_box.png\" alt=\"Gini box\" />
    <img src=\"balance_gini_paired_lines.png\" alt=\"Gini paired lines\" />
    <img src=\"balance_gini_diff_hist.png\" alt=\"Gini diff hist\" />
    <img src=\"balance_gini_qq.png\" alt=\"Gini QQ plot\" />

    <h3>4.2 Coefficient of Variation (CV)</h3>
    {balance_html_sections.get('desc_cv', '')}
    {balance_html_sections.get('norm_cv', '')}
    {balance_html_sections.get('paired_cv', '')}
    <img src=\"balance_cv_bar.png\" alt=\"CV bar\" />
    <img src=\"balance_cv_box.png\" alt=\"CV box\" />
    <img src=\"balance_cv_paired_lines.png\" alt=\"CV paired lines\" />
    <img src=\"balance_cv_diff_hist.png\" alt=\"CV diff hist\" />
    <img src=\"balance_cv_qq.png\" alt=\"CV QQ plot\" />

    <hr />
    <p>Auto-generated. Confidence level {int(100*cfg.ci)}%, bootstrap {cfg.bootstrap_iters} iterations.</p>
  </body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def run(cfg: Config) -> None:
    ensure_outdir(cfg.outdir)

    df = load_data(cfg.input_csv)
    grp, wide = build_group_means(df)

    # 输出表：组级描述性
    desc_df = desc_groups(grp)
    desc_df.to_csv(cfg.outdir / "desc_groups.csv", index=False)

    # 输出表：每组配对结果
    wide_out = wide.copy()
    wide_out = wide_out.rename(columns={"AI": "GroupMean_AI", "NoAI": "GroupMean_NoAI"})
    wide_out.to_csv(cfg.outdir / "group_paired_results.csv", index=False)

    # 正态性与检验
    diff = wide["Diff"].values.astype(float)
    norm_res = normality_test(diff)
    pd.DataFrame([norm_res]).to_csv(cfg.outdir / "normality_groups.csv", index=False)

    paired_res = paired_test(diff, cfg)
    pd.DataFrame([paired_res]).to_csv(cfg.outdir / "paired_groups_summary.csv", index=False)

    # 图表
    plot_groups_bar(wide, cfg.outdir / "groups_bar_op_freq.png", cfg)
    plot_groups_box(wide, cfg.outdir / "groups_box_op_freq.png")
    plot_paired_lines(wide, cfg.outdir / "paired_lines_groups.png")
    plot_diff_hist(diff, cfg.outdir / "diff_hist_groups.png")
    plot_qq(diff, cfg.outdir / "qqplot_groups_diff.png")

    # -------- Balance metrics: Gini & CV --------
    def _gini(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan
        if np.all(x == 0):
            return 0.0
        if np.any(x < 0):
            x = np.clip(x, 0, None)
        x_sorted = np.sort(x)
        n = x_sorted.size
        cumx = np.cumsum(x_sorted)
        g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
        return float(g)

    def _cv(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 2:
            return np.nan
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        if mean <= 0:
            return np.nan
        return float(sd / mean)

    # compute per-group per-condition balance metrics from member-level Op_Freq
    balance_records: list[dict] = []
    for (g, t), sub in df.groupby(["Group", "Type"]):
        vals = sub["Op_Freq"].values.astype(float)
        balance_records.append({
            "Group": str(g),
            "Type": str(t),
            "Gini": _gini(vals),
            "CV": _cv(vals),
            "n_members": int(np.sum(~np.isnan(vals)))
        })
    balance_long = pd.DataFrame(balance_records)
    balance_long.to_csv(cfg.outdir / "balance_long.csv", index=False)

    # Wide tables for paired comparisons
    def _wide_metric(metric: str) -> pd.DataFrame:
        w = balance_long.pivot(index="Group", columns="Type", values=metric).reset_index()
        w = w.dropna(subset=["AI", "NoAI"]).copy()
        w["Diff"] = w["AI"] - w["NoAI"]
        return w

    wide_gini = _wide_metric("Gini")
    wide_cv = _wide_metric("CV")
    wide_gini.rename(columns={"AI": "Gini_AI", "NoAI": "Gini_NoAI"}).to_csv(cfg.outdir / "balance_gini_paired_results.csv", index=False)
    wide_cv.rename(columns={"AI": "CV_AI", "NoAI": "CV_NoAI"}).to_csv(cfg.outdir / "balance_cv_paired_results.csv", index=False)

    # Descriptives for balance metrics
    def _desc_balance(metric: str) -> pd.DataFrame:
        return balance_long.groupby("Type")[metric].agg(mean="mean", sd="std", n="count").reset_index()

    desc_gini = _desc_balance("Gini"); desc_gini.to_csv(cfg.outdir / "desc_balance_gini.csv", index=False)
    desc_cv = _desc_balance("CV"); desc_cv.to_csv(cfg.outdir / "desc_balance_cv.csv", index=False)

    # Normality & paired tests for balance metrics
    gini_diff = wide_gini["Diff"].values.astype(float)
    cv_diff = wide_cv["Diff"].values.astype(float)
    norm_gini = normality_test(gini_diff); pd.DataFrame([norm_gini]).to_csv(cfg.outdir / "normality_balance_gini.csv", index=False)
    norm_cv = normality_test(cv_diff); pd.DataFrame([norm_cv]).to_csv(cfg.outdir / "normality_balance_cv.csv", index=False)

    paired_gini = paired_test(gini_diff, cfg); pd.DataFrame([paired_gini]).to_csv(cfg.outdir / "paired_balance_gini_summary.csv", index=False)
    paired_cv = paired_test(cv_diff, cfg); pd.DataFrame([paired_cv]).to_csv(cfg.outdir / "paired_balance_cv_summary.csv", index=False)

    # Plots for balance metrics
    def _plot_balance_bar(desc_df: pd.DataFrame, outpath: Path, ylabel: str, title: str) -> None:
        plt.figure(figsize=(5, 3.2))
        # manual bars from desc stats
        order = ["NoAI", "AI"] if set(desc_df["Type"]) == {"AI", "NoAI"} else list(desc_df["Type"]) 
        means = [float(desc_df.loc[desc_df["Type"]==k, "mean"]) if (desc_df["Type"]==k).any() else np.nan for k in order]
        sds = [float(desc_df.loc[desc_df["Type"]==k, "sd"]) if (desc_df["Type"]==k).any() else np.nan for k in order]
        ns = [int(desc_df.loc[desc_df["Type"]==k, "n"]) if (desc_df["Type"]==k).any() else 0 for k in order]
        xs = np.arange(len(order))
        bars = plt.bar(xs, means, color="#88aaff", width=0.6)
        for i, (m, sd, n) in enumerate(zip(means, sds, ns)):
            if n and n > 1 and sd and sd > 0:
                se = sd / np.sqrt(n)
                tcrit = stats.t.ppf(1 - (1 - cfg.ci) / 2, df=n - 1)
                lo, hi = m - tcrit * se, m + tcrit * se
                plt.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333", capsize=4, lw=1.2)
        plt.xticks(xs, order)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_balance_box(long_df: pd.DataFrame, metric: str, outpath: Path, ylabel: str, title: str) -> None:
        data = long_df[["Type", metric]].rename(columns={metric: "value"})
        plt.figure(figsize=(5, 3.2))
        ax = sns.boxplot(data=data, x="Type", y="value")
        sns.stripplot(data=data, x="Type", y="value", color="#555", size=2.5, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_title(title)
        sns.despine()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_balance_paired_lines(wide_df: pd.DataFrame, outpath: Path, ylabel: str, title: str) -> None:
        plt.figure(figsize=(6, 4))
        for _, row in wide_df.iterrows():
            plt.plot(["NoAI", "AI"], [row["NoAI"], row["AI"]], marker="o", markersize=3.5, linewidth=1.0, color="#1f77b4", alpha=0.75)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_balance_diff_hist(diff: np.ndarray, outpath: Path, xlabel: str, title: str) -> None:
        plt.figure(figsize=(5, 3.2))
        sns.histplot(diff, kde=True, color="#6aa84f", bins='auto')
        plt.axvline(0, color="#333", linestyle="--", linewidth=1)
        plt.xlabel(xlabel)
        plt.title(title)
        sns.despine()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_balance_qq(diff: np.ndarray, outpath: Path, title: str) -> None:
        plt.figure(figsize=(4.2, 4.2))
        stats.probplot(diff, dist="norm", plot=plt)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

    # Gini plots
    _plot_balance_bar(desc_gini, cfg.outdir / "balance_gini_bar.png", "Gini coefficient", "Group balance (Gini): mean ± 95% CI")
    _plot_balance_box(balance_long, "Gini", cfg.outdir / "balance_gini_box.png", "Gini coefficient", "Distribution of Gini by condition")
    _plot_balance_paired_lines(wide_gini.rename(columns={"AI": "AI", "NoAI": "NoAI"}), cfg.outdir / "balance_gini_paired_lines.png", "Gini coefficient", "Paired Gini by group")
    _plot_balance_diff_hist(gini_diff, cfg.outdir / "balance_gini_diff_hist.png", "Gini Diff = AI - NoAI", "Distribution of Gini differences")
    _plot_balance_qq(gini_diff, cfg.outdir / "balance_gini_qq.png", "QQ-Plot (Gini differences)")

    # CV plots
    _plot_balance_bar(desc_cv, cfg.outdir / "balance_cv_bar.png", "Coefficient of Variation (CV)", "Group balance (CV): mean ± 95% CI")
    _plot_balance_box(balance_long, "CV", cfg.outdir / "balance_cv_box.png", "Coefficient of Variation (CV)", "Distribution of CV by condition")
    _plot_balance_paired_lines(wide_cv.rename(columns={"AI": "AI", "NoAI": "NoAI"}), cfg.outdir / "balance_cv_paired_lines.png", "Coefficient of Variation (CV)", "Paired CV by group")
    _plot_balance_diff_hist(cv_diff, cfg.outdir / "balance_cv_diff_hist.png", "CV Diff = AI - NoAI", "Distribution of CV differences")
    _plot_balance_qq(cv_diff, cfg.outdir / "balance_cv_qq.png", "QQ-Plot (CV differences)")

    # Build HTML fragments for balance section
    balance_sections = {
        "desc_gini": desc_gini.to_html(index=False),
        "norm_gini": pd.DataFrame([norm_gini]).to_html(index=False),
        "paired_gini": pd.DataFrame([paired_gini]).to_html(index=False),
        "desc_cv": desc_cv.to_html(index=False),
        "norm_cv": pd.DataFrame([norm_cv]).to_html(index=False),
        "paired_cv": pd.DataFrame([paired_cv]).to_html(index=False),
    }

    # 报告（相对引用图像文件名，便于整体移动目录）
    write_html_report(cfg, wide, desc_df, norm_res, paired_res, balance_sections)

    print(f"完成：输出目录 {cfg.outdir}")


def main():
    parser = argparse.ArgumentParser(description="Group-level analysis for page behavior (Op_Freq)")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[0] / "page_behavior_preprocessed.csv"),
        help="输入 CSV，需包含 Group,speaker,Type,Op_Freq",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[0] / "analysis"),
        help="输出目录（将自动创建）",
    )
    parser.add_argument("--ci", type=float, default=0.95, help="置信区间，默认 0.95")
    parser.add_argument("--boots", type=int, default=10000, help="bootstrap 次数，默认 10000")
    args = parser.parse_args()

    cfg = Config(
        input_csv=Path(args.input),
        outdir=Path(args.outdir),
        ci=float(args.ci),
        bootstrap_iters=int(args.boots),
        random_seed=42,
    )
    run(cfg)


if __name__ == "__main__":
    main()


