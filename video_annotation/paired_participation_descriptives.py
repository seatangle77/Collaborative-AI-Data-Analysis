import argparse
import os
import base64
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats


# ----------------------------
# 数据结构与常量
# ----------------------------

CHINESE_TYPE_LABELS: Dict[str, str] = {
    "posture": "姿态",
    "gaze": "视线",
}

ENGLISH_TYPE_LABELS: Dict[str, str] = {
    "posture": "Posture",
    "gaze": "Gaze",
}

WIDE_COLUMNS = {
    "posture": {
        "ai": "AI_Mean_Level_posture",
        "noai": "NoAI_Mean_Level_posture",
    },
    "gaze": {
        "ai": "AI_Mean_Level_gaze",
        "noai": "NoAI_Mean_Level_gaze",
    },
}


@dataclass
class SummaryResult:
    type_key: str
    condition: str
    mean: float
    sd: float
    n: int
    ci_low: float
    ci_high: float


@dataclass
class DiffResult:
    type_key: str
    mean_diff: float
    sd_diff: float
    min_diff: float
    max_diff: float
    median: float
    iqr: float
    n: int
    shapiro_w: float
    shapiro_p: float


# ----------------------------
# 工具函数
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def t_confidence_interval(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = data.size
    if n <= 1:
        return (np.nan, np.nan)
    mean = data.mean()
    sd = data.std(ddof=1)
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def to_chinese_type(type_key: str) -> str:
    return CHINESE_TYPE_LABELS.get(type_key, type_key)


def to_english_type(type_key: str) -> str:
    return ENGLISH_TYPE_LABELS.get(type_key, type_key)


def compute_group_summary(values: np.ndarray) -> Tuple[float, float, int, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = values.size
    if n == 0:
        return (np.nan, np.nan, 0, np.nan, np.nan)
    mean = values.mean()
    sd = values.std(ddof=1) if n > 1 else 0.0
    ci_low, ci_high = t_confidence_interval(values)
    return mean, sd, n, ci_low, ci_high


def melt_long(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for type_key, cols in WIDE_COLUMNS.items():
        for condition_key, col_name in (("AI", cols["ai"]), ("NoAI", cols["noai"])):
            if col_name not in df.columns:
                continue
            subset = df[["Group", "MemberID", col_name]].copy()
            subset = subset.rename(columns={col_name: "Mean_Level"})
            subset["Condition"] = condition_key
            subset["Type_Key"] = type_key
            subset["Type"] = to_english_type(type_key)
            records.append(subset)
    if not records:
        return pd.DataFrame(columns=["Group", "MemberID", "Mean_Level", "Condition", "Type_Key", "Type"])  # empty
    long_df = pd.concat(records, ignore_index=True)
    long_df["Mean_Level"] = pd.to_numeric(long_df["Mean_Level"], errors="coerce").round(precision)
    return long_df


def compute_descriptives(df: pd.DataFrame) -> List[SummaryResult]:
    results: List[SummaryResult] = []
    for type_key, cols in WIDE_COLUMNS.items():
        for condition, col in (("AI", cols["ai"]), ("NoAI", cols["noai"])):
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").to_numpy()
            mean, sd, n, ci_low, ci_high = compute_group_summary(values)
            results.append(
                SummaryResult(type_key=type_key, condition=condition, mean=mean, sd=sd, n=n, ci_low=ci_low, ci_high=ci_high)
            )
    return results


def compute_diff_stats(df: pd.DataFrame, type_key: str) -> DiffResult:
    cols = WIDE_COLUMNS[type_key]
    a = pd.to_numeric(df.get(cols["ai"]), errors="coerce").to_numpy()
    b = pd.to_numeric(df.get(cols["noai"]), errors="coerce").to_numpy()
    diff = a - b
    diff = diff[~np.isnan(diff)]
    n = diff.size
    if n == 0:
        return DiffResult(
            type_key=type_key,
            mean_diff=np.nan,
            sd_diff=np.nan,
            min_diff=np.nan,
            max_diff=np.nan,
            median=np.nan,
            iqr=np.nan,
            n=0,
            shapiro_w=np.nan,
            shapiro_p=np.nan,
        )
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    min_diff = float(np.min(diff))
    max_diff = float(np.max(diff))
    median = float(np.median(diff))
    q1, q3 = np.percentile(diff, [25, 75])
    iqr = float(q3 - q1)
    # Shapiro-Wilk
    try:
        w, p = stats.shapiro(diff)
        shapiro_w, shapiro_p = float(w), float(p)
    except Exception:
        shapiro_w, shapiro_p = (np.nan, np.nan)
    return DiffResult(
        type_key=type_key,
        mean_diff=mean_diff,
        sd_diff=sd_diff,
        min_diff=min_diff,
        max_diff=max_diff,
        median=median,
        iqr=iqr,
        n=n,
        shapiro_w=shapiro_w,
        shapiro_p=shapiro_p,
    )


def results_to_dataframe(results: List[SummaryResult], precision: int) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Type": to_english_type(r.type_key),
                "Condition": r.condition,
                "Mean": None if np.isnan(r.mean) else round(r.mean, precision),
                "SD": None if np.isnan(r.sd) else round(r.sd, precision),
                "N": r.n,
                "CI_low": None if np.isnan(r.ci_low) else round(r.ci_low, precision),
                "CI_high": None if np.isnan(r.ci_high) else round(r.ci_high, precision),
            }
        )
    df = pd.DataFrame(rows)
    # 排序：姿态在前，AI 在前
    type_order = [ENGLISH_TYPE_LABELS["posture"], ENGLISH_TYPE_LABELS["gaze"]]
    cond_order = ["AI", "NoAI"]
    df["Type"] = pd.Categorical(df["Type"], categories=type_order, ordered=True)
    df["Condition"] = pd.Categorical(df["Condition"], categories=cond_order, ordered=True)
    return df.sort_values(["Type", "Condition"]).reset_index(drop=True)


def diffs_to_dataframe(results: List[DiffResult], precision: int) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Type": to_english_type(r.type_key),
                "Mean_Diff": None if np.isnan(r.mean_diff) else round(r.mean_diff, precision),
                "SD_Diff": None if np.isnan(r.sd_diff) else round(r.sd_diff, precision),
                "Min_Diff": None if np.isnan(r.min_diff) else round(r.min_diff, precision),
                "Max_Diff": None if np.isnan(r.max_diff) else round(r.max_diff, precision),
                "Median": None if np.isnan(r.median) else round(r.median, precision),
                "IQR": None if np.isnan(r.iqr) else round(r.iqr, precision),
                "N": r.n,
                "Shapiro_W": None if np.isnan(r.shapiro_w) else round(r.shapiro_w, 3),
                "Shapiro_p": None if np.isnan(r.shapiro_p) else round(r.shapiro_p, 4),
            }
        )
    df = pd.DataFrame(rows)
    type_order = [ENGLISH_TYPE_LABELS["posture"], ENGLISH_TYPE_LABELS["gaze"]]
    df["Type"] = pd.Categorical(df["Type"], categories=type_order, ordered=True)
    return df.sort_values(["Type"]).reset_index(drop=True)


def save_markdown_table(df: pd.DataFrame, path: str) -> None:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join([" --- " for _ in df.columns]) + "|"]
    for _, row in df.iterrows():
        cells = [str(row[c]) if pd.notna(row[c]) else "" for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------
# 绘图函数
# ----------------------------

def plot_diff_histograms(df: pd.DataFrame, out_dir: str, precision: int) -> None:
    ensure_dir(out_dir)
    for type_key in ("posture", "gaze"):
        cols = WIDE_COLUMNS[type_key]
        if cols["ai"] not in df or cols["noai"] not in df:
            continue
        a = pd.to_numeric(df[cols["ai"]], errors="coerce").to_numpy()
        b = pd.to_numeric(df[cols["noai"]], errors="coerce").to_numpy()
        diff = a - b
        diff = diff[~np.isnan(diff)]
        if diff.size == 0:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(diff, bins=10, kde=True, color="#4C78A8")
        plt.title(f"{to_english_type(type_key)} diff (AI - NoAI)")
        plt.xlabel("Difference")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"diff_hist_{type_key}.png"), dpi=200)
        plt.close()


def plot_qq(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    for type_key in ("posture", "gaze"):
        cols = WIDE_COLUMNS[type_key]
        if cols["ai"] not in df or cols["noai"] not in df:
            continue
        a = pd.to_numeric(df[cols["ai"]], errors="coerce").to_numpy()
        b = pd.to_numeric(df[cols["noai"]], errors="coerce").to_numpy()
        diff = a - b
        diff = diff[~np.isnan(diff)]
        if diff.size == 0:
            continue
        sm = stats.probplot(diff, dist="norm")
        theoretical_quants = np.array(sm[0][0])
        ordered_responses = np.array(sm[0][1])
        plt.figure(figsize=(6, 4))
        plt.scatter(theoretical_quants, ordered_responses, s=18, color="#F58518")
        plt.plot([theoretical_quants.min(), theoretical_quants.max()],
                 [ordered_responses.min(), ordered_responses.max()],
                 color="#555", linewidth=1)
        plt.title(f"{to_english_type(type_key)} diff Q-Q plot")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"diff_qq_{type_key}.png"), dpi=200)
        plt.close()


def plot_box_violin(long_df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    for type_label in long_df["Type"].dropna().unique():
        sub = long_df[long_df["Type"] == type_label].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(6, 4))
        sns.violinplot(data=sub, x="Condition", y="Mean_Level", inner=None, cut=0, palette="Set2")
        sns.boxplot(data=sub, x="Condition", y="Mean_Level", width=0.25, boxprops={"zorder": 2},
                    showmeans=True, meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"})
        plt.title(f"{type_label}: AI vs NoAI")
        plt.xlabel("")
        plt.ylabel("Mean participation level")
        plt.tight_layout()
        file_key = 'posture' if type_label == 'Posture' else 'gaze'
        plt.savefig(os.path.join(out_dir, f"box_violin_{file_key}.png"), dpi=200)
        plt.close()


def plot_paired_lines(long_df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    for type_label in long_df["Type"].dropna().unique():
        sub = long_df[long_df["Type"] == type_label].copy()
        if sub.empty:
            continue
        # 透视为宽：每个参与者两列 AI/NoAI
        pivot = sub.pivot_table(index=["Group", "MemberID"], columns="Condition", values="Mean_Level")
        pivot = pivot.dropna(subset=["AI", "NoAI"], how="any")
        if pivot.empty:
            continue
        plt.figure(figsize=(7, 4))
        x_positions = {"NoAI": 0, "AI": 1}
        for (_, _), row in pivot.iterrows():
            plt.plot([x_positions["NoAI"], x_positions["AI"]], [row["NoAI"], row["AI"]],
                     color="#4C78A8", alpha=0.6)
            plt.scatter([x_positions["NoAI"], x_positions["AI"]], [row["NoAI"], row["AI"]],
                        color=["#F58518", "#54A24B"], zorder=3, s=20)
        plt.xticks([0, 1], ["NoAI", "AI"])
        plt.ylabel("Mean participation level")
        plt.title(f"{type_label}: paired changes")
        plt.tight_layout()
        file_key = 'posture' if type_label == 'Posture' else 'gaze'
        plt.savefig(os.path.join(out_dir, f"paired_lines_{file_key}.png"), dpi=200)
        plt.close()


def plot_bar_with_ci(descriptive_df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    # 使用我们计算好的 t 置信区间
    for type_label in descriptive_df["Type"].dropna().unique():
        sub = descriptive_df[descriptive_df["Type"] == type_label].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(6, 4))
        x = np.arange(sub.shape[0])
        means = sub["Mean"].astype(float).to_numpy()
        ci_low = sub["CI_low"].astype(float).to_numpy()
        ci_high = sub["CI_high"].astype(float).to_numpy()
        errors = np.vstack([means - ci_low, ci_high - means])
        plt.bar(x, means, color=["#54A24B" if c == "AI" else "#F58518" for c in sub["Condition"]], alpha=0.9)
        plt.errorbar(x, means, yerr=errors, fmt="none", ecolor="#333", capsize=4, linewidth=1.2)
        plt.xticks(x, sub["Condition"].tolist())
        plt.ylabel("Mean participation level")
        plt.title(f"{type_label}: mean and 95% CI")
        plt.tight_layout()
        file_key = "posture" if type_label == "Posture" else "gaze"
        plt.savefig(os.path.join(out_dir, f"bar_ci_{file_key}.png"), dpi=200)
        plt.close()


# ----------------------------
# 主流程
# ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="配对数据的描述性统计、差值、正态性检验与可视化")
    default_input = os.path.join(os.path.dirname(__file__), "paired_participation_by_member_type.csv")
    default_out_dir = os.path.join(os.path.dirname(__file__), "summary")
    default_fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    default_report = os.path.join(os.path.dirname(__file__), "descriptive_report.html")

    parser.add_argument("--input", type=str, default=default_input, help="输入CSV（宽表）")
    parser.add_argument("--out-dir", type=str, default=default_out_dir, help="统计结果输出目录")
    parser.add_argument("--fig-dir", type=str, default=default_fig_dir, help="图像输出目录")
    parser.add_argument("--precision", type=int, default=2, help="小数位（默认2）")
    parser.add_argument("--report", type=str, default=default_report, help="整合描述性分析的HTML报告输出路径")

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.fig_dir)

    # 读数据
    df = pd.read_csv(args.input)

    # 3.1 均值与标准差
    desc_results = compute_descriptives(df)
    desc_df = results_to_dataframe(desc_results, precision=args.precision)
    desc_csv = os.path.join(args.out_dir, "summary_mean_sd.csv")
    desc_df.to_csv(desc_csv, index=False)

    # 3.2 差值统计 & 3.3 正态性（在 compute_diff_stats 内含 Shapiro）
    diff_results = [compute_diff_stats(df, type_key) for type_key in ("posture", "gaze")]
    diff_df = diffs_to_dataframe(diff_results, precision=args.precision)
    diff_csv = os.path.join(args.out_dir, "diff_stats.csv")
    diff_df.to_csv(diff_csv, index=False)

    # 3.3 可视化：差值直方图 & Q-Q 图
    plot_diff_histograms(df, args.fig_dir, precision=args.precision)
    plot_qq(df, args.fig_dir)

    # 3.4 可视化：箱/小提琴、配对线、条形+CI
    long_df = melt_long(df, precision=args.precision)
    if not long_df.empty:
        sns.set_theme(style="whitegrid")
        plot_box_violin(long_df, args.fig_dir)
        plot_paired_lines(long_df, args.fig_dir)
        plot_bar_with_ci(desc_df, args.fig_dir)

    # 生成整合 HTML 报告（内嵌base64图片，单文件）
    def _img_to_data_uri(path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # 组织图像路径
    def _paths_for(type_key: str) -> Dict[str, str]:
        return {
            "hist": os.path.join(args.fig_dir, f"diff_hist_{type_key}.png"),
            "qq": os.path.join(args.fig_dir, f"diff_qq_{type_key}.png"),
            "violin": os.path.join(args.fig_dir, f"box_violin_{type_key}.png"),
            "paired": os.path.join(args.fig_dir, f"paired_lines_{type_key}.png"),
            "bar": os.path.join(args.fig_dir, f"bar_ci_{type_key}.png"),
        }

    fig_paths = {k: _paths_for(k) for k in ("posture", "gaze")}

    # HTML 内容
    html_parts: List[str] = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang=\"en\">\n<head>")
    html_parts.append("<meta charset=\"utf-8\">")
    html_parts.append("<title>Paired Participation — Descriptive Report</title>")
    html_parts.append(
        "<style>"
        + "body{font-family:Arial,Helvetica,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;margin:24px;}"
        + "h1,h2{margin:12px 0;} table{border-collapse:collapse;margin:12px 0;}"
        + "table,th,td{border:1px solid #ccc;padding:6px 8px;} th{background:#f7f7f7;} "
        + "img{max-width:100%;height:auto;border:1px solid #eee;margin:6px 0;} "
        + ".sec{margin-bottom:28px;} "
        + ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;}"
        + "</style>\n</head>\n<body>"
    )
    html_parts.append("<h1>Paired Participation — Descriptive Report</h1>")
    html_parts.append(f"<p>Data source: {os.path.relpath(args.input, start=os.path.dirname(__file__))}</p>")

    # 3.1 表格
    html_parts.append("<div class=\"sec\">\n<h2>3.1 Mean and SD</h2>")
    html_parts.append(desc_df.to_html(index=False))
    html_parts.append("</div>")

    # 3.2 差值统计
    html_parts.append("<div class=\"sec\">\n<h2>3.2 Difference (AI - NoAI)</h2>")
    html_parts.append(diff_df.to_html(index=False))
    # 正态性口头描述
    for _, row in diff_df.iterrows():
        tlabel = row["Type"]
        p = row.get("Shapiro_p", np.nan)
        n = row.get("N", np.nan)
        if pd.notna(p):
            note = "approximately normal" if p >= 0.05 else "deviates from normality"
            html_parts.append(f"<p><b>{tlabel}</b>: Shapiro-Wilk p={p} (N={n}), {note}.</p>")
    html_parts.append("</div>")

    # 3.3 正态性检验（Shapiro + QQ图）
    html_parts.append("<div class=\"sec\">\n<h2>3.3 Normality test: Shapiro–Wilk + Q-Q plots</h2>")
    shapiro_tbl = diff_df[["Type", "Shapiro_W", "Shapiro_p", "N"]].copy()
    html_parts.append(shapiro_tbl.to_html(index=False))
    for type_key in ("posture", "gaze"):
        label = ENGLISH_TYPE_LABELS[type_key]
        html_parts.append(f"<h3>{label}</h3>")
        img = _img_to_data_uri(fig_paths[type_key]["qq"])
        if img:
            html_parts.append(f"<div><img src=\"{img}\" alt=\"{label} Q-Q\"></div>")
    html_parts.append("</div>")

    # 3.3 差值分布
    html_parts.append("<div class=\"sec\">\n<h2>3.4 Distribution of differences</h2>")
    for type_key in ("posture", "gaze"):
        label = ENGLISH_TYPE_LABELS[type_key]
        html_parts.append(f"<h3>{ENGLISH_TYPE_LABELS[type_key]}</h3>")
        html_parts.append("<div class=\"grid\">")
        for k in ("hist", "qq"):
            img = _img_to_data_uri(fig_paths[type_key][k])
            if img:
                html_parts.append(f"<div><img src=\"{img}\" alt=\"{label} {k}\"></div>")
        html_parts.append("</div>")
    html_parts.append("</div>")

    # 3.4 可视化
    html_parts.append("<div class=\"sec\">\n<h2>3.5 Visualizations</h2>")
    for type_key in ("posture", "gaze"):
        label = ENGLISH_TYPE_LABELS[type_key]
        html_parts.append(f"<h3>{label}</h3>")
        html_parts.append("<div class=\"grid\">")
        for k in ("violin", "paired", "bar"):
            img = _img_to_data_uri(fig_paths[type_key][k])
            if img:
                html_parts.append(f"<div><img src=\"{img}\" alt=\"{label} {k}\"></div>")
        html_parts.append("</div>")
    html_parts.append("</div>")

    html_parts.append("</body>\n</html>")
    html_content = "\n".join(html_parts)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"已输出描述性统计到: {desc_csv}")
    print(f"已输出差值统计到: {diff_csv}")
    print(f"图像已保存到目录: {args.fig_dir}")
    print(f"描述性HTML报告: {args.report}")


if __name__ == "__main__":
    main()


