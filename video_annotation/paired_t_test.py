import os
import base64
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_input(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    df = pd.read_csv(input_path)
    return df


def t_ci_for_mean_diff(diff: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    diff = diff[~np.isnan(diff)]
    n = diff.size
    if n <= 1:
        return (np.nan, np.nan)
    sd = np.std(diff, ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    mean = float(np.mean(diff))
    return mean - tcrit * se, mean + tcrit * se


def paired_t_for_type(df: pd.DataFrame, ai_col: str, noai_col: str, precision: int = 3) -> Dict[str, float]:
    ai = pd.to_numeric(df.get(ai_col), errors="coerce").to_numpy()
    noai = pd.to_numeric(df.get(noai_col), errors="coerce").to_numpy()
    mask = np.isfinite(ai) & np.isfinite(noai)
    ai = ai[mask]
    noai = noai[mask]
    n = int(ai.size)
    result: Dict[str, float] = {"N": n}
    if n == 0:
        # return NaNs for all metrics
        keys = [
            "mean_AI",
            "sd_AI",
            "mean_NoAI",
            "sd_NoAI",
            "mean_diff",
            "sd_diff",
            "se_diff",
            "t",
            "df",
            "p_value",
            "CI_low",
            "CI_high",
            "cohen_dz",
            "Shapiro_W",
            "Shapiro_p",
        ]
        for k in keys:
            result[k] = np.nan
        return result

    diff = ai - noai
    mean_ai = float(np.mean(ai))
    mean_noai = float(np.mean(noai))
    sd_ai = float(np.std(ai, ddof=1)) if n > 1 else 0.0
    sd_noai = float(np.std(noai, ddof=1)) if n > 1 else 0.0
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_diff = float(sd_diff / np.sqrt(n)) if n > 0 else np.nan

    # paired t-test
    if n > 1:
        t_stat, p_val = stats.ttest_rel(ai, noai)
        dfree = n - 1
        ci_low, ci_high = t_ci_for_mean_diff(diff, alpha=0.05)
    else:
        t_stat, p_val, dfree, ci_low, ci_high = (np.nan, np.nan, np.nan, np.nan, np.nan)

    # effect size (paired): Cohen's dz
    cohen_dz = mean_diff / sd_diff if sd_diff > 0 else np.nan

    # Shapiro-Wilk for diffs
    try:
        if 3 <= n <= 5000:
            w, p = stats.shapiro(diff)
            sh_w, sh_p = float(w), float(p)
        else:
            sh_w, sh_p = (np.nan, np.nan)
    except Exception:
        sh_w, sh_p = (np.nan, np.nan)

    result.update(
        {
            "mean_AI": round(mean_ai, precision),
            "sd_AI": round(sd_ai, precision),
            "mean_NoAI": round(mean_noai, precision),
            "sd_NoAI": round(sd_noai, precision),
            "mean_diff": round(mean_diff, precision),
            "sd_diff": round(sd_diff, precision),
            "se_diff": round(se_diff, precision) if np.isfinite(se_diff) else np.nan,
            "t": round(float(t_stat), precision) if np.isfinite(t_stat) else np.nan,
            "df": int(dfree) if np.isfinite(dfree) else np.nan,
            "p_value": round(float(p_val), 4) if np.isfinite(p_val) else np.nan,
            "CI_low": round(float(ci_low), precision) if np.isfinite(ci_low) else np.nan,
            "CI_high": round(float(ci_high), precision) if np.isfinite(ci_high) else np.nan,
            "cohen_dz": round(float(cohen_dz), precision) if np.isfinite(cohen_dz) else np.nan,
            "Shapiro_W": round(sh_w, 3) if np.isfinite(sh_w) else np.nan,
            "Shapiro_p": round(sh_p, 4) if np.isfinite(sh_p) else np.nan,
        }
    )
    return result


def build_bar_plot(mean_diffs: Dict[str, Dict[str, float]]) -> bytes:
    labels = ["Posture", "Gaze"]
    means = [mean_diffs["posture"]["mean_diff"], mean_diffs["gaze"]["mean_diff"]]
    ci_lows = [mean_diffs["posture"]["CI_low"], mean_diffs["gaze"]["CI_low"]]
    ci_highs = [mean_diffs["posture"]["CI_high"], mean_diffs["gaze"]["CI_high"]]
    yerr = np.vstack([
        np.array(means) - np.array(ci_lows),
        np.array(ci_highs) - np.array(means),
    ])

    plt.figure(figsize=(5.5, 3.6))
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#F58518"]
    plt.bar(x, means, yerr=yerr, capsize=4, color=colors)
    plt.xticks(x, labels)
    plt.ylabel("Mean difference (AI − NoAI)")
    plt.title("Paired t-test: mean differences with 95% CI")
    plt.tight_layout()

    # export to PNG bytes
    import io

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    return buf.read()


def image_bytes_to_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    base_dir = os.path.dirname(__file__)
    input_csv = os.path.join(base_dir, "paired_participation_by_member_type.csv")
    out_dir = os.path.join(base_dir, "tests")
    report_path = os.path.join(base_dir, "paired_ttest_report.html")

    ensure_dir(out_dir)

    df = read_input(input_csv)

    # Compute per-type results
    results: Dict[str, Dict[str, float]] = {}
    results["posture"] = paired_t_for_type(df, "AI_Mean_Level_posture", "NoAI_Mean_Level_posture")
    results["gaze"] = paired_t_for_type(df, "AI_Mean_Level_gaze", "NoAI_Mean_Level_gaze")

    # Build CSV
    rows: List[Dict[str, float]] = []
    for tkey, res in results.items():
        row = {"Type": "Posture" if tkey == "posture" else "Gaze"}
        row.update(res)
        rows.append(row)
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "paired_ttest_results.csv")
    results_df.to_csv(csv_path, index=False)

    # Build a compact bar plot
    plot_png = build_bar_plot(results)
    img_uri = image_bytes_to_data_uri(plot_png)

    # HTML report (single file)
    html: List[str] = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang=\"en\">\n<head>")
    html.append("<meta charset=\"utf-8\">")
    html.append("<title>Paired t-test Report</title>")
    html.append(
        "<style>"
        "body{font-family:Arial,Helvetica,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;margin:24px;}"
        "h1,h2{margin:12px 0;} table{border-collapse:collapse;margin:12px 0;}"
        "table,th,td{border:1px solid #ccc;padding:6px 8px;} th{background:#f7f7f7;}"
        "img{max-width:100%;height:auto;border:1px solid #eee;margin:6px 0;}"
        "</style>\n</head>\n<body>"
    )
    html.append("<h1>Paired t-test Report</h1>")
    html.append(f"<p>Data source: {os.path.relpath(input_csv, start=base_dir)}</p>")

    # Summary cards (simple table)
    html.append("<h2>Results</h2>")
    display_cols = [
        "Type",
        "N",
        "mean_AI",
        "sd_AI",
        "mean_NoAI",
        "sd_NoAI",
        "mean_diff",
        "sd_diff",
        "t",
        "df",
        "p_value",
        "CI_low",
        "CI_high",
        "cohen_dz",
        "Shapiro_W",
        "Shapiro_p",
    ]
    html.append(results_df[display_cols].to_html(index=False))

    # Compact figure
    html.append("<h2>Mean differences with 95% CI</h2>")
    html.append(f"<img src=\"{img_uri}\" alt=\"mean differences\">")

    # Interpretation hints
    html.append(
        "<p>Two-sided paired t-tests were performed for Posture and Gaze. "
        "Shapiro–Wilk results are shown for the per-subject differences (AI − NoAI). "
        "If normality is violated in other datasets, consider using the Wilcoxon signed-rank test.</p>"
    )

    html.append("</body>\n</html>")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"Saved CSV: {csv_path}")
    print(f"Saved HTML report: {report_path}")


if __name__ == "__main__":
    main()


