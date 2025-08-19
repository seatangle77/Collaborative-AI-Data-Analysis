import argparse
import os
import re
from typing import List, Tuple

import pandas as pd


def extract_member_numeric_id(member_id: str) -> int:
    """Extract the numeric part from a MemberID like 'P12'.

    Returns 0 if no digits are found to keep sort stable.
    """
    match = re.search(r"(\d+)", str(member_id))
    return int(match.group(1)) if match else 0


def type_sort_key(type_value: str) -> int:
    """Return a deterministic sort key for Type with order: posture < gaze < others."""
    order = {"posture": 0, "gaze": 1}
    return order.get(str(type_value), 99)


def normalize_type_to_english(type_value: str) -> str:
    """Normalize Type to English labels.

    - 姿态 -> posture
    - 视线 -> gaze
    Leaves other values as-is (lowercased if ascii letters).
    """
    value = str(type_value).strip()
    mapping = {
        "姿态": "posture",
        "视线": "gaze",
        "posture": "posture",
        "gaze": "gaze",
    }
    if value in mapping:
        return mapping[value]
    # Fallback: keep original but lowercase basic ascii words
    lowered = value.lower()
    return lowered


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少必要列: {missing}")


def find_duplicate_rows(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    grouped_sizes = df.groupby(key_cols, dropna=False).size().reset_index(name="_count")
    duplicates = grouped_sizes[grouped_sizes["_count"] > 1]
    return duplicates


def aggregate_duplicates(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Aggregate duplicates by:
    - Total_Frames: sum
    - Mean_Level: weighted average by Total_Frames when possible; fallback to simple mean.
    """
    def weighted_mean(group: pd.DataFrame) -> float:
        if (group["Total_Frames"] > 0).all():
            return (group["Mean_Level"] * group["Total_Frames"]).sum() / group["Total_Frames"].sum()
        return group["Mean_Level"].mean()

    aggregations = {
        "Total_Frames": "sum",
        "Mean_Level": weighted_mean,
    }
    aggregated = df.groupby(key_cols, dropna=False, as_index=False).agg(aggregations)
    return aggregated


def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["Group", "MemberID", "Type"],
        columns="Condition",
        values=["Total_Frames", "Mean_Level"],
        aggfunc="first",
    )

    # Flatten columns like ("Mean_Level","AI") -> "AI_Mean_Level"
    pivot.columns = [f"{col_level2}_{col_level1}" for col_level1, col_level2 in pivot.columns]
    pivot = pivot.reset_index()

    # Ensure expected columns exist even if one condition is entirely missing
    for base in ("Total_Frames", "Mean_Level"):
        for cond in ("AI", "NoAI"):
            expected = f"{cond}_{base}"
            if expected not in pivot.columns:
                pivot[expected] = pd.NA

    # Reorder columns
    desired_order = [
        "Group",
        "MemberID",
        "Type",
        "AI_Total_Frames",
        "NoAI_Total_Frames",
        "AI_Mean_Level",
        "NoAI_Mean_Level",
    ]
    existing = [c for c in desired_order if c in pivot.columns]
    remaining = [c for c in pivot.columns if c not in existing]
    pivot = pivot[existing + remaining]
    return pivot


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Diff_Mean_Level"] = df["AI_Mean_Level"] - df["NoAI_Mean_Level"]
    # Avoid division by zero or NaN
    df["Rel_Change_Mean_Level"] = df["Diff_Mean_Level"] / df["NoAI_Mean_Level"]
    return df


def round_numeric_columns(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    df = df.copy()
    mean_cols = [c for c in df.columns if c.endswith("_Mean_Level") or c in ("Diff_Mean_Level", "Rel_Change_Mean_Level")]
    for col in mean_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(precision)
    # Cast Total_Frames to Int64 (nullable)
    for col in [c for c in df.columns if c.endswith("_Total_Frames")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def sort_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_member_num"] = df["MemberID"].map(extract_member_numeric_id)
    df["_type_order"] = df["Type"].map(type_sort_key)
    df = df.sort_values(["_member_num", "_type_order", "MemberID", "Type"], kind="stable")
    df = df.drop(columns=["_member_num", "_type_order"])
    return df


def pivot_type_wider(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Pivot Type values into columns so each MemberID occupies one row.

    Column naming strategy: keep metric names first, append Type as suffix, e.g.,
    AI_Mean_Level_posture, NoAI_Mean_Level_posture, AI_Mean_Level_gaze, NoAI_Mean_Level_gaze, ...
    Includes any derived columns if present.
    """
    if df_wide.empty:
        return df_wide

    base_key = ["Group", "MemberID"]
    type_values = list(df_wide["Type"].dropna().unique())

    # Stable and meaningful ordering of Type
    type_values.sort(key=type_sort_key)

    # Determine metric columns (everything except keys)
    metric_columns = [c for c in df_wide.columns if c not in (base_key + ["Type"]) ]

    result = df_wide[base_key].drop_duplicates().copy()
    for type_value in type_values:
        subset = df_wide[df_wide["Type"] == type_value][base_key + metric_columns].copy()
        # Rename metric columns to append type suffix
        rename_map = {c: f"{c}_{type_value}" for c in metric_columns}
        subset = subset.rename(columns=rename_map)
        result = result.merge(subset, on=base_key, how="left")

    # Sort rows by member id numeric
    result["_member_num"] = result["MemberID"].map(extract_member_numeric_id)
    result = result.sort_values(["_member_num", "MemberID"], kind="stable").drop(columns=["_member_num"]) 
    return result


def export_long_table(df_wide: pd.DataFrame) -> pd.DataFrame:
    # Build long table from wide columns
    base_cols = ["Group", "MemberID", "Type"]
    ai_subset = df_wide[base_cols + ["AI_Total_Frames", "AI_Mean_Level"]].copy()
    ai_subset = ai_subset.rename(columns={"AI_Total_Frames": "Total_Frames", "AI_Mean_Level": "Mean_Level"})
    ai_subset["Condition"] = "AI"

    noai_subset = df_wide[base_cols + ["NoAI_Total_Frames", "NoAI_Mean_Level"]].copy()
    noai_subset = noai_subset.rename(columns={"NoAI_Total_Frames": "Total_Frames", "NoAI_Mean_Level": "Mean_Level"})
    noai_subset["Condition"] = "NoAI"

    long_df = pd.concat([ai_subset, noai_subset], ignore_index=True)
    # Keep rows where at least Mean_Level is present
    long_df = long_df[long_df["Mean_Level"].notna()].reset_index(drop=True)
    # Sort similarly
    long_df["_member_num"] = long_df["MemberID"].map(extract_member_numeric_id)
    long_df["_type_order"] = long_df["Type"].map(type_sort_key)
    long_df = long_df.sort_values(["_member_num", "Type", "Condition"], kind="stable")
    long_df = long_df.drop(columns=["_member_num", "_type_order"], errors="ignore")
    return long_df


def main():
    parser = argparse.ArgumentParser(description="按 MemberID × Type 配对（AI vs NoAI）并生成宽表/长表")
    default_input = os.path.join(os.path.dirname(__file__), "video_annotation_participation_sorted.csv")
    default_output_wide = os.path.join(os.path.dirname(__file__), "paired_participation_by_member_type.csv")
    default_quality_report = os.path.join(os.path.dirname(__file__), "paired_participation_quality_report.csv")
    default_output_long = os.path.join(os.path.dirname(__file__), "paired_participation_by_member_type_long.csv")

    parser.add_argument("--input", type=str, default=default_input, help="输入CSV路径（默认：video_annotation_participation_sorted.csv）")
    parser.add_argument("--output-wide", type=str, default=default_output_wide, help="输出宽表CSV路径")
    parser.add_argument("--export-long", action="store_true", help="同时导出长表")
    parser.add_argument("--output-long", type=str, default=default_output_long, help="长表输出路径（需配合 --export-long）")
    parser.add_argument("--quality-report", action="store_true", help="导出数据质量报告CSV")
    parser.add_argument("--quality-report-path", type=str, default=default_quality_report, help="质量报告输出路径")
    parser.add_argument("--keep-incomplete", action="store_true", help="保留不完整配对（默认丢弃）")
    parser.add_argument("--with-diff", action="store_true", help="生成差值与相对变化列")
    parser.add_argument("--aggregate-duplicates", action="store_true", help="若存在重复键，按加权规则聚合；否则报错/报告")
    parser.add_argument("--precision", type=int, default=3, help="小数精度（默认3）")
    parser.add_argument("--pivot-type", action="store_true", help="按 Type 也展开为列（每个 MemberID 一行，超宽表）")

    args = parser.parse_args()

    required_columns = [
        "Group",
        "MemberID",
        "Condition",
        "Type",
        "Total_Frames",
        "Mean_Level",
    ]

    df = pd.read_csv(args.input)
    validate_required_columns(df, required_columns)

    # 仅保留必要列，忽略 Session
    df = df[required_columns].copy()

    # 规范 Condition 值
    df["Condition"] = df["Condition"].astype(str)

    # 标准化 Type 到英文（posture/gaze）
    df["Type"] = df["Type"].map(normalize_type_to_english)

    # 查找重复：同一 Group + MemberID + Type + Condition
    key_cols = ["Group", "MemberID", "Type", "Condition"]
    duplicates = find_duplicate_rows(df, key_cols)

    if not duplicates.empty:
        if args.quality_report:
            os.makedirs(os.path.dirname(args.quality_report_path), exist_ok=True)
            duplicates.to_csv(args.quality_report_path, index=False)
        if not args.aggregate_duplicates:
            raise ValueError(
                "检测到重复键（Group + MemberID + Type + Condition）。" \
                "可使用 --aggregate-duplicates 聚合，或 --quality-report 输出报告后清洗数据。"
            )
        # 聚合重复
        df = aggregate_duplicates(df, key_cols)

    # 透视成宽表
    wide = pivot_to_wide(df)

    # 严格配对：默认丢弃任何一边缺失的行
    if not args.keep_incomplete:
        wide = wide[wide[["AI_Mean_Level", "NoAI_Mean_Level"]].notna().all(axis=1)].reset_index(drop=True)

    # 衍生列
    if args.with_diff:
        wide = add_derived_columns(wide)

    # 数字格式
    wide = round_numeric_columns(wide, precision=args.precision)

    # 排序
    wide = sort_wide_table(wide)

    # 可选：按 Type 展开为更宽的表
    if args.pivot_type:
        wide_by_member = pivot_type_wider(wide)
        os.makedirs(os.path.dirname(args.output_wide), exist_ok=True)
        wide_by_member.to_csv(args.output_wide, index=False)
    else:
        # 输出宽表（每行=MemberID×Type）
        os.makedirs(os.path.dirname(args.output_wide), exist_ok=True)
        wide.to_csv(args.output_wide, index=False)

    # 可选导出长表
    if args.export_long:
        long_df = export_long_table(wide)
        long_df = long_df.copy()
        # 长表保留原始精度的 Mean_Level 已被 round_numeric_columns 处理，无需再处理
        os.makedirs(os.path.dirname(args.output_long), exist_ok=True)
        long_df.to_csv(args.output_long, index=False)

    # 可选质量报告：若没有重复但需要报告，也输出缺配对信息
    if args.quality_report and duplicates.empty:
        # 找缺配对：对于 Group+MemberID+Type，检查两种 Condition 是否齐全
        completeness = df.groupby(["Group", "MemberID", "Type"])\
            ["Condition"].nunique().reset_index(name="num_conditions")
        missing_pairs = completeness[completeness["num_conditions"] < 2].copy()
        os.makedirs(os.path.dirname(args.quality_report_path), exist_ok=True)
        missing_pairs.to_csv(args.quality_report_path, index=False)

    print(f"已生成宽表: {args.output_wide}")
    if args.export_long:
        print(f"已生成长表: {args.output_long}")
    if args.quality_report:
        print(f"已生成质量报告: {args.quality_report_path}")


if __name__ == "__main__":
    main()


