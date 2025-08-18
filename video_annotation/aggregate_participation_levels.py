# -*- coding: utf-8 -*-
"""
aggregate_participation_levels.py

功能：
1) 读取原始标注 Excel（每个 sheet = 一个小组 G0~G11）
2) 提取每位成员在两次任务 (-1 / -2) 下：
   - 姿态数值行的平均参与度 (1/2/3)
   - 视线数值行的平均参与度 (1/2/3)
   - 有效帧数（该数值行的非空数目）
3) 根据内置的组别映射（AI / NoAI）将 Gx1/Gx2 归到 Condition
4) 为每个小组的本地成员编号（P1, P2, P3 ...）分配全局 MemberID（P1~P36）
5) 输出汇总 CSV：video_annotation_participation.csv

依赖：
  pip install pandas openpyxl
用法：
  python aggregate_participation_levels.py /path/to/posture_gaze_behavior.xlsx
"""

import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd


# === 组内对照映射（内置，不依赖外部 JSON）==============================
# 含义：每个组（sheet）下面哪一次是 AI，哪一次是 NoAI。
# 这里的键是组（sheet）名，值是 { "AI": <group_name>, "NoAI": <group_name> }
# 其中 group_name 采用你要求的格式：G01/G02, G11/G12, ... （而非 Session1/2）
GROUP_CONDITION_MAP: Dict[str, Dict[str, str]] = {
    "G0": {"AI": "G02", "NoAI": "G01"},
    "G1": {"AI": "G11", "NoAI": "G12"},
    "G2": {"AI": "G22", "NoAI": "G21"},
    "G3": {"AI": "G31", "NoAI": "G32"},
    "G4": {"AI": "G42", "NoAI": "G41"},
    "G5": {"AI": "G51", "NoAI": "G52"},
    "G6": {"AI": "G62", "NoAI": "G61"},
    "G7": {"AI": "G71", "NoAI": "G72"},
    "G8": {"AI": "G82", "NoAI": "G81"},
    "G9": {"AI": "G91", "NoAI": "G92"},
    "Ga": {"AI": "Ga2", "NoAI": "Ga1"},
    "Gb": {"AI": "Gb1", "NoAI": "Gb2"},
}


# 调试开关：True 时打印详细过程信息
DEBUG: bool = True


def make_group_name(sheet: str, session_num: str) -> str:
    """
    统一生成 group_name：
    - 常规：G0~G9 → f"{sheet}{session_num}"，如 G0+1 → G01
    - 特例：G10 → Ga{session_num}；G11 → Gb{session_num}
      这是为兼容最后两张表实际使用的 Ga1/Ga2、Gb1/Gb2 标记。
    """
    if sheet == "G10":
        return f"Ga{session_num}"
    if sheet == "G11":
        return f"Gb{session_num}"
    return f"{sheet}{session_num}"


def is_numeric_row(row: pd.Series) -> bool:
    """
    判断该标注行是否为'数值行'（1/2/3 参与度），而不是文本行（T1./Z1. ...）。
    策略：尝试将第2列开始的值转为数值，若有较多（>=3）有效数字，则认为是数值行。
    """
    nums = pd.to_numeric(row.iloc[1:], errors="coerce")
    # 放宽阈值：至少有 1 个有效数字即可视为数值行
    return nums.notna().sum() >= 1


def extract_numeric_mean_and_frames(row: pd.Series) -> Tuple[float, int]:
    """
    从数值行提取平均参与度 & 有效帧数（非空数值的个数）。
    """
    vals = pd.to_numeric(row.iloc[1:], errors="coerce").dropna()
    if len(vals) == 0:
        return float("nan"), 0
    return float(vals.mean()), int(vals.count())


def main(xlsx_path: Path) -> None:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"找不到文件：{xlsx_path}")

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    if DEBUG:
        print("读取到的 sheet 列表:", list(xls.sheet_names))
    # 全局成员编号（P1~P36）分配计数器
    global_member_counter = 1
    # 每个组一个本地->全局的映射：{"P1": "P1", "P2": "P2", ...}
    group_local_to_global: Dict[str, Dict[str, str]] = {}

    records: List[Dict[str, Any]] = []

    # 统计每个 sheet 写入的记录数，便于排查缺失
    records_per_sheet: Dict[str, int] = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=object, engine="openpyxl")
        if df.empty:
            continue

        # 标准化第一列列名
        df = df.rename(columns={df.columns[0]: "label"})
        # 丢掉空 label
        df = df[~df["label"].isna()].copy()
        if DEBUG:
            print(f"—— 处理 sheet: {sheet}，非空 label 行数: {len(df)}")

        # 初始化本组的本地->全局成员映射
        group_local_to_global[sheet] = {}
        local_ids_in_order: List[str] = []

        # 先扫描出本组出现过的所有本地参与者编号（按数字顺序）
        # 匹配形如 "P1-1姿态"/"P1-1视线"/"P2-2姿态" 等，允许空格与全角连字符
        pat_all = re.compile(r"^(P(\d+))\s*[-－]\s*(\d)\s*(姿态|视线)\s*$")
        for _, row in df.iterrows():
            label = str(row["label"]).strip()
            m = pat_all.match(label)
            if not m:
                continue
            local_pid = m.group(1)   # 如 "P1"
            if local_pid not in local_ids_in_order:
                local_ids_in_order.append(local_pid)

        # 将本地 P1/P2/P3... 映射为全局 P1~P36（按组顺序、组内顺序依次分配）
        for _local in sorted(local_ids_in_order, key=lambda x: int(x[1:])):
            group_local_to_global[sheet][_local] = f"P{global_member_counter}"
            global_member_counter += 1
        if DEBUG:
            print(f"   本组识别到的本地参与者: {sorted(local_ids_in_order, key=lambda x: int(x[1:]))}")
            print(f"   已分配的全局 MemberID 数: {len(group_local_to_global[sheet])}")

        # 正则：仅处理'数值行'，允许空格与全角连字符
        pat_numeric = re.compile(r"^(P(\d+))\s*[-－]\s*(\d)\s*(姿态|视线)\s*$")

        # 统计匹配情况，帮助排查缺失
        unmatched_examples: List[str] = []
        matched_label_count: int = 0
        numeric_rows_count: int = 0

        for _, row in df.iterrows():
            label = str(row["label"]).strip()
            m = pat_numeric.match(label)
            if not m:
                if len(unmatched_examples) < 5:
                    unmatched_examples.append(label)
                continue
            matched_label_count += 1

            # 过滤：只处理数值行
            if not is_numeric_row(row):
                continue
            numeric_rows_count += 1

            local_pid = m.group(1)      # e.g., "P1"
            session_num = m.group(3)    # "1" or "2"
            typ = m.group(4)            # "姿态" or "视线"

            # 数值平均 & 帧数
            mean_level, total_frames = extract_numeric_mean_and_frames(row)

            # group_name: 统一由 make_group_name 生成，处理 G10/G11 特例
            group_name = make_group_name(sheet, session_num)

            # Condition: 由映射表决定
            cond = None
            mapping = GROUP_CONDITION_MAP.get(sheet, {})
            for k, v in mapping.items():
                if v == group_name:
                    cond = k
                    break
            if DEBUG and cond is None:
                print(f"   [WARN] 无法从映射表判定 Condition: sheet={sheet}, group_name={group_name}, 映射={mapping}")

            # 全局成员 ID
            member_global = group_local_to_global[sheet].get(local_pid, None)
            if DEBUG and member_global is None:
                print(f"   [WARN] 未找到全局 MemberID: sheet={sheet}, local_pid={local_pid}")

            records.append({
                "Group": sheet,
                "group_name": group_name,
                "Participant": f"{local_pid}-{session_num}",  # 保留原始局部编号+会次
                "MemberID": member_global,                    # 全局 P1~P36
                "Session": int(session_num),
                "Condition": cond,                            # "AI" / "NoAI"
                "Type": typ,                                  # "姿态" / "视线"
                "Total_Frames": total_frames,
                "Mean_Level": mean_level
            })

            records_per_sheet[sheet] = records_per_sheet.get(sheet, 0) + 1

        if DEBUG:
            print(f"   行匹配统计：匹配到标签={matched_label_count}，数值行={numeric_rows_count}，写入记录={records_per_sheet.get(sheet, 0)}")
            if unmatched_examples:
                print(f"   未匹配示例(最多5条)：{unmatched_examples}")

    if not records:
        print("未从表格中提取到有效的数值行（1/2/3 参与度）。请检查 Excel 结构。")
        return

    df_long = pd.DataFrame.from_records(records)

    # 恢复稳定的宽表（不在这里横向拼 AI/NoAI），避免层级合并问题
    df_wide = df_long.pivot_table(
        index=["Group", "group_name", "Participant", "MemberID", "Condition"],
        columns="Type",
        values=["Total_Frames", "Mean_Level"],
        aggfunc="first"
    ).reset_index()

    # 展平多级列名
    df_wide.columns = ["_".join(col).strip("_") for col in df_wide.columns.values]

    # 友好列名
    rename_map = {
        "Group_": "Group",
        "group_name_": "group_name",
        "Participant_": "Participant",
        "MemberID_": "MemberID",
        "Condition_": "Condition",
        "Total_Frames_姿态": "Total_Frames_Posture",
        "Total_Frames_视线": "Total_Frames_Gaze",
        "Mean_Level_姿态": "Posture_Mean",
        "Mean_Level_视线": "Gaze_Mean",
    }
    df_wide = df_wide.rename(columns=rename_map)

    # 排序更友好
    sort_cols = ["Group", "MemberID", "group_name"]
    for c in sort_cols:
        if c not in df_wide.columns:
            sort_cols.remove(c)
    df_wide = df_wide.sort_values(sort_cols).reset_index(drop=True)

    out_csv = xlsx_path.parent / "video_annotation_participation.csv"
    df_wide.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("✅ 已输出：", out_csv)
    # 打印各 sheet 计数，便于定位哪些 sheet 未被识别
    if records_per_sheet:
        print("每个 sheet 的记录数：")
        for _s, _n in records_per_sheet.items():
            print(f"  - { _s }: { _n }")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df_wide.head())

    # 额外输出：仅重排行次序的“长表”，让 AI/NoAI 紧邻（不改变字段结构）
    try:
        df_long_sorted = df_long.copy()
        if "Condition" in df_long_sorted.columns:
            df_long_sorted["Condition"] = pd.Categorical(
                df_long_sorted["Condition"], categories=["AI", "NoAI"], ordered=True
            )
        sort_keys = [c for c in ["Group", "MemberID", "Participant", "Type", "Condition"] if c in df_long_sorted.columns]
        df_long_sorted = df_long_sorted.sort_values(sort_keys).reset_index(drop=True)
        out_csv_sorted = xlsx_path.parent / "video_annotation_participation_sorted.csv"
        df_long_sorted.to_csv(out_csv_sorted, index=False, encoding="utf-8-sig")
        print("✅ 另存长表（仅重排行次序，AI/NoAI 紧邻）：", out_csv_sorted)
    except Exception as _e:
        print("[WARN] 生成长表排序版失败：", _e)


if __name__ == "__main__":
    # 支持：
    # 1) 直接运行（无参数）：默认读取脚本同目录的 posture_gaze_behavior.xlsx
    # 2) 提供一个参数：自定义 Excel 路径
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path(__file__).parent / "posture_gaze_behavior.xlsx"
        print(f"未提供路径，默认读取：{input_path}")

    try:
        main(input_path)
    except FileNotFoundError as e:
        print(f"错误：{e}")
        print("用法：python aggregate_participation_levels.py /path/to/posture_gaze_behavior.xlsx")
        sys.exit(1)