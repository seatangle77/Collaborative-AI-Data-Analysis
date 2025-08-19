#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
笔记编辑数据预处理脚本

将原始的10秒间隔笔记编辑数据转换为成员级别的汇总数据，
用于后续的参与度（Engagement）和平衡度（Gini系数）分析。

输入：all_groups_note_edit_history_by_10s.csv
输出：note_edit_summary_by_member.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_and_clean_data(file_path):
    """
    加载并清洗原始数据
    
    Args:
        file_path (str): CSV文件路径
        
    Returns:
        pd.DataFrame: 清洗后的数据框
    """
    print("正在加载数据...")
    
    # 读取CSV文件，跳过第0行（说明行）
    df = pd.read_csv(file_path, skiprows=1)
    
    print(f"原始数据形状: {df.shape}")
    print(f"列数: {len(df.columns)}")
    
    # 检查数据结构
    print(f"前几列: {list(df.columns[:10])}")
    
    return df

def extract_time_windows(df):
    """
    提取时间窗口信息并识别字符数和编辑次数列
    
    Args:
        df (pd.DataFrame): 原始数据框
        
    Returns:
        tuple: (char_cols, edit_cols, time_windows)
    """
    print("正在提取时间窗口信息...")
    
    # 获取所有列名
    all_cols = df.columns.tolist()
    
    # 识别字符数列和编辑次数列
    char_cols = [col for col in all_cols if col.startswith('note_edit_chars_')]
    edit_cols = [col for col in all_cols if col.startswith('note_edit_count_')]
    
    # 提取时间窗口信息
    time_windows = []
    for col in char_cols:
        time_window = col.replace('note_edit_chars_', '')
        time_windows.append(time_window)
    
    print(f"找到 {len(char_cols)} 个时间窗口")
    print(f"时间窗口示例: {time_windows[:5]}")
    
    return char_cols, edit_cols, time_windows

def calculate_member_summary(df, char_cols, edit_cols):
    """
    计算每个成员的汇总数据
    
    Args:
        df (pd.DataFrame): 原始数据框
        char_cols (list): 字符数列名列表
        edit_cols (list): 编辑次数列名列表
        
    Returns:
        pd.DataFrame: 成员级别汇总数据
    """
    print("正在计算成员级别汇总数据...")
    
    # 创建结果列表
    results = []
    
    # 按Group, Speaker, Type分组处理
    for (group, speaker, exp_type), group_data in df.groupby(['Group', 'Speaker', 'Type']):
        
        # 计算该成员的汇总指标
        net_chars = 0
        total_edit_count = 0
        active_time_windows = 0
        
        # 遍历每个时间窗口
        for char_col, edit_col in zip(char_cols, edit_cols):
            # 获取字符数（处理空值）
            char_value = group_data[char_col].iloc[0]
            if pd.notna(char_value) and str(char_value).strip() != '':
                try:
                    char_value = float(char_value)
                    net_chars += char_value
                except (ValueError, TypeError):
                    char_value = 0
            
            # 获取编辑次数（处理空值）
            edit_value = group_data[edit_col].iloc[0]
            if pd.notna(edit_value) and str(edit_value).strip() != '':
                try:
                    edit_value = int(float(edit_value))
                    total_edit_count += edit_value
                    if edit_value > 0:
                        active_time_windows += 1
                except (ValueError, TypeError):
                    edit_value = 0
        
        # 添加到结果列表
        results.append({
            'Group': group,
            'Speaker': speaker,
            'Type': exp_type,
            'Net_Chars': net_chars,
            'Total_Edit_Count': total_edit_count,
            'Active_Time_Windows': active_time_windows
        })
    
    # 转换为DataFrame
    summary_df = pd.DataFrame(results)
    
    print(f"汇总数据形状: {summary_df.shape}")
    
    return summary_df

def save_results(summary_df, output_path):
    """
    保存结果到CSV文件
    
    Args:
        summary_df (pd.DataFrame): 汇总数据框
        output_path (Path): 输出文件路径
    """
    print(f"正在保存结果到: {output_path}")
    
    # 保存为CSV
    summary_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 同时保存为Excel格式（可选）
    excel_path = output_path.with_suffix('.xlsx')
    summary_df.to_excel(excel_path, index=False, engine='openpyxl')
    
    print(f"结果已保存到:")
    print(f"  CSV: {output_path}")
    print(f"  Excel: {excel_path}")

def generate_summary_report(summary_df):
    """
    生成数据摘要报告
    
    Args:
        summary_df (pd.DataFrame): 汇总数据框
    """
    print("\n" + "="*50)
    print("数据预处理摘要报告")
    print("="*50)
    
    print(f"总记录数: {len(summary_df)}")
    print(f"小组数: {summary_df['Group'].nunique()}")
    print(f"实验条件: {summary_df['Type'].unique()}")
    
    print("\n按实验条件统计:")
    type_stats = summary_df.groupby('Type').agg({
        'Net_Chars': ['count', 'mean', 'std', 'min', 'max'],
        'Total_Edit_Count': ['mean', 'std'],
        'Active_Time_Windows': ['mean', 'std']
    }).round(2)
    print(type_stats)
    
    print("\n按小组统计:")
    group_stats = summary_df.groupby('Group').agg({
        'Net_Chars': ['sum', 'mean', 'std'],
        'Total_Edit_Count': ['sum', 'mean'],
        'Active_Time_Windows': ['sum', 'mean']
    }).round(2)
    print(group_stats.head(10))  # 只显示前10个小组
    
    print("\n数据质量检查:")
    print(f"Net_Chars 缺失值: {summary_df['Net_Chars'].isnull().sum()}")
    print(f"Total_Edit_Count 缺失值: {summary_df['Total_Edit_Count'].isnull().sum()}")
    print(f"Active_Time_Windows 缺失值: {summary_df['Active_Time_Windows'].isnull().sum()}")

def main():
    """主函数"""
    print("笔记编辑数据预处理脚本")
    print("="*50)
    
    # 设置文件路径
    current_dir = Path(__file__).parent
    input_file = current_dir / "all_groups_note_edit_history_by_10s.csv"
    output_file = current_dir / "note_edit_summary_by_member.csv"
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    try:
        # 1. 加载和清洗数据
        df = load_and_clean_data(input_file)
        
        # 2. 提取时间窗口信息
        char_cols, edit_cols, time_windows = extract_time_windows(df)
        
        # 3. 计算成员级别汇总数据
        summary_df = calculate_member_summary(df, char_cols, edit_cols)
        
        # 4. 保存结果
        save_results(summary_df, output_file)
        
        # 5. 生成摘要报告
        generate_summary_report(summary_df)
        
        print("\n数据预处理完成！")
        print(f"输出文件: {output_file}")
        
        # 显示前几行结果
        print("\n前5行结果预览:")
        print(summary_df.head())
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
