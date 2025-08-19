#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发言指标计算器
计算每个小组的发言总时长、发言次数（段落数）和发言均衡性（Gini系数）
"""

import pandas as pd
import numpy as np
# from scipy.stats import gini  # 某些版本可能没有这个函数
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(file_path):
    """
    读取并清理数据
    
    Args:
        file_path (str): CSV文件路径
        
    Returns:
        pd.DataFrame: 清理后的数据
    """
    print("正在读取数据...")
    
    # 读取CSV文件，跳过第一行（标题行有问题）
    df = pd.read_csv(file_path, skiprows=1)
    
    # 检查数据完整性
    if df.empty:
        raise ValueError("数据文件为空")
    
    # 检查必要的列
    required_cols = ['group_key', 'speaker', 'type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 识别时间窗格列（排除前3列）
    time_cols = [col for col in df.columns if col not in required_cols]
    
    if not time_cols:
        raise ValueError("未找到时间窗格列")
    
    print(f"数据读取完成，共 {len(df)} 行，{len(time_cols)} 个时间窗格")
    
    return df, time_cols


def wide_to_long_format(df, time_cols):
    """
    将宽表转换为长表格式
    
    Args:
        df (pd.DataFrame): 原始宽表数据
        time_cols (list): 时间窗格列名列表
        
    Returns:
        pd.DataFrame: 长表格式数据
    """
    print("正在转换数据格式...")
    
    # 使用melt将宽表转长表
    long_df = df.melt(
        id_vars=['group_key', 'speaker', 'type'],
        value_vars=time_cols,
        var_name='time_window',
        value_name='speak_time'
    )
    
    # 确保speak_time为数值类型
    long_df['speak_time'] = pd.to_numeric(long_df['speak_time'], errors='coerce').fillna(0)
    
    # 按组别、条件、发言者、时间窗格排序
    long_df = long_df.sort_values(['group_key', 'type', 'speaker', 'time_window'])
    
    print(f"数据转换完成，长表共 {len(long_df)} 行")
    
    return long_df


def identify_speech_segments(df):
    """
    识别发言段落（连续发言的时间窗格）
    
    Args:
        df (pd.DataFrame): 长表格式数据
        
    Returns:
        pd.DataFrame: 包含段落信息的数据
    """
    print("正在识别发言段落...")
    
    # 标记有发言的时间窗格
    df['has_speech'] = df['speak_time'] > 0
    
    # 按组别、条件、发言者分组，识别连续段落
    segment_data = []
    
    for (group, condition, speaker), group_data in df.groupby(['group_key', 'type', 'speaker']):
        # 重置索引以便使用shift
        group_data = group_data.reset_index(drop=True)
        
        # 识别段落边界：从无发言变为有发言时开始新段落
        group_data['segment_start'] = (
            (group_data['has_speech'] == True) & 
            (group_data['has_speech'].shift(1) == False)
        )
        
        # 第一个有发言的窗格也是段落开始
        if len(group_data) > 0 and group_data['has_speech'].iloc[0]:
            group_data.loc[0, 'segment_start'] = True
        
        # 计算段落ID
        group_data['segment_id'] = group_data['segment_start'].cumsum()
        
        # 只保留有发言的窗格
        speech_windows = group_data[group_data['has_speech']]
        
        if not speech_windows.empty:
            # 计算段落数
            num_segments = speech_windows['segment_id'].nunique()
            
            segment_data.append({
                'group_key': group,
                'type': condition,
                'speaker': speaker,
                'num_segments': num_segments,
                'total_speech_time': group_data['speak_time'].sum()
            })
    
    segment_df = pd.DataFrame(segment_data)
    
    print(f"段落识别完成，共识别 {len(segment_df)} 个发言者")
    
    return segment_df


def calculate_gini_coefficient(speaker_totals):
    """
    计算发言均衡性的Gini系数
    
    Args:
        speaker_totals (list): 各发言者的总发言时长列表
        
    Returns:
        float: Gini系数
    """
    if not speaker_totals or len(speaker_totals) == 0:
        return 0.0
    
    # 过滤掉0值
    non_zero_totals = [t for t in speaker_totals if t > 0]
    
    if len(non_zero_totals) == 0:
        return 0.0
    
    if len(non_zero_totals) == 1:
        return 0.0
    
    # 如果所有值相等，Gini系数为0
    if len(set(non_zero_totals)) == 1:
        return 0.0
    
    # 直接使用自定义实现，因为某些scipy版本可能没有gini函数
    return custom_gini(non_zero_totals)


def custom_gini(values):
    """
    自定义Gini系数计算（备用方案）
    
    Args:
        values (list): 数值列表
        
    Returns:
        float: Gini系数
    """
    if len(values) == 0:
        return 0.0
    
    values = np.array(values)
    n = len(values)
    
    if n == 1:
        return 0.0
    
    # 计算Gini系数
    sorted_values = np.sort(values)
    cumsum = np.cumsum(sorted_values)
    
    # Gini = (n+1-2*sum((n+1-i)*yi))/n
    gini_coef = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_values / cumsum[-1])) / n
    
    return gini_coef


def aggregate_metrics(segment_df):
    """
    聚合计算最终指标
    
    Args:
        segment_df (pd.DataFrame): 包含段落信息的数据
        
    Returns:
        pd.DataFrame: 最终结果
    """
    print("正在计算最终指标...")
    
    results = []
    
    for (group, condition), group_data in segment_df.groupby(['group_key', 'type']):
        # 发言总时长
        total_speech_time = group_data['total_speech_time'].sum()
        
        # 发言段落数
        total_segments = group_data['num_segments'].sum()
        
        # 发言均衡性（Gini系数）
        speaker_totals = group_data['total_speech_time'].tolist()
        gini_coef = calculate_gini_coefficient(speaker_totals)
        
        results.append({
            'group_id': group,
            'condition': condition,
            'speak_total_s': round(total_speech_time, 2),
            'speak_count': total_segments,
            'gini_speak': round(gini_coef, 3)
        })
    
    result_df = pd.DataFrame(results)
    
    # 按组别排序
    result_df = result_df.sort_values('group_id')
    
    print(f"指标计算完成，共 {len(result_df)} 个组别")
    
    return result_df


def main():
    """
    主函数
    """
    try:
        # 文件路径
        input_file = "speech/all_groups_speaker_time_by_10s.csv"
        output_file = "speech/speech_metrics_results.csv"
        
        print("=" * 50)
        print("发言指标计算器")
        print("=" * 50)
        
        # 1. 读取和清理数据
        df, time_cols = load_and_clean_data(input_file)
        
        # 2. 转换为长表格式
        long_df = wide_to_long_format(df, time_cols)
        
        # 3. 识别发言段落
        segment_df = identify_speech_segments(long_df)
        
        # 4. 聚合计算指标
        result_df = aggregate_metrics(segment_df)
        
        # 5. 保存结果
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n结果已保存到: {output_file}")
        print("\n计算结果预览:")
        print(result_df.to_string(index=False))
        
        # 6. 输出统计信息
        print(f"\n统计信息:")
        print(f"- 总组别数: {len(result_df)}")
        print(f"- NoAI组别数: {len(result_df[result_df['condition'] == 'NoAI'])}")
        print(f"- AI组别数: {len(result_df[result_df['condition'] == 'AI'])}")
        
        # 按条件统计平均值
        if len(result_df) > 0:
            print(f"\n按条件统计平均值:")
            for condition in ['NoAI', 'AI']:
                condition_data = result_df[result_df['condition'] == condition]
                if len(condition_data) > 0:
                    print(f"{condition}:")
                    print(f"  平均发言时长: {condition_data['speak_total_s'].mean():.2f}秒")
                    print(f"  平均段落数: {condition_data['speak_count'].mean():.1f}")
                    print(f"  平均Gini系数: {condition_data['gini_speak'].mean():.3f}")
        
        print("\n处理完成！")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
