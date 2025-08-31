#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参与度水平分析脚本 - 6个维度2×3面板完整版
Participation Levels Analyzer - 6 Dimensions 2×3 Panels Complete Version

功能：
1. 按维度独立配对筛选，计算Mean, SE, N
2. 生成2×3小面板分组柱状图
3. 输出integrated_levels_summary.csv和paired_tests_per_metric.csv

作者：AI Assistant
版本：2.0
日期：2024
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_rel
import warnings
from plot_and_output import create_2x3_subplot_figure, output_summary_files

# 添加viz模块路径
sys.path.append(str(Path(__file__).parent.parent))
from viz.style_kit import apply_style

# 应用统一样式
apply_style()
warnings.filterwarnings('ignore')

class ParticipationLevelsAnalyzer:
    """参与度水平分析器主类"""
    
    def __init__(self, project_root: str = None):
        """初始化分析器"""
        if project_root is None:
            current_dir = Path(__file__).parent
            self.project_root = current_dir.parent
        else:
            self.project_root = Path(project_root)
            
        # 设置数据文件路径
        self.data_paths = {
            'speech': self.project_root / 'speech' / 'paired_speech_data.csv',
            'edit': self.project_root / 'edit_note' / 'engagement_analysis.csv',
            'browsing': self.project_root / 'page_behavior' / 'analysis' / 'group_paired_results.csv',
            'video': self.project_root / 'video_annotation' / 'paired_participation_by_member_type.csv'
        }
        
        # 设置输出目录
        self.output_dir = Path(__file__).parent
        self.figures_dir = self.output_dir / 'figures'
        self.data_dir = self.output_dir / 'data'
        
        # 创建输出目录
        for dir_path in [self.figures_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # 定义6个维度及其属性
        self.metrics = [
            'Speaking_Duration',
            'Speaking_Frequency', 
            'Editing_Activity',
            'Browsing_Activity',
            'Posture_Score',
            'Gaze_Score'
        ]
        
        # 维度标签和单位
        self.metric_info = {
            'Speaking_Duration': {'label': 'Speaking Duration', 'unit': 'sec/min'},
            'Speaking_Frequency': {'label': 'Speaking Frequency', 'unit': 'count/min'},
            'Editing_Activity': {'label': 'Editing Activity', 'unit': 'chars/min'},
            'Browsing_Activity': {'label': 'Browsing Activity', 'unit': 'ops/min'},
            'Posture_Score': {'label': 'Posture Score', 'unit': '1-5'},
            'Gaze_Score': {'label': 'Gaze Score', 'unit': '1-5'}
        }
        
        # 颜色配置
        self.colors = {
            'Baseline': '#64B5F6',
            'EngageCue': '#F48FB1'
        }
        
        print("=== 参与度水平分析器初始化完成 ===")
        print(f"项目根目录: {self.project_root}")
        print(f"输出目录: {self.output_dir}")

    def load_speaking_data(self, minutes: int) -> pd.DataFrame:
        """加载发言数据"""
        print("正在加载发言数据...")
        
        if not self.data_paths['speech'].exists():
            print(f"✗ 发言数据文件不存在: {self.data_paths['speech']}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.data_paths['speech'])
        print(f"✓ 发言数据加载成功: {df.shape}")
        
        # 处理发言时长和次数，除以任务时长
        results = []
        
        for _, row in df.iterrows():
            group = row['group_id']
            
            # Speaking Duration (sec/min)
            baseline_duration = row['noai_total'] / minutes
            engagecue_duration = row['ai_total'] / minutes
            
            results.extend([
                {'Metric': 'Speaking_Duration', 'Condition': 'Baseline', 'Group': group, 'Value': baseline_duration},
                {'Metric': 'Speaking_Duration', 'Condition': 'EngageCue', 'Group': group, 'Value': engagecue_duration},
                {'Metric': 'Speaking_Frequency', 'Condition': 'Baseline', 'Group': group, 'Value': row['noai_count'] / minutes},
                {'Metric': 'Speaking_Frequency', 'Condition': 'EngageCue', 'Group': group, 'Value': row['ai_count'] / minutes}
            ])
        
        return pd.DataFrame(results)

    def load_editing_data(self) -> pd.DataFrame:
        """加载编辑数据"""
        print("正在加载编辑数据...")
        
        if not self.data_paths['edit'].exists():
            print(f"✗ 编辑数据文件不存在: {self.data_paths['edit']}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.data_paths['edit'])
        print(f"✓ 编辑数据加载成功: {df.shape}")
        
        # 按Group×Condition聚合Engagement字段
        results = []
        for _, row in df.iterrows():
            # 检查Type列而不是Condition列
            condition = 'EngageCue' if row['Type'] == 'AI' else 'Baseline'
            results.append({
                'Metric': 'Editing_Activity',
                'Condition': condition,
                'Group': row['Group'],
                'Value': row['Engagement']
            })
        
        return pd.DataFrame(results)

    def load_browsing_data(self) -> pd.DataFrame:
        """加载浏览数据"""
        print("正在加载浏览数据...")
        
        if not self.data_paths['browsing'].exists():
            print(f"✗ 浏览数据文件不存在: {self.data_paths['browsing']}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.data_paths['browsing'])
        print(f"✓ 浏览数据加载成功: {df.shape}")
        
        # 使用GroupMean_AI/GroupMean_NoAI
        results = []
        for _, row in df.iterrows():
            group = row['Group']
            results.extend([
                {'Metric': 'Browsing_Activity', 'Condition': 'Baseline', 'Group': group, 'Value': row['GroupMean_NoAI']},
                {'Metric': 'Browsing_Activity', 'Condition': 'EngageCue', 'Group': group, 'Value': row['GroupMean_AI']}
            ])
        
        return pd.DataFrame(results)

    def load_video_data(self) -> pd.DataFrame:
        """加载视频标注数据"""
        print("正在加载视频标注数据...")
        
        if not self.data_paths['video'].exists():
            print(f"✗ 视频标注数据文件不存在: {self.data_paths['video']}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.data_paths['video'])
        print(f"✓ 视频标注数据加载成功: {df.shape}")
        
        # 标准化Group列名：G0->Group0, G1->Group1, Ga->Group10, Gb->Group11
        df['Group'] = df['Group'].replace({
            'G0': 'Group0', 'G1': 'Group1', 'G2': 'Group2', 'G3': 'Group3', 'G4': 'Group4',
            'G5': 'Group5', 'G6': 'Group6', 'G7': 'Group7', 'G8': 'Group8', 'G9': 'Group9',
            'Ga': 'Group10', 'Gb': 'Group11'
        })
        
        # 转换为长格式，处理posture和gaze数据
        results = []
        
        for _, row in df.iterrows():
            group = row['Group']
            
            # Posture scores
            results.extend([
                {
                    'Metric': 'Posture_Score',
                    'Condition': 'Baseline',
                    'Group': group,
                    'Value': row['NoAI_Mean_Level_posture']
                },
                {
                    'Metric': 'Posture_Score',
                    'Condition': 'EngageCue',
                    'Group': group,
                    'Value': row['AI_Mean_Level_posture']
                }
            ])
            
            # Gaze scores  
            results.extend([
                {
                    'Metric': 'Gaze_Score',
                    'Condition': 'Baseline',
                    'Group': group,
                    'Value': row['NoAI_Mean_Level_gaze']
                },
                {
                    'Metric': 'Gaze_Score',
                    'Condition': 'EngageCue',
                    'Group': group,
                    'Value': row['AI_Mean_Level_gaze']
                }
            ])
        
        return pd.DataFrame(results)

    def calculate_per_metric_statistics(self, all_data: pd.DataFrame) -> Dict[str, Dict]:
        """按维度独立计算统计量"""
        print("正在计算各维度统计量...")
        
        metric_stats = {}
        
        for metric in self.metrics:
            print(f"  处理维度: {self.metric_info[metric]['label']}")
            
            # 筛选该维度数据
            metric_data = all_data[all_data['Metric'] == metric].copy()
            
            if metric_data.empty:
                print(f"    ✗ 维度 {metric} 没有数据")
                continue
            
            # 按组配对：找到该维度内Baseline和EngageCue都存在的组
            baseline_groups = set(metric_data[metric_data['Condition'] == 'Baseline']['Group'])
            engagecue_groups = set(metric_data[metric_data['Condition'] == 'EngageCue']['Group'])
            paired_groups = baseline_groups & engagecue_groups
            
            if not paired_groups:
                print(f"    ✗ 维度 {metric} 没有配对组")
                continue
            
            # 筛选配对数据并按组对齐，对每组取均值
            baseline_group_means = metric_data[metric_data['Condition'] == 'Baseline'].groupby('Group')['Value'].mean()
            engagecue_group_means = metric_data[metric_data['Condition'] == 'EngageCue'].groupby('Group')['Value'].mean()
            
            # 按组对齐，确保只提取配对组的数据
            aligned_groups = sorted(paired_groups)
            baseline_aligned = [baseline_group_means[group] for group in aligned_groups]
            engagecue_aligned = [engagecue_group_means[group] for group in aligned_groups]
            
            # 计算统计量
            n_pairs = len(paired_groups)
            baseline_mean = np.mean(baseline_aligned)
            engagecue_mean = np.mean(engagecue_aligned)
            baseline_se = np.std(baseline_aligned, ddof=1) / np.sqrt(n_pairs)
            engagecue_se = np.std(engagecue_aligned, ddof=1) / np.sqrt(n_pairs)
            
            # 配对t检验
            if n_pairs > 1:
                t_stat, p_value = ttest_rel(baseline_aligned, engagecue_aligned)
                
                # 计算效应量 (Cohen's d)
                diff = np.array(engagecue_aligned) - np.array(baseline_aligned)
                pooled_std = np.sqrt((np.var(baseline_aligned, ddof=1) + np.var(engagecue_aligned, ddof=1)) / 2)
                effect_size = np.mean(diff) / pooled_std if pooled_std > 0 else np.nan
            else:
                t_stat, p_value, effect_size = np.nan, np.nan, np.nan
            
            # 确保p_value是标量
            if hasattr(p_value, '__len__') and len(p_value) > 1:
                p_value_scalar = p_value[0] if hasattr(p_value, '__getitem__') else float(p_value)
            else:
                p_value_scalar = float(p_value) if not np.isnan(p_value) else p_value
            
            metric_stats[metric] = {
                'n_pairs': n_pairs,
                'baseline_mean': baseline_mean,
                'baseline_se': baseline_se,
                'engagecue_mean': engagecue_mean,
                'engagecue_se': engagecue_se,
                't_stat': t_stat,
                'p_value': p_value_scalar,
                'effect_size': effect_size,
                'significant': p_value_scalar < 0.05 if not np.isnan(p_value_scalar) else False
            }
            
            print(f"    ✓ 完成，配对组数: {n_pairs}")
        
        return metric_stats

    def run_analysis(self, minutes: int):
        """运行完整分析"""
        print(f"=== 开始参与度水平分析 ===")
        print(f"Task minutes = {minutes}")
        print()
        
        # 加载所有数据
        all_data_frames = []
        
        speech_data = self.load_speaking_data(minutes)
        if not speech_data.empty:
            all_data_frames.append(speech_data)
        
        edit_data = self.load_editing_data()
        if not edit_data.empty:
            all_data_frames.append(edit_data)
        
        browsing_data = self.load_browsing_data()
        if not browsing_data.empty:
            all_data_frames.append(browsing_data)
        
        video_data = self.load_video_data()
        if not video_data.empty:
            all_data_frames.append(video_data)
        
        if not all_data_frames:
            print("✗ 没有可用的数据")
            return
        
        # 合并所有数据
        all_data = pd.concat(all_data_frames, ignore_index=True)
        print(f"✓ 数据整合完成，总计: {len(all_data)} 条记录\n")
        
        # 按维度计算统计量
        metric_stats = self.calculate_per_metric_statistics(all_data)
        
        # 生成图表
        create_2x3_subplot_figure(self, metric_stats)
        
        # 输出汇总文件
        output_summary_files(self, metric_stats)
        
        print("\n=== 参与度水平分析完成！===")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='参与度水平分析器')
    parser.add_argument('--minutes', type=int, required=True, help='任务时长（分钟）')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = ParticipationLevelsAnalyzer()
    analyzer.run_analysis(args.minutes)

if __name__ == "__main__":
    main()