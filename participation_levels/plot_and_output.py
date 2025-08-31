"""
图表生成和输出功能模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# 添加viz模块路径
sys.path.append(str(Path(__file__).parent.parent))
from viz.style_kit import (
    apply_style, get_palette, add_headroom, annotate_values,
    annotate_significance, annotate_n, style_legend, style_axes,
    grouped_bars, boxplot, save_figure
)

def create_2x3_subplot_figure(analyzer, metric_stats):
    """创建2×3小面板图表"""
    print("正在生成2×3小面板图表...")
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35, hspace=0.45)
    
    # 扁平化axes以便索引
    axes_flat = axes.flatten()
    
    # 绘制每个维度
    for i, metric in enumerate(analyzer.metrics):
        ax = axes_flat[i]
        
        if metric not in metric_stats:
            # 如果没有数据，显示空面板
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            style_axes(ax, title=f"{analyzer.metric_info[metric]['label']} ({analyzer.metric_info[metric]['unit']})")
            continue
        
        stats = metric_stats[metric]
        
        means_b = [stats['baseline_mean']]
        ses_b = [stats['baseline_se']]
        means_e = [stats['engagecue_mean']]
        ses_e = [stats['engagecue_se']]
        x_positions = [0]
        
        # 使用统一样式的分组柱状图
        x_coords, y_tops = grouped_bars(ax, means_b, ses_b, means_e, ses_e, x_positions)
        
        # 设置面板属性
        ylabel = f"{analyzer.metric_info[metric]['label']} ({analyzer.metric_info[metric]['unit']})"
        style_axes(ax, title=ylabel)
        
        # 设置轴范围 - 适应精致柱状图
        ax.set_xlim(-0.5, 0.5)
        
        # 根据指标特点设置不同的Y轴上限
        if metric == 'Posture_Score':
            ax.set_ylim(0, 2.5)
        elif metric == 'Gaze_Score':
            ax.set_ylim(0, 3.5)
        else:
            y_max = max(stats['baseline_mean'] + stats['baseline_se'],
                       stats['engagecue_mean'] + stats['engagecue_se'])
            ax.set_ylim(0, y_max * 1.15)
        
        # 添加顶部留白
        add_headroom(ax)
        
        # 标注柱顶均值
        annotate_values(ax, x_coords, [means_b[0], means_e[0]], [ses_b[0], ses_e[0]])
        
        # 添加显著性标记
        annotate_significance(ax)
        
        # 添加样本量标注
        annotate_n(ax, f'n = {stats["n_pairs"]}')
    
    # 添加共享图例
    palette = get_palette()
    legend_elements = [
        mpatches.Patch(color=palette['Baseline'], label='Baseline'),
        mpatches.Patch(color=palette['EngageCue'], label='EngageCue')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              framealpha=0.15, fontsize=11)
    
    # 添加整体标题
    fig.suptitle('Participation Levels Across Six Dimensions', 
                fontsize=16, fontweight='semibold', y=0.95)
    
    # 添加说明
    caption = ("Note: Panels have different scales; comparisons are within each metric. "
              "Error bars show standard errors based on paired groups within each dimension.")
    fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=9, 
            color='#666666', wrap=True)
    
    # 保存图表
    output_path = analyzer.figures_dir / 'participation_levels_2x3'
    save_figure(fig, output_path)
    
    plt.show()

def output_summary_files(analyzer, metric_stats):
    """输出汇总文件"""
    print("正在输出汇总文件...")
    
    # 1. 输出integrated_levels_summary.csv
    summary_data = []
    for metric in analyzer.metrics:
        if metric not in metric_stats:
            continue
        
        stats = metric_stats[metric]
        
        summary_data.extend([
            {
                'Metric': metric,
                'Condition': 'Baseline',
                'Mean': stats['baseline_mean'],
                'SE': stats['baseline_se'],
                'N': stats['n_pairs']
            },
            {
                'Metric': metric,
                'Condition': 'EngageCue',
                'Mean': stats['engagecue_mean'],
                'SE': stats['engagecue_se'],
                'N': stats['n_pairs']
            }
        ])
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = analyzer.output_dir / 'integrated_levels_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ 汇总统计已保存: {summary_path}")
    
    # 2. 输出paired_tests_per_metric.csv
    test_data = []
    for metric in analyzer.metrics:
        if metric not in metric_stats:
            continue
        
        stats = metric_stats[metric]
        
        test_data.append({
            'Metric': metric,
            'N_Pairs': stats['n_pairs'],
            'T_Statistic': stats['t_stat'],
            'P_Value': stats['p_value'],
            'Effect_Size': stats['effect_size'],
            'Significant': stats['significant']
        })
    
    test_df = pd.DataFrame(test_data)
    test_path = analyzer.output_dir / 'paired_tests_per_metric.csv'
    test_df.to_csv(test_path, index=False)
    print(f"✓ 配对检验结果已保存: {test_path}")
