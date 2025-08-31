#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成三联图：Speaking/Editing/Browsing 的基尼系数对比
每个子图显示 Baseline vs EngageCue 的箱线图
适合 CHI 双栏排版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加viz模块路径
sys.path.append(str(Path(__file__).parent.parent))
from viz.style_kit import (
    apply_style, get_palette, add_headroom, annotate_values,
    annotate_significance, annotate_n, style_legend, style_axes,
    grouped_bars, boxplot, save_figure
)

# 应用统一样式
apply_style()

def load_speech_data():
    """加载speech数据"""
    try:
        speech_file = Path("speech/paired_speech_data.csv")
        if speech_file.exists():
            df = pd.read_csv(speech_file)
            # 提取基尼系数数据
            speech_data = []
            for _, row in df.iterrows():
                speech_data.append({
                    'Condition': 'Baseline',
                    'Gini': row['noai_gini']
                })
                speech_data.append({
                    'Condition': 'EngageCue',
                    'Gini': row['ai_gini']
                })
            return pd.DataFrame(speech_data), True
        else:
            print(f"警告：speech文件不存在: {speech_file}")
            return None, False
    except Exception as e:
        print(f"加载speech数据时出错: {e}")
        return None, False

def load_edit_note_data():
    """加载edit_note数据"""
    try:
        edit_file = Path("edit_note/gini_analysis.csv")
        if edit_file.exists():
            df = pd.read_csv(edit_file)
            # 透视成宽表
            pivot_df = df.pivot(index='Group', columns='Type', values='Gini_Coefficient')
            pivot_df = pivot_df.reset_index()
            
            edit_data = []
            for _, row in pivot_df.iterrows():
                if pd.notna(row['NoAI']) and pd.notna(row['AI']):
                    edit_data.append({
                        'Condition': 'Baseline',
                        'Gini': row['NoAI']
                    })
                    edit_data.append({
                        'Condition': 'EngageCue',
                        'Gini': row['AI']
                    })
            return pd.DataFrame(edit_data), True
        else:
            print(f"警告：edit_note文件不存在: {edit_file}")
            return None, False
    except Exception as e:
        print(f"加载edit_note数据时出错: {e}")
        return None, False

def load_page_behavior_data():
    """加载page_behavior数据"""
    try:
        page_file = Path("page_behavior/analysis/balance_gini_paired_results.csv")
        if page_file.exists():
            df = pd.read_csv(page_file)
            page_data = []
            for _, row in df.iterrows():
                if pd.notna(row['Gini_NoAI']) and pd.notna(row['Gini_AI']):
                    page_data.append({
                        'Condition': 'Baseline',
                        'Gini': row['Gini_NoAI']
                    })
                    page_data.append({
                        'Condition': 'EngageCue',
                        'Gini': row['Gini_AI']
                    })
            return pd.DataFrame(page_data), True
        else:
            print(f"警告：page_behavior文件不存在: {page_file}")
            return None, False
    except Exception as e:
        print(f"加载page_behavior数据时出错: {e}")
        return None, False

def create_gini_triptych():
    """创建三联图"""
    print("正在加载数据...")
    
    # 加载三个域的数据
    speech_data, speech_loaded = load_speech_data()
    edit_data, edit_loaded = load_edit_note_data()
    page_data, page_loaded = load_page_behavior_data()
    
    if not any([speech_loaded, edit_loaded, page_loaded]):
        print("错误：没有成功加载任何数据")
        return
    
    # 计算统一的y轴范围
    all_gini_values = []
    if speech_loaded and speech_data is not None:
        all_gini_values.extend(speech_data['Gini'].dropna().tolist())
    if edit_loaded and edit_data is not None:
        all_gini_values.extend(edit_data['Gini'].dropna().tolist())
    if page_loaded and page_data is not None:
        all_gini_values.extend(page_data['Gini'].dropna().tolist())
    
    if all_gini_values:
        y_min = 0
        y_max = max(all_gini_values) + 0.05
        y_max = max(y_max, 0.5)  # 最低不小于0.5
        y_max = min(y_max, 0.6)  # 最高不超过0.6
    else:
        y_min, y_max = 0, 0.5
    
    print(f"统一y轴范围: {y_min:.3f} - {y_max:.3f}")
    
    # 创建三联图
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    fig.suptitle('Gini Coefficient Comparison Across Domains', fontsize=14, fontweight='semibold', y=0.95, fontfamily='sans-serif')
    
    # 1. Speaking 子图
    if speech_loaded and speech_data is not None:
        ax1 = axes[0]
        create_subplot(ax1, speech_data, 'Speaking', y_min, y_max, p_value=0.044, significant=True)
    else:
        # 创建空的子图
        ax1 = axes[0]
        ax1.text(0.5, 0.5, 'No Speech Data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Speaking')
        ax1.set_ylim(y_min, y_max)
    
    # 2. Editing 子图
    if edit_loaded and edit_data is not None:
        ax2 = axes[1]
        create_subplot(ax2, edit_data, 'Editing', y_min, y_max, p_value=None, significant=False, is_editing=True)
    else:
        # 创建空的子图
        ax2 = axes[1]
        ax2.text(0.5, 0.5, 'No Edit Data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Editing')
        ax2.set_ylim(0, 0.7)  # Editing子图单独设置y轴上限
    
    # 3. Browsing 子图
    if page_loaded and page_data is not None:
        ax3 = axes[2]
        create_subplot(ax3, page_data, 'Browsing', y_min, y_max, p_value=None, significant=False)
    else:
        # 创建空的子图
        ax3 = axes[2]
        ax3.text(0.5, 0.5, 'No Browsing Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Browsing')
        ax3.set_ylim(y_min, y_max)
    
    # 添加整体说明
    fig.text(0.5, 0.02, 'Gini coefficient (lower = more balanced)', ha='center', va='bottom', fontsize=10, style='italic', fontfamily='sans-serif')
    
    plt.tight_layout()
    
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.38)
    
    # 保存图片
    output_path = Path("figures/gini_overall_triptych")
    output_path.parent.mkdir(exist_ok=True)
    save_figure(fig, output_path)
    
    plt.show()
    
    return fig

def create_subplot(ax, data, title, y_min, y_max, p_value=None, significant=False, is_editing=False):
    """创建单个子图"""
    # 准备箱线图数据
    baseline_data = data[data['Condition'] == 'Baseline']['Gini'].dropna().tolist()
    engage_data = data[data['Condition'] == 'EngageCue']['Gini'].dropna().tolist()
    
    # 使用统一样式的箱线图
    box_plot = boxplot(ax, baseline_data, engage_data)
    
    # 设置标题和标签
    style_axes(ax, ylabel='Gini coefficient (lower = more balanced)', title=title)
    
    # 根据是否为Editing子图设置不同的y轴范围
    if is_editing:
        ax.set_ylim(0, 0.7)  # Editing子图单独设置y轴上限
    else:
        ax.set_ylim(y_min, y_max)  # Speaking和Browsing子图保持原样
    
    # 添加顶部留白
    add_headroom(ax)
    
    # 添加显著性标记
    if significant and p_value is not None:
        annotate_significance(ax, f'* p = {p_value:.3f}')
    else:
        annotate_significance(ax)
    
    # 标注中位数
    median_baseline = np.median(baseline_data)
    median_engage = np.median(engage_data)
    annotate_values(ax, [1.0, 1.75], [median_baseline, median_engage], fmt="%.2f")

def save_integrated_data():
    """保存整合的数据表"""
    print("正在保存整合数据...")
    
    integrated_data = []
    
    # 加载并整合数据
    speech_data, speech_loaded = load_speech_data()
    if speech_loaded and speech_data is not None:
        speech_data['Domain'] = 'Speaking'
        integrated_data.append(speech_data)
    
    edit_data, edit_loaded = load_edit_note_data()
    if edit_loaded and edit_data is not None:
        edit_data['Domain'] = 'Editing'
        integrated_data.append(edit_data)
    
    page_data, page_loaded = load_page_behavior_data()
    if page_loaded and page_data is not None:
        page_data['Domain'] = 'Browsing'
        integrated_data.append(page_data)
    
    if integrated_data:
        # 合并所有数据
        all_data = pd.concat(integrated_data, ignore_index=True)
        
        # 重新排列列顺序
        all_data = all_data[['Domain', 'Condition', 'Gini']]
        
        # 保存整合数据
        output_path = Path("integrated_gini_summary.csv")
        all_data.to_csv(output_path, index=False)
        print(f"整合数据已保存到: {output_path}")
        
        # 显示数据摘要
        print("\n数据摘要:")
        print(all_data.groupby(['Domain', 'Condition'])['Gini'].describe())
        
        return all_data
    else:
        print("没有数据可保存")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("基尼系数三联图生成器")
    print("=" * 60)
    
    # 保存整合数据
    integrated_data = save_integrated_data()
    
    # 生成三联图
    if integrated_data is not None:
        create_gini_triptych()
        print("\n三联图生成完成！")
    else:
        print("无法生成三联图：没有有效数据")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
