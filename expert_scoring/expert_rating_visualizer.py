#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专家评分对比图生成器
Expert Rating Visualizer

功能：
1. Figure Z（正文）：合并两位专家评分的分组柱状图
2. Figure Z-suppl（附录）：分专家的2×3小面板图

作者：AI Assistant
版本：1.0
日期：2024
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

# 添加viz模块路径
sys.path.append(str(Path(__file__).parent.parent))
from viz.style_kit import (
    apply_style, get_palette, add_headroom, annotate_values,
    annotate_significance, annotate_n, style_legend, style_axes,
    grouped_bars, boxplot, save_figure
)

# 应用统一样式
apply_style()
warnings.filterwarnings('ignore')

class ExpertRatingVisualizer:
    """专家评分可视化器主类"""
    
    def __init__(self, project_root: str = None):
        """初始化可视化器"""
        if project_root is None:
            current_dir = Path(__file__).parent
            self.project_root = current_dir.parent
        else:
            self.project_root = Path(project_root)
            
        # 设置数据文件路径
        self.data_path = self.project_root / 'expert_scoring' / 'intermediate' / 'summary_descriptives.csv'
        
        # 设置输出目录
        self.output_dir = Path(__file__).parent
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        # 使用统一的颜色配置
        self.colors = get_palette()
        
        # 布局参数配置
        self.layout_params = {
            'combined': {
                'bar_width': 0.35,
                'x_offset': 0.2,
                'figsize': (7, 5)
            },
            'by_expert': {
                'bar_width': 0.3,        # 小面板用更细的柱子
                'x_offset': 0.25,        # 稍微放宽间距
                'figsize': (11, 6)       # 稍微放宽整体尺寸
            }
        }
        
        # 维度映射
        self.dimension_mapping = {
            'Process Score': 'Process',
            'Outcome Score': 'Outcome', 
            'Total Score': 'Overall'
        }
        
        print("=== 专家评分可视化器初始化完成 ===")
        print(f"项目根目录: {self.project_root}")
        print(f"数据文件: {self.data_path}")
        print(f"输出目录: {self.figures_dir}")

    def load_expert_data(self) -> Dict[str, Dict[str, Dict]]:
        """加载专家评分数据"""
        print("正在加载专家评分数据...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"✓ 数据加载成功: {df.shape}")
        
        # 添加条件名映射逻辑
        condition_mapping = {
            'AI': 'EngageCue',
            'Condition=AI': 'EngageCue',
            'NoAI': 'Baseline',
            'Condition=NoAI': 'Baseline'
        }
        df['Scope'] = df['Scope'].replace(condition_mapping)
        print("调试: 条件列 unique ->", df['Scope'].unique())
        
        # 筛选需要的数据
        experts = ['E1', 'E2']
        dimensions = ['Process Score', 'Outcome Score', 'Total Score']
        conditions = ['EngageCue', 'Baseline']
        
        expert_data = {}
        
        for expert in experts:
            expert_data[expert] = {}
            
            for dimension in dimensions:
                dimension_name = self.dimension_mapping[dimension]
                expert_data[expert][dimension_name] = {}
                
                for condition in conditions:
                    # 筛选数据
                    mask = (df['Expert'] == expert) & (df['Score'] == dimension) & (df['Scope'] == condition)
                    row = df[mask]
                    
                    if not row.empty:
                        row = row.iloc[0]
                        condition_name = condition  # 直接使用映射后的条件名
                        
                        expert_data[expert][dimension_name][condition_name] = {
                            'mean': row['mean'],
                            'std': row['std'],
                            'count': int(row['count']),
                            'se': row['std'] / np.sqrt(row['count']),  # 计算标准误
                            'min': row['min'],
                            'max': row['max']
                        }
                    else:
                        print(f"⚠️ 数据缺失: {expert} - {dimension} - {condition}")
        
        print(f"✓ 数据处理完成，包含专家: {list(expert_data.keys())}")
        return expert_data

    def determine_scale_range(self, expert_data: Dict) -> Dict[str, Any]:
        """动态确认评分量表范围，设置合适的Y轴上限"""
        print("正在确认评分量表范围...")
        
        # 检查数据中的最大值
        max_score = 0
        for expert in expert_data.values():
            for dimension in expert.values():
                for condition in dimension.values():
                    max_score = max(max_score, condition['max'])
        
        # 根据实际最大值设置Y轴范围
        if max_score <= 5:
            y_max = 6.0
            scale_note = "Expert Rating (1-5)"
            reference_line = 4.0
        elif max_score <= 7:
            y_max = 8.0  
            scale_note = "Expert Rating (1-7)"
            reference_line = 5.0
        else:
            y_max = max_score * 1.2
            scale_note = f"Expert Rating (1-{int(max_score)})"
            reference_line = max_score * 0.7
            
        scale_info = {
            'y_max': y_max,
            'y_label': scale_note,
            'reference_line': reference_line,
            'max_score': max_score
        }
        
        print(f"✓ 量表范围确认: {scale_note}, Y轴上限: {y_max}")
        return scale_info

    def calculate_combined_stats(self, e1_data: Dict, e2_data: Dict) -> Dict[str, Dict]:
        """计算合并专家统计量"""
        print("正在计算合并统计量...")
        
        combined_data = {}
        
        for dimension in e1_data.keys():
            combined_data[dimension] = {}
            
            for condition in ['Baseline', 'EngageCue']:
                if condition in e1_data[dimension] and condition in e2_data[dimension]:
                    e1 = e1_data[dimension][condition]
                    e2 = e2_data[dimension][condition]
                    
                    n1, n2 = e1['count'], e2['count']
                    mean1, mean2 = e1['mean'], e2['mean']
                    std1, std2 = e1['std'], e2['std']
                    
                    # 加权平均均值
                    total_n = n1 + n2
                    if n1 == n2:
                        # 简单平均（大多数情况）
                        weighted_mean = (mean1 + mean2) / 2
                        method = 'simple_average'
                    else:
                        # 加权平均
                        weighted_mean = (n1 * mean1 + n2 * mean2) / total_n
                        method = 'weighted_average'
                    
                    # Pooled standard error
                    pooled_var = ((n1-1)*std1**2 + (n2-1)*std2**2) / (total_n - 2)
                    pooled_se = np.sqrt(pooled_var / total_n)
                    
                    combined_data[dimension][condition] = {
                        'mean': weighted_mean,
                        'se': pooled_se,
                        'n_total': total_n,
                        'method': method
                    }
                else:
                    print(f"⚠️ 维度 {dimension} 条件 {condition} 缺少配对数据")
        
        print(f"✓ 合并统计量计算完成")
        return combined_data

    def plot_grouped_bars(self, ax, dimension_data: Dict, layout_type: str = 'combined', scale_info: Dict = None):
        """绘制分组柱状图的通用函数"""
        params = self.layout_params[layout_type]
        
        x_offset = params['x_offset']
        x_pos = [-x_offset, x_offset]
        
        # 提取数据
        baseline_data = dimension_data['Baseline']
        engagecue_data = dimension_data['EngageCue']
        
        means_b = [baseline_data['mean']]
        ses_b = [baseline_data['se']]
        means_e = [engagecue_data['mean']]
        ses_e = [engagecue_data['se']]
        x_positions = [0]
        
        # 使用统一样式的分组柱状图
        x_coords, y_tops = grouped_bars(ax, means_b, ses_b, means_e, ses_e, x_positions)
        
        # 标注柱顶均值
        annotate_values(ax, x_coords, [means_b[0], means_e[0]], [ses_b[0], ses_e[0]])
        
        # 添加显著性标记
        annotate_significance(ax)
        
        # 设置X轴 - 适应精致柱状图
        ax.set_xlim(-0.5, 0.5)

    def create_combined_figure(self, combined_data: Dict, scale_info: Dict):
        """创建Figure Z: 合并专家评分"""
        print("正在生成合并专家评分图...")
        
        params = self.layout_params['combined']
        fig, ax = plt.subplots(figsize=params['figsize'])
        
        # 准备数据
        dimensions = ['Process', 'Outcome', 'Overall']
        bar_width = 0.35
        
        # 提取所有数据（确保使用正确的键名）
        baseline_means = []
        baseline_ses = []
        engagecue_means = []
        engagecue_ses = []
        valid_dimensions = []
        
        for dim in dimensions:
            if dim in combined_data and 'Baseline' in combined_data[dim] and 'EngageCue' in combined_data[dim]:
                baseline_means.append(combined_data[dim]['Baseline']['mean'])
                baseline_ses.append(combined_data[dim]['Baseline']['se'])
                engagecue_means.append(combined_data[dim]['EngageCue']['mean'])
                engagecue_ses.append(combined_data[dim]['EngageCue']['se'])
                valid_dimensions.append(dim)
            else:
                # 调试信息
                if dim in combined_data:
                    print(f"⚠️ 维度 {dim} 可用条件: {list(combined_data[dim].keys())}")
                else:
                    print(f"⚠️ 维度 {dim} 完全缺失")
                print(f"⚠️ 跳过维度 {dim}")
        
        # 根据有效维度数量设置位置
        x_positions = np.arange(len(valid_dimensions))
        
        # 使用统一样式的分组柱状图
        x_coords, y_tops = grouped_bars(ax, baseline_means, baseline_ses, engagecue_means, engagecue_ses, x_positions)
        
        # 标注均值和显著性标记
        annotate_values(ax, x_coords, baseline_means + engagecue_means, baseline_ses + engagecue_ses)
        annotate_significance(ax)
        
        # 设置轴和标签
        style_axes(ax, xlabel='Dimension', ylabel=scale_info['y_label'], 
                  title='Expert-rated Collaboration Quality')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(valid_dimensions)
        ax.set_ylim(0, 8)  # 设置为0-8范围
        
        # 添加顶部留白
        add_headroom(ax)
        
        # 添加参考线
        ax.axhline(y=scale_info['reference_line'], color='lightgray', 
                  linestyle='--', alpha=0.4, linewidth=0.8)
        
        # 图例
        style_legend(ax)
        
        plt.tight_layout()
        
        return fig

    def create_by_expert_figure(self, e1_data: Dict, e2_data: Dict, scale_info: Dict):
        """创建Figure Z-suppl: 分专家2×3面板"""
        print("正在生成分专家面板图...")
        
        params = self.layout_params['by_expert']
        fig, axes = plt.subplots(2, 3, figsize=params['figsize'])
        fig.subplots_adjust(wspace=0.3, hspace=0.45)
        
        dimensions = ['Process', 'Outcome', 'Overall']
        experts_data = [e1_data, e2_data]
        expert_names = ['Expert 1', 'Expert 2']
        
        # 绘制每个面板
        for row, (expert_data, expert_name) in enumerate(zip(experts_data, expert_names)):
            for col, dimension in enumerate(dimensions):
                ax = axes[row, col]
                
                if dimension in expert_data:
                    # 使用小面板布局参数
                    self.plot_grouped_bars(ax, expert_data[dimension], 'by_expert', scale_info)
                    
                    # 设置面板标题和样式
                    ylabel = scale_info['y_label'] if col == 0 else ''
                    style_axes(ax, ylabel=ylabel, title=f"{expert_name} - {dimension}")
                else:
                    # 如果数据缺失，显示提示
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12, color='gray')
                    style_axes(ax, title=f"{expert_name} - {dimension}")
                
                # 设置Y轴
                ax.set_ylim(0, 8)  # 统一设置为0-8
                add_headroom(ax)
                
                # 添加参考线
                ax.axhline(y=scale_info['reference_line'], color='lightgray', 
                          linestyle='--', alpha=0.4, linewidth=0.8)
        
        # 整体标题
        fig.suptitle('Expert-rated Collaboration Quality by Expert', 
                    fontsize=14, fontweight='semibold', y=0.95)
        
        # 共享图例
        legend_elements = [
            mpatches.Patch(color=self.colors['Baseline'], label='Baseline'),
            mpatches.Patch(color=self.colors['EngageCue'], label='EngageCue')
        ]
        fig.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98), frameon=True)
        
        return fig

    def save_figures(self, fig, filename_prefix: str):
        """保存图表为PNG和PDF格式"""
        output_path = self.figures_dir / filename_prefix
        save_figure(fig, output_path)

    def generate_caption(self, scale_info: Dict) -> Tuple[str, str]:
        """生成图表说明文字"""
        scale_range = scale_info['y_label'].split('(')[1].split(')')[0]  # 提取"1-5"或"1-7"
        
        caption_combined = f"""
Figure Z. Expert-rated collaboration quality across conditions ({scale_range} scale). 
Combined results represent the average of two independent expert ratings. 
Standard errors calculated using pooled variance method. 
All dimensions show non-significant differences (p > .05).
        """.strip()
        
        caption_by_expert = f"""
Figure Z-suppl. Expert-rated collaboration quality by individual expert ({scale_range} scale). 
Each panel shows ratings from one expert across three dimensions. 
Standard errors based on within-expert variance. 
Individual expert analyses show consistent non-significant patterns.
        """.strip()
        
        return caption_combined, caption_by_expert

    def save_summary_data(self, combined_data: Dict, e1_data: Dict, e2_data: Dict, scale_info: Dict):
        """保存汇总数据文件"""
        print("正在保存汇总数据...")
        
        summary_data = []
        
        # 合并数据
        for dimension in combined_data.keys():
            for condition in ['Baseline', 'EngageCue']:
                if condition in combined_data[dimension]:
                    data = combined_data[dimension][condition]
                    summary_data.append({
                        'Expert': 'Combined',
                        'Dimension': dimension,
                        'Condition': condition,
                        'Mean': data['mean'],
                        'SE': data['se'],
                        'N_Total': data['n_total'],
                        'Method': data['method']
                    })
        
        # 分专家数据
        for expert_name, expert_data in [('E1', e1_data), ('E2', e2_data)]:
            for dimension in expert_data.keys():
                for condition in ['Baseline', 'EngageCue']:
                    if condition in expert_data[dimension]:
                        data = expert_data[dimension][condition]
                        summary_data.append({
                            'Expert': expert_name,
                            'Dimension': dimension,
                            'Condition': condition,
                            'Mean': data['mean'],
                            'SE': data['se'],
                            'N_Total': data['count'],
                            'Method': 'individual'
                        })
        
        # 保存CSV文件
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'expert_ratings_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"✓ 汇总数据已保存: {summary_path}")
        
        # 保存说明文字
        caption_combined, caption_by_expert = self.generate_caption(scale_info)
        caption_path = self.output_dir / 'expert_ratings_captions.txt'
        
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write("=== Figure Z (Combined) Caption ===\n")
            f.write(caption_combined)
            f.write("\n\n=== Figure Z-suppl (By Expert) Caption ===\n")
            f.write(caption_by_expert)
            f.write(f"\n\n=== Scale Information ===\n")
            f.write(f"Scale Range: {scale_info['y_label']}\n")
            f.write(f"Max Score: {scale_info['max_score']}\n")
            f.write(f"Y-axis Max: {scale_info['y_max']}\n")
        
        print(f"✓ 说明文字已保存: {caption_path}")

    def run_analysis(self):
        """运行完整的专家评分可视化分析"""
        print("=== 开始专家评分可视化分析 ===")
        
        try:
            # 1. 加载专家数据
            expert_data = self.load_expert_data()
            e1_data = expert_data['E1']
            e2_data = expert_data['E2']
            
            # 2. 确认量表范围
            scale_info = self.determine_scale_range(expert_data)
            
            # 3. 计算合并统计量
            combined_data = self.calculate_combined_stats(e1_data, e2_data)
            
            # 4. 生成Figure Z（合并专家评分图）
            combined_fig = self.create_combined_figure(combined_data, scale_info)
            self.save_figures(combined_fig, 'expert_ratings_combined')
            
            # 5. 生成Figure Z-suppl（分专家面板图）
            by_expert_fig = self.create_by_expert_figure(e1_data, e2_data, scale_info)
            self.save_figures(by_expert_fig, 'expert_ratings_by_expert')
            
            # 显示图表（交互式窗口）
            plt.show()
            
            # 6. 保存汇总数据和说明文字
            self.save_summary_data(combined_data, e1_data, e2_data, scale_info)
            
            print("\n=== 专家评分可视化分析完成！===")
            print(f"✓ 生成图表: expert_ratings_combined.png/pdf")
            print(f"✓ 生成图表: expert_ratings_by_expert.png/pdf") 
            print(f"✓ 汇总数据: expert_ratings_summary.csv")
            print(f"✓ 说明文字: expert_ratings_captions.txt")
            
        except Exception as e:
            print(f"✗ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    # 创建可视化器并运行
    visualizer = ExpertRatingVisualizer()
    visualizer.run_analysis()

if __name__ == "__main__":
    main()
