#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一图表样式模块
Unified Chart Style Kit

功能：
- 统一matplotlib配置
- 标准化绘图函数
- 一致的颜色和样式
- 统一的图表元素处理

作者：AI Assistant
版本：1.0
日期：2024
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# 设置随机种子确保一致性
np.random.seed(42)

# 全局精致柱状图参数
REFINED_BAR_WIDTH = 0.28
REFINED_BAR_SPACING = 0.35

class ChartStyleKit:
    """图表样式工具包主类"""
    
    def __init__(self, config_path: str = None):
        """初始化样式工具包"""
        if config_path is None:
            self.config_path = Path(__file__).parent / 'chart_config.yaml'
        else:
            self.config_path = Path(config_path)
            
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"⚠️ 配置文件不存在: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            print(f"⚠️ 配置文件加载失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'matplotlib': {
                'font_family': 'DejaVu Sans',
                'title_size': 16,
                'label_size': 12,
                'tick_size': 10,
                'legend_size': 11,
                'line_width': 1.1,
                'grid_alpha': 0.2,
                'grid_ls': '--',
                'dpi': 300,
                'save_bbox': 'tight',
                'save_pad_inches': 0.05
            },
            'palette': {
                'baseline': '#64B5F6',
                'engagecue': '#F48FB1',
                'median': '#D33A2C',
                'error': '#333333'
            },
            'bars': {
                'width': 0.52,
                'alpha': 0.95,
                'error_capsize': 3,
                'error_elinewidth': 1.1
            },
            'boxplot': {
                'face_alpha': 0.35,
                'edge_lw': 1.1,
                'box_lw': 1.1,
                'whisker_lw': 1.1,
                'cap_lw': 1.1,
                'median_lw': 2.2
            },
            'text': {
                'value_size': 10,
                'sig_size': 11,
                'ns_text': 'n.s.',
                'sig_y_rel': 0.92,
                'n_label_size': 9,
                'n_label_y_rel': -0.12
            },
            'layout': {
                'legend_loc': 'upper right',
                'legend_framealpha': 0.15,
                'headroom_ratio': 0.06
            }
        }
        
    def apply_style(self):
        """设定全局matplotlib参数"""
        config = self.config['matplotlib']
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [config['font_family'], 'Arial', 'Liberation Sans'],
            'axes.unicode_minus': False,
            'axes.titlesize': config['title_size'],
            'axes.labelsize': config['label_size'],
            'xtick.labelsize': config['tick_size'],
            'ytick.labelsize': config['tick_size'],
            'legend.fontsize': config['legend_size'],
            'axes.linewidth': config['line_width'],
            'grid.alpha': config['grid_alpha'],
            'grid.linestyle': config['grid_ls'],
            'savefig.dpi': config['dpi'],
            'savefig.bbox': config['save_bbox'],
            'savefig.pad_inches': config['save_pad_inches'],
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
    def get_palette(self) -> Dict[str, str]:
        """返回调色板"""
        palette = self.config['palette']
        return {
            'Baseline': palette['baseline'],
            'EngageCue': palette['engagecue']
        }
        
    def add_headroom(self, ax, top_ratio: float = None):
        """为y轴添加顶部留白"""
        if top_ratio is None:
            top_ratio = self.config['layout']['headroom_ratio']
            
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        new_y_max = y_max + y_range * top_ratio
        ax.set_ylim(y_min, new_y_max)
        
    def annotate_values(self, ax, xs: List[float], ys: List[float], 
                       errs: List[float] = None, fmt: str = "%.2f", dy: float = 0.02):
        """在柱顶/箱顶标注数值"""
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        offset = y_range * dy
        
        for i, (x, y) in enumerate(zip(xs, ys)):
            if errs is not None and i < len(errs):
                y_pos = y + errs[i] + offset
            else:
                y_pos = y + offset
                
            ax.text(x, y_pos, fmt % y, ha='center', va='bottom',
                   fontsize=self.config['text']['value_size'], 
                   color='#222222', weight='bold')
                   
    def annotate_significance(self, ax, text: str = None, y_rel: float = None):
        """在面板顶部居中标记显著性"""
        if text is None:
            text = self.config['text']['ns_text']
        if y_rel is None:
            y_rel = self.config['text']['sig_y_rel']
            
        color = '#d32f2f' if '*' in text else '#666666'
        
        ax.text(0.5, y_rel, text, ha='center', va='center',
               transform=ax.transAxes, 
               fontsize=self.config['text']['sig_size'], 
               weight='bold', color=color)
               
    def annotate_n(self, ax, n_text: str):
        """在x轴下方标注样本量"""
        y_rel = self.config['text']['n_label_y_rel']
        
        ax.text(0.5, y_rel, n_text, ha='center', va='top',
               transform=ax.transAxes,
               fontsize=self.config['text']['n_label_size'],
               color='#666666')
               
    def style_legend(self, ax, labels: List[str] = None):
        """设置右上角半透明图例"""
        if labels is None:
            labels = ['Baseline', 'EngageCue']
            
        handles = [
            mpatches.Patch(color=self.get_palette()['Baseline'], label=labels[0]),
            mpatches.Patch(color=self.get_palette()['EngageCue'], label=labels[1])
        ]
        
        ax.legend(handles=handles, 
                 loc=self.config['layout']['legend_loc'],
                 framealpha=self.config['layout']['legend_framealpha'],
                 fontsize=self.config['matplotlib']['legend_size'])
                 
    def style_axes(self, ax, xlabel: str = None, ylabel: str = None, title: str = None):
        """统一轴与标题风格"""
        if title:
            ax.set_title(title, fontweight='semibold', 
                        fontsize=self.config['matplotlib']['title_size'],
                        color='#222222')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config['matplotlib']['label_size'],
                         color='#222222')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config['matplotlib']['label_size'],
                         color='#222222')
                         
        # 设置网格
        ax.grid(True, alpha=self.config['matplotlib']['grid_alpha'],
               linestyle=self.config['matplotlib']['grid_ls'], axis='y')
               
        # 设置边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(self.config['matplotlib']['line_width'])
        ax.spines['bottom'].set_linewidth(self.config['matplotlib']['line_width'])
        
    def boxplot(self, ax, data_baseline: List[float], data_engage: List[float], 
               labels: Tuple[str, str] = ("Baseline", "EngageCue")):
        """统一样式的箱线图"""
        palette = self.get_palette()
        config = self.config['boxplot']
        
        box_data = [data_baseline, data_engage]
        colors = [palette['Baseline'], palette['EngageCue']]
        
        # 绘制箱线图
        box_plot = ax.boxplot(box_data, labels=labels, 
                             patch_artist=True,
                             positions=[1.0, 1.75],
                             widths=0.45,
                             showfliers=False,
                             whis=[0, 100],
                             boxprops={'linewidth': config['box_lw']},
                             whiskerprops={'color': 'black', 'linewidth': config['whisker_lw']},
                             capprops={'color': 'black', 'linewidth': config['cap_lw']},
                             medianprops={'color': self.config['palette']['median'], 
                                        'linewidth': config['median_lw']})
        
        # 设置填充颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(config['face_alpha'])
            patch.set_edgecolor('black')
            patch.set_linewidth(config['edge_lw'])
            
        return box_plot
        
    def grouped_bars(self, ax, means_b: List[float], ses_b: List[float], 
                    means_e: List[float], ses_e: List[float], 
                    x_positions: List[float], width: float = None,
                    labels: Tuple[str, str] = ("Baseline", "EngageCue")):
        """统一样式的分组柱状图"""
        if width is None:
            width = self.config['bars']['width']
            
        palette = self.get_palette()
        bar_config = self.config['bars']
        
        # 使用更精致的柱子间距
        spacing = bar_config.get('spacing', 0.35)
        x_baseline = [x - spacing/2 for x in x_positions]
        x_engage = [x + spacing/2 for x in x_positions]
        
        # 绘制柱状图
        bars1 = ax.bar(x_baseline, means_b, width, 
                      color=palette['Baseline'], alpha=bar_config['alpha'],
                      label=labels[0], edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x_engage, means_e, width,
                      color=palette['EngageCue'], alpha=bar_config['alpha'],
                      label=labels[1], edgecolor='white', linewidth=0.8)
        
        # 添加误差线
        ax.errorbar(x_baseline, means_b, yerr=ses_b, 
                   fmt='none', color=self.config['palette']['error'],
                   capsize=bar_config['error_capsize'], 
                   elinewidth=bar_config['error_elinewidth'])
        ax.errorbar(x_engage, means_e, yerr=ses_e,
                   fmt='none', color=self.config['palette']['error'],
                   capsize=bar_config['error_capsize'], 
                   elinewidth=bar_config['error_elinewidth'])
        
        # 返回x坐标和顶端y值用于后续标注
        x_coords = x_baseline + x_engage
        y_tops = [m + s for m, s in zip(means_b, ses_b)] + [m + s for m, s in zip(means_e, ses_e)]
        
        return x_coords, y_tops
        
    def save_figure(self, fig, filepath: Union[str, Path], formats: List[str] = ['png', 'pdf']):
        """统一保存图表"""
        filepath = Path(filepath)
        config = self.config['matplotlib']
        
        for fmt in formats:
            output_path = filepath.with_suffix(f'.{fmt}')
            if fmt == 'pdf':
                fig.savefig(output_path, format='pdf',
                          bbox_inches=config['save_bbox'],
                          pad_inches=config['save_pad_inches'],
                          facecolor='white')
            else:
                fig.savefig(output_path, dpi=config['dpi'],
                          bbox_inches=config['save_bbox'],
                          pad_inches=config['save_pad_inches'],
                          facecolor='white')
            print(f"✓ 图表已保存: {output_path}")

# 创建全局实例
_style_kit = ChartStyleKit()

# 导出函数接口
def apply_style():
    """设定全局matplotlib参数"""
    _style_kit.apply_style()

def get_palette() -> Dict[str, str]:
    """返回调色板"""
    return _style_kit.get_palette()

def add_headroom(ax, top_ratio: float = None):
    """为y轴添加顶部留白"""
    _style_kit.add_headroom(ax, top_ratio)

def annotate_values(ax, xs: List[float], ys: List[float], 
                   errs: List[float] = None, fmt: str = "%.2f", dy: float = 0.02):
    """在柱顶/箱顶标注数值"""
    _style_kit.annotate_values(ax, xs, ys, errs, fmt, dy)

def annotate_significance(ax, text: str = None, y_rel: float = None):
    """在面板顶部居中标记显著性"""
    _style_kit.annotate_significance(ax, text, y_rel)

def annotate_n(ax, n_text: str):
    """在x轴下方标注样本量"""
    _style_kit.annotate_n(ax, n_text)

def style_legend(ax, labels: List[str] = None):
    """设置右上角半透明图例"""
    _style_kit.style_legend(ax, labels)

def style_axes(ax, xlabel: str = None, ylabel: str = None, title: str = None):
    """统一轴与标题风格"""
    _style_kit.style_axes(ax, xlabel, ylabel, title)

def boxplot(ax, data_baseline: List[float], data_engage: List[float], 
           labels: Tuple[str, str] = ("Baseline", "EngageCue")):
    """统一样式的箱线图"""
    return _style_kit.boxplot(ax, data_baseline, data_engage, labels)

def grouped_bars(ax, means_b: List[float], ses_b: List[float], 
                means_e: List[float], ses_e: List[float], 
                x_positions: List[float], width: float = None,
                labels: Tuple[str, str] = ("Baseline", "EngageCue")):
    """统一样式的分组柱状图"""
    return _style_kit.grouped_bars(ax, means_b, ses_b, means_e, ses_e, x_positions, width, labels)

def save_figure(fig, filepath: Union[str, Path], formats: List[str] = ['png', 'pdf']):
    """统一保存图表"""
    _style_kit.save_figure(fig, filepath, formats)

def set_style_config(config_path: str):
    """设置自定义配置文件路径"""
    global _style_kit
    _style_kit = ChartStyleKit(config_path)

# 导出精致柱状图参数常量
def get_refined_bar_params():
    """获取精致柱状图参数"""
    return {
        'width': REFINED_BAR_WIDTH,
        'spacing': REFINED_BAR_SPACING
    }
