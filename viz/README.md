# 统一图表样式工具包

## 概述

这个模块提供了统一的图表样式配置，确保项目中所有图表具有一致的视觉风格。

## 文件结构

```
viz/
├── README.md           # 本文档
├── chart_config.yaml   # 样式配置文件
└── style_kit.py       # 核心样式工具包
```

## 使用方法

### 1. 基本导入和应用

```python
from viz.style_kit import (
    apply_style, get_palette, add_headroom, annotate_values,
    annotate_significance, annotate_n, style_legend, style_axes,
    grouped_bars, boxplot, save_figure
)

# 应用统一样式
apply_style()
```

### 2. 主要函数说明

#### 基础样式函数

- `apply_style()`: 应用全局matplotlib配置
- `get_palette()`: 获取统一的调色板
- `style_axes(ax, xlabel, ylabel, title)`: 设置统一的轴样式
- `style_legend(ax, labels)`: 设置统一的图例样式

#### 图表元素标注

- `annotate_values(ax, xs, ys, errs, fmt)`: 标注数值
- `annotate_significance(ax, text, y_rel)`: 标注显著性
- `annotate_n(ax, n_text)`: 标注样本量
- `add_headroom(ax, top_ratio)`: 添加顶部留白

#### 绘图函数

- `boxplot(ax, data_baseline, data_engage, labels)`: 统一样式箱线图
- `grouped_bars(ax, means_b, ses_b, means_e, ses_e, x_positions)`: 统一样式分组柱状图（精致版）
- `save_figure(fig, filepath, formats)`: 统一保存图表
- `get_refined_bar_params()`: 获取精致柱状图参数

### 3. 配置说明

配置文件 `chart_config.yaml` 包含以下主要参数：

- **matplotlib**: 字体、字号、线宽等matplotlib全局设置
- **palette**: 颜色方案（Baseline: 浅蓝, EngageCue: 浅粉）
- **bars**: 柱状图相关参数
- **boxplot**: 箱线图相关参数
- **text**: 文本标注相关参数
- **layout**: 布局相关参数

### 4. 已统一的脚本

以下4个脚本已经统一使用新的样式系统：

1. `integrated_gini/generate_gini_triptych.py` - 基尼系数三联图
2. `participation_levels/participation_levels_analyzer.py` - 参与度分析
3. `expert_scoring/expert_rating_visualizer.py` - 专家评分可视化
4. `self_scale/qcge_visualizer.py` - QCGE-SAS自评量表

## 特性

### 统一的视觉元素

- **颜色方案**: Baseline (#64B5F6) vs EngageCue (#F48FB1)
- **字体**: DejaVu Sans 系列
- **柱状图**: 精致柱宽(0.28)，优雅间距(0.35)
- **误差线**: 统一的capsize=3, linewidth=1.1
- **显著性标记**: 统一的"n.s."和"* p = .XXX"格式

### 标准化的输出

- **文件格式**: 同时生成PNG (300dpi) 和PDF格式
- **文件命名**: 规范化的命名规则
- **保存参数**: 统一的bbox_inches='tight'等参数

### 灵活的配置

- 支持通过YAML文件自定义样式参数
- 可以通过`set_style_config()`函数使用自定义配置文件

## 扩展使用

如果需要为特定图表自定义样式，可以：

1. 复制 `chart_config.yaml` 并修改参数
2. 使用 `set_style_config(custom_config_path)` 加载自定义配置
3. 或者直接修改现有配置文件

## 测试

运行 `test_unified_styles.py` 可以测试样式工具包的功能。
