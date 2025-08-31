# 专家评分可视化器

## 功能概述

生成两个专家评分对比图：
1. **Figure Z（正文）**: 合并两位专家评分的分组柱状图
2. **Figure Z-suppl（附录）**: 分专家的2×3小面板图

## 使用方法

```bash
cd expert_scoring
python expert_rating_visualizer.py
```

## 输出文件

### 图表文件
- `figures/expert_ratings_combined.png/pdf` - 合并专家评分图（正文用）
- `figures/expert_ratings_by_expert.png/pdf` - 分专家面板图（附录用）

### 数据文件
- `expert_ratings_summary.csv` - 汇总统计数据
- `expert_ratings_captions.txt` - 图表说明文字

## 数据源

- **输入**: `expert_scoring/intermediate/summary_descriptives.csv`
- **专家**: E1和E2的独立评分
- **维度**: Process Score, Outcome Score, Total Score（Overall）
- **条件**: Baseline (NoAI) vs EngageCue (AI)

## 分析特点

1. **自动量表检测**: 根据数据自动设置Y轴范围（1-5或1-7）
2. **严谨合并方法**: 使用pooled variance计算合并标准误
3. **加权平均**: 当专家评分数量不同时自动使用加权平均
4. **统一视觉风格**: 与participation_levels脚本保持一致的配色和样式

## 图表说明

### Figure Z（合并图）
- **尺寸**: 7×5英寸，适合论文双栏
- **布局**: 3个维度的分组柱状图
- **数据**: 两位专家评分的合并结果

### Figure Z-suppl（分专家图）
- **尺寸**: 11×6英寸，适合附录展示
- **布局**: 2×3小面板（每行一位专家，每列一个维度）
- **数据**: 各专家的独立评分结果

## 统计方法

- **合并均值**: 加权平均（n1≠n2时）或简单平均（n1=n2时）
- **标准误**: 基于pooled variance的严谨计算
- **显著性**: 统一标注"n.s."（基于原始分析结果）

