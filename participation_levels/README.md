# 参与度水平分析脚本

## 功能概述

6个维度的参与度水平分析：
1. **Speaking Duration** (sec/min) - 发言时长
2. **Speaking Frequency** (count/min) - 发言次数
3. **Editing Activity** (chars/min) - 编辑活跃度
4. **Browsing Activity** (ops/min) - 浏览活跃度
5. **Posture Score** (1-5) - 姿态评分
6. **Gaze Score** (1-5) - 视线评分

## 使用方法

```bash
python participation_levels_analyzer.py --minutes 30
```

其中 `--minutes` 参数是必需的，指定任务时长（分钟）。

## 输出文件

### 图表文件
- `figures/participation_levels_2x3.png` - PNG格式图表（300 DPI）
- `figures/participation_levels_2x3.pdf` - PDF格式图表

### 数据文件
- `integrated_levels_summary.csv` - 汇总统计（Mean, SE, N按条件）
- `paired_tests_per_metric.csv` - 配对检验结果（t统计量、p值、效应量）

## 数据源

- **发言数据**: `speech/paired_speech_data.csv`
- **编辑数据**: `edit_note/engagement_analysis.csv`  
- **浏览数据**: `page_behavior/analysis/group_paired_results.csv`
- **视频标注**: `video_annotation/paired_participation_by_member_type.csv`

## 分析特点

1. **按维度独立配对**：每个维度保留该维度内Baseline/EngageCue都存在的组
2. **SE计算准确**：SE=SD/√n，n为该维度的配对组数  
3. **2×3小面板布局**：每个指标独立Y轴，避免量纲冲突
4. **专业学术格式**：包含误差线、显著性标记、样本量标注

## 图表说明

- **颜色**: Baseline #64B5F6, EngageCue #F48FB1
- **误差线**: 基于配对组的标准误
- **标注**: 柱顶Mean值、上方"n.s."、下方"n=XX"
- **警告**: 不同面板量纲不同，不可跨指标比较

