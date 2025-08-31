# 基尼系数三联图生成器

## 功能描述
生成三联图（Speaking/Editing/Browsing），每个子图显示 Baseline vs EngageCue 的基尼系数箱线图对比。

## 图表特性
- **三联图布局**：1行3列，figsize=(10.2, 3.6)
- **统一y轴**：0-0.5范围
- **显著性标记**：
  - Speaking: "* p = .044"
  - Editing: "n.s."
  - Browsing: "n.s."
- **箱线图元素**：红色中位数线、数值标注、抖动散点
- **输出格式**：300 dpi PNG，适合CHI双栏排版

## 输入数据源
脚本会自动读取以下文件：
1. `../speech/paired_speech_data.csv` - 发言数据
2. `../edit_note/gini_analysis.csv` - 笔记编辑数据  
3. `../page_behavior/analysis/balance_gini_paired_results.csv` - 页面行为数据

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行脚本
```bash
cd integrated_gini
python generate_gini_triptych.py
```

### 3. 输出文件
- `figures/gini_overall_triptych.png` - 三联图（300 dpi）
- `integrated_gini_summary.csv` - 整合后的数据表

## 数据格式要求

### 输入数据列名
- **speech**: `ai_gini`, `noai_gini`
- **edit_note**: `Group`, `Type`, `Gini_Coefficient`
- **page_behavior**: `Group`, `Gini_AI`, `Gini_NoAI`

### 输出数据格式
- `Domain`: Speaking/Editing/Browsing
- `Condition`: Baseline/EngageCue
- `Gini`: 基尼系数值

## 图表样式
- 黑白/低饱和配色，适合打印
- 清晰的网格线和边框
- 统一的字体大小和标签样式
- 底部说明："Lower Gini = more balanced"

## 注意事项
- 确保输入数据文件存在且格式正确
- 脚本会自动处理缺失数据，显示警告信息
- 如果某个域的数据缺失，会显示"No Data"提示
- 生成的图表适合学术论文发表使用

