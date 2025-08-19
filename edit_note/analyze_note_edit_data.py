#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
笔记编辑数据分析脚本

分析预处理后的数据，计算参与度、平衡度等指标，
进行统计检验并生成可视化图表和HTML报告。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_rel, wilcoxon
import warnings
from pathlib import Path
import jinja2
import os

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

class NoteEditAnalyzer:
    """笔记编辑数据分析器"""
    
    def __init__(self, data_path):
        """
        初始化分析器
        
        Args:
            data_path (str): 预处理后的数据文件路径
        """
        self.data_path = data_path
        self.df = None
        self.engagement_df = None
        self.gini_df = None
        self.stats_results = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成，共 {len(self.df)} 条记录")
        print(f"数据列: {list(self.df.columns)}")
        return self.df
    
    def calculate_net_chars(self):
        """
        计算每个成员的净产出
        注意：这里假设Net_Chars已经是净产出，如果需要重新计算请修改逻辑
        """
        print("正在计算净产出...")
        
        # 如果Net_Chars已经是净产出，直接使用
        if 'Net_Chars' in self.df.columns:
            self.df['Net_Chars_Calculated'] = self.df['Net_Chars']
        else:
            # 这里可以添加计算净产出的逻辑
            # 例如：最后一个时间窗口 - 第一个时间窗口
            pass
        
        return self.df
    
    def calculate_engagement(self, task_duration_minutes=30):
        """
        计算每个成员的参与度
        
        Args:
            task_duration_minutes (int): 任务总时长（分钟）
        """
        print("正在计算参与度...")
        
        # 复制数据框
        self.engagement_df = self.df.copy()
        
        # 计算参与度
        self.engagement_df['Engagement'] = (
            self.engagement_df['Net_Chars'] / task_duration_minutes
        )
        
        # 添加净产出列（如果不存在）
        if 'Net_Chars_Calculated' not in self.engagement_df.columns:
            self.engagement_df['Net_Chars_Calculated'] = self.engagement_df['Net_Chars']
        
        print(f"参与度计算完成，范围: {self.engagement_df['Engagement'].min():.2f} - {self.engagement_df['Engagement'].max():.2f}")
        
        return self.engagement_df
    
    def calculate_gini_coefficient(self, values):
        """
        计算Gini系数
        
        Args:
            values (array-like): 数值数组
            
        Returns:
            float: Gini系数
        """
        if len(values) <= 1:
            return 0.0
        
        values = np.array(values)
        n = len(values)
        
        if np.mean(values) == 0:
            return 0.0
        
        # 计算所有两两差值
        diff_sum = 0
        for i in range(n):
            for j in range(n):
                diff_sum += abs(values[i] - values[j])
        
        gini = diff_sum / (2 * n * n * np.mean(values))
        return gini
    
    def calculate_group_gini(self):
        """计算每个小组的Gini系数"""
        print("正在计算小组Gini系数...")
        
        gini_results = []
        
        for group in self.engagement_df['Group'].unique():
            group_data = self.engagement_df[self.engagement_df['Group'] == group]
            
            # 分别计算AI和NoAI条件下的Gini系数
            for exp_type in ['AI', 'NoAI']:
                type_data = group_data[group_data['Type'] == exp_type]
                
                if len(type_data) > 0:
                    net_chars = type_data['Net_Chars'].values
                    gini = self.calculate_gini_coefficient(net_chars)
                    
                    gini_results.append({
                        'Group': group,
                        'Type': exp_type,
                        'Gini_Coefficient': gini,
                        'Member_Count': len(type_data),
                        'Total_Net_Chars': np.sum(net_chars),
                        'Mean_Net_Chars': np.mean(net_chars)
                    })
        
        self.gini_df = pd.DataFrame(gini_results)
        print(f"Gini系数计算完成，共 {len(self.gini_df)} 个小组-条件组合")
        
        return self.gini_df
    
    def perform_statistical_tests(self):
        """执行统计检验"""
        print("正在执行统计检验...")
        
        # 准备配对数据
        paired_data = []
        
        for group in self.engagement_df['Group'].unique():
            group_data = self.engagement_df[self.engagement_df['Group'] == group]
            
            ai_data = group_data[group_data['Type'] == 'AI']
            noai_data = group_data[group_data['Type'] == 'NoAI']
            
            if len(ai_data) > 0 and len(noai_data) > 0:
                # 计算差值
                for _, ai_row in ai_data.iterrows():
                    for _, noai_row in noai_data.iterrows():
                        if ai_row['Speaker'] == noai_row['Speaker']:
                            paired_data.append({
                                'Group': group,
                                'Speaker': ai_row['Speaker'],
                                'AI_Engagement': ai_row['Engagement'],
                                'NoAI_Engagement': noai_row['Engagement'],
                                'AI_Net_Chars': ai_row['Net_Chars'],
                                'NoAI_Net_Chars': noai_row['Net_Chars'],
                                'Engagement_Diff': ai_row['Engagement'] - noai_row['Engagement'],
                                'Net_Chars_Diff': ai_row['Net_Chars'] - noai_row['Net_Chars']
                            })
        
        paired_df = pd.DataFrame(paired_data)
        
        if len(paired_df) == 0:
            print("警告：没有找到配对数据")
            return None
        
        # 正态性检验
        engagement_diff = paired_df['Engagement_Diff'].dropna()
        net_chars_diff = paired_df['Net_Chars_Diff'].dropna()
        
        # 参与度差值的正态性检验
        if len(engagement_diff) >= 3:
            shapiro_engagement = shapiro(engagement_diff)
            engagement_normal = shapiro_engagement.pvalue > 0.05
            engagement_shapiro_p = shapiro_engagement.pvalue
        else:
            engagement_normal = False
            engagement_shapiro_p = np.nan
        
        # 净产出差值的正态性检验
        if len(net_chars_diff) >= 3:
            shapiro_net_chars = shapiro(net_chars_diff)
            net_chars_normal = shapiro_net_chars.pvalue > 0.05
            net_chars_shapiro_p = shapiro_net_chars.pvalue
        else:
            net_chars_normal = False
            net_chars_shapiro_p = np.nan
        
        # 统计检验
        results = {}
        
        # 参与度检验
        if engagement_normal:
            t_stat, t_pvalue = ttest_rel(paired_df['AI_Engagement'], paired_df['NoAI_Engagement'])
            test_type = 'Paired t-test'
            test_stat = t_stat
            test_pvalue = t_pvalue
        else:
            w_stat, w_pvalue = wilcoxon(paired_df['AI_Engagement'], paired_df['NoAI_Engagement'])
            test_type = 'Wilcoxon signed-rank test'
            test_stat = w_stat
            test_pvalue = w_pvalue
        
        # 计算效应量 (Cohen's d)
        cohens_d = self.calculate_cohens_d(paired_df['AI_Engagement'], paired_df['NoAI_Engagement'])
        
        results['Engagement'] = {
            'test_type': test_type,
            'test_statistic': test_stat,
            'p_value': test_pvalue,
            'is_normal': engagement_normal,
            'shapiro_p_value': engagement_shapiro_p,
            'cohens_d': cohens_d,
            'n_pairs': len(paired_df)
        }
        
        # 净产出检验
        if net_chars_normal:
            t_stat, t_pvalue = ttest_rel(paired_df['AI_Net_Chars'], paired_df['NoAI_Net_Chars'])
            test_type = 'Paired t-test'
            test_stat = t_stat
            test_pvalue = t_pvalue
        else:
            w_stat, w_pvalue = wilcoxon(paired_df['AI_Net_Chars'], paired_df['NoAI_Net_Chars'])
            test_type = 'Wilcoxon signed-rank test'
            test_stat = w_stat
            test_pvalue = w_pvalue
        
        cohens_d = self.calculate_cohens_d(paired_df['AI_Net_Chars'], paired_df['NoAI_Net_Chars'])
        
        results['Net_Chars'] = {
            'test_type': test_type,
            'test_statistic': test_stat,
            'p_value': test_pvalue,
            'is_normal': net_chars_normal,
            'shapiro_p_value': net_chars_shapiro_p,
            'cohens_d': cohens_d,
            'n_pairs': len(paired_df)
        }
        
        self.stats_results = results
        print("统计检验完成")
        
        return results
    
    def calculate_cohens_d(self, group1, group2):
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return cohens_d
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("正在创建可视化图表...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Note Edit Data Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. 参与度分布箱线图
        ax1 = axes[0, 0]
        sns.boxplot(data=self.engagement_df, x='Type', y='Engagement', ax=ax1)
        ax1.set_title('Engagement Distribution by Condition')
        ax1.set_xlabel('Experimental Condition')
        ax1.set_ylabel('Engagement (chars/min)')
        
        # 2. 参与度分布小提琴图
        ax2 = axes[0, 1]
        sns.violinplot(data=self.engagement_df, x='Type', y='Engagement', ax=ax2)
        ax2.set_title('Engagement Distribution (Violin Plot)')
        ax2.set_xlabel('Experimental Condition')
        ax2.set_ylabel('Engagement (chars/min)')
        
        # 3. 小组Gini系数对比条形图
        ax3 = axes[1, 0]
        if self.gini_df is not None:
            gini_pivot = self.gini_df.pivot(index='Group', columns='Type', values='Gini_Coefficient')
            gini_pivot.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
            ax3.set_title('Gini Coefficient by Group and Condition')
            ax3.set_xlabel('Group')
            ax3.set_ylabel('Gini Coefficient')
            ax3.legend(title='Condition')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 净产出分布直方图
        ax4 = axes[1, 1]
        for exp_type in ['AI', 'NoAI']:
            type_data = self.engagement_df[self.engagement_df['Type'] == exp_type]['Net_Chars']
            ax4.hist(type_data, alpha=0.7, label=exp_type, bins=20)
        ax4.set_title('Net Characters Distribution by Condition')
        ax4.set_xlabel('Net Characters')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存图表
        output_path = Path(self.data_path).parent / 'note_edit_analysis_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
        
        return output_path
    
    def generate_descriptive_stats(self):
        """生成描述性统计"""
        print("正在生成描述性统计...")
        
        # 按条件分组的描述性统计
        desc_stats = self.engagement_df.groupby('Type').agg({
            'Engagement': ['count', 'mean', 'std', 'min', 'max'],
            'Net_Chars': ['mean', 'std', 'min', 'max'],
            'Total_Edit_Count': ['mean', 'std'],
            'Active_Time_Windows': ['mean', 'std']
        }).round(3)
        
        # 扁平化列名
        desc_stats.columns = ['_'.join(col).strip() for col in desc_stats.columns]
        
        # 保存描述性统计
        output_path = Path(self.data_path).parent / 'descriptive_statistics.csv'
        desc_stats.to_csv(output_path)
        print(f"描述性统计已保存到: {output_path}")
        
        return desc_stats
    
    def save_analysis_results(self):
        """保存分析结果"""
        print("正在保存分析结果...")
        
        base_path = Path(self.data_path).parent
        
        # 保存参与度数据
        if self.engagement_df is not None:
            engagement_path = base_path / 'engagement_analysis.csv'
            self.engagement_df.to_csv(engagement_path, index=False)
            print(f"参与度分析结果已保存到: {engagement_path}")
        
        # 保存Gini系数数据
        if self.gini_df is not None:
            gini_path = base_path / 'gini_analysis.csv'
            self.gini_df.to_csv(gini_path, index=False)
            print(f"Gini系数分析结果已保存到: {gini_path}")
        
        # 保存统计检验结果
        if self.stats_results is not None:
            stats_path = base_path / 'statistical_results.csv'
            
            # 转换统计结果为DataFrame
            stats_data = []
            for metric, results in self.stats_results.items():
                stats_data.append({
                    'Metric': metric,
                    'Test_Type': results['test_type'],
                    'Test_Statistic': results['test_statistic'],
                    'P_Value': results['p_value'],
                    'Is_Normal': results['is_normal'],
                    'Shapiro_Wilk_P_Value': results['shapiro_p_value'],
                    'Cohens_d': results['cohens_d'],
                    'N_Pairs': results['n_pairs']
                })
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_csv(stats_path, index=False)
            print(f"统计检验结果已保存到: {stats_path}")
    
    def generate_html_report(self):
        """生成HTML报告"""
        print("正在生成HTML报告...")
        
        # HTML模板
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Note Edit Data Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .table-container { overflow-x: auto; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .highlight { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; }
        .normal-test { background-color: #d1ecf1; padding: 15px; border-radius: 5px; border-left: 4px solid #17a2b8; }
        .plot { text-align: center; margin: 20px 0; }
        .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .footer { margin-top: 50px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; text-align: center; color: #6c757d; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Note Edit Data Analysis Report</h1>
        <p><strong>Analysis Date:</strong> {{ analysis_date }}</p>
        <p><strong>Data Source:</strong> {{ data_source }}</p>
        <p><strong>Total Records:</strong> {{ total_records }}</p>
    </div>

    <div class="section">
        <h2>🔍 Study Overview</h2>
        <p>This analysis examines the impact of AI assistance on collaborative note editing behavior, focusing on:</p>
        <ul>
            <li><strong>Engagement:</strong> Individual participation level (characters per minute)</li>
            <li><strong>Balance:</strong> Group participation distribution (Gini coefficient)</li>
            <li><strong>Comparison:</strong> AI vs NoAI conditions</li>
        </ul>
    </div>

    <div class="section">
        <h2>📈 Descriptive Statistics</h2>
        <div class="table-container">
            {{ descriptive_stats_table }}
        </div>
    </div>

    <div class="section">
        <h2>🔬 Normality Test Results</h2>
        <div class="normal-test">
            <h3>Shapiro-Wilk Normality Test for Differences (AI - NoAI):</h3>
            {{ normality_test_table }}
        </div>
        <p><em>Note: p > 0.05 indicates normal distribution. Normal data uses paired t-test, non-normal data uses Wilcoxon signed-rank test.</em></p>
    </div>

    <div class="section">
        <h2>🧮 Statistical Analysis Results</h2>
        <div class="table-container">
            {{ statistical_results_table }}
        </div>
        
        <div class="highlight">
            <h3>Key Findings:</h3>
            <ul>
                {% for metric, results in key_findings.items() %}
                <li><strong>{{ metric }}:</strong> {{ results.summary }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>📊 Visualizations</h2>
        <div class="plot">
            <img src="{{ plot_image }}" alt="Analysis Plots">
            <p><em>Figure: Engagement distribution, Gini coefficients, and net characters distribution by experimental condition</em></p>
        </div>
    </div>

    <div class="section">
        <h2>📋 Data Quality Information</h2>
        <ul>
            <li><strong>Missing Values:</strong> {{ missing_values_info }}</li>
            <li><strong>Data Range:</strong> {{ data_range_info }}</li>
            <li><strong>Outliers:</strong> {{ outliers_info }}</li>
        </ul>
    </div>

    <div class="footer">
        <p>Report generated automatically by Note Edit Data Analyzer</p>
        <p>For questions or issues, please contact the research team</p>
    </div>
</body>
</html>
        """
        
        # 准备报告数据
        if self.engagement_df is not None:
            total_records = len(self.engagement_df)
            missing_values = self.engagement_df.isnull().sum().sum()
            data_range = f"Engagement: {self.engagement_df['Engagement'].min():.2f} - {self.engagement_df['Engagement'].max():.2f}"
        else:
            total_records = 0
            missing_values = 0
            data_range = "N/A"
        
        # 生成描述性统计表格
        if hasattr(self, 'engagement_df') and self.engagement_df is not None:
            desc_stats = self.generate_descriptive_stats()
            descriptive_stats_table = desc_stats.to_html(classes='table table-striped')
        else:
            descriptive_stats_table = "<p>No descriptive statistics available</p>"
        
        # 生成统计结果表格
        if self.stats_results is not None:
            stats_data = []
            for metric, results in self.stats_results.items():
                stats_data.append({
                    'Metric': metric,
                    'Test Type': results['test_type'],
                    'P-Value': f"{results['p_value']:.4f}",
                    'Effect Size (Cohen\'s d)': f"{results['cohens_d']:.3f}",
                    'Sample Size': results['n_pairs']
                })
            
            stats_df = pd.DataFrame(stats_data)
            statistical_results_table = stats_df.to_html(classes='table table-striped', index=False)
            
            # 生成关键发现
            key_findings = {}
            for metric, results in self.stats_results.items():
                if results['p_value'] < 0.05:
                    significance = "Significant"
                else:
                    significance = "Not significant"
                
                key_findings[metric] = {
                    'summary': f"{significance} difference (p={results['p_value']:.4f}, d={results['cohens_d']:.3f})"
                }
        else:
            statistical_results_table = "<p>No statistical results available</p>"
            key_findings = {}
        
        # 生成正态性检验表格
        if self.stats_results is not None:
            normality_data = []
            for metric, results in self.stats_results.items():
                # 格式化正态性检验p值
                if pd.notna(results['shapiro_p_value']):
                    shapiro_p_formatted = f"{results['shapiro_p_value']:.4f}"
                else:
                    shapiro_p_formatted = "N/A"
                
                normality_data.append({
                    'Metric': metric,
                    'Is_Normal': 'Yes' if results['is_normal'] else 'No',
                    'Shapiro_Wilk_p': shapiro_p_formatted,
                    'Test_Used': results['test_type'],
                    'Interpretation': 'Normal distribution - Paired t-test used' if results['is_normal'] else 'Non-normal distribution - Wilcoxon test used'
                })
            
            normality_df = pd.DataFrame(normality_data)
            normality_test_table = normality_df.to_html(classes='table table-striped', index=False)
        else:
            normality_test_table = "<p>No normality test results available</p>"
        
        # 渲染HTML
        template = jinja2.Template(html_template)
        html_content = template.render(
            analysis_date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            data_source=self.data_path,
            total_records=total_records,
            descriptive_stats_table=descriptive_stats_table,
            statistical_results_table=statistical_results_table,
            normality_test_table=normality_test_table,
            key_findings=key_findings,
            plot_image='note_edit_analysis_plots.png',
            missing_values_info=f"{missing_values} missing values found",
            data_range_info=data_range,
            outliers_info="Outlier detection not implemented in this version"
        )
        
        # 保存HTML报告
        output_path = Path(self.data_path).parent / 'note_edit_analysis_report.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {output_path}")
        return output_path
    
    def run_full_analysis(self, task_duration_minutes=30):
        """运行完整分析流程"""
        print("="*60)
        print("开始运行完整分析流程")
        print("="*60)
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 计算净产出
            self.calculate_net_chars()
            
            # 3. 计算参与度
            self.calculate_engagement(task_duration_minutes)
            
            # 4. 计算Gini系数
            self.calculate_group_gini()
            
            # 5. 执行统计检验
            self.perform_statistical_tests()
            
            # 6. 创建可视化
            self.create_visualizations()
            
            # 7. 保存结果
            self.save_analysis_results()
            
            # 8. 生成HTML报告
            self.generate_html_report()
            
            print("\n" + "="*60)
            print("分析完成！")
            print("="*60)
            
            # 显示关键结果摘要
            if self.stats_results:
                print("\n关键统计结果:")
                for metric, results in self.stats_results.items():
                    print(f"{metric}: {results['test_type']}, p={results['p_value']:.4f}, d={results['cohens_d']:.3f}")
            
        except Exception as e:
            print(f"分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("笔记编辑数据分析脚本")
    print("="*60)
    
    # 设置文件路径
    current_dir = Path(__file__).parent
    input_file = current_dir / "note_edit_summary_by_member.csv"
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 创建分析器并运行分析
    analyzer = NoteEditAnalyzer(str(input_file))
    analyzer.run_full_analysis(task_duration_minutes=30)

if __name__ == "__main__":
    main()
