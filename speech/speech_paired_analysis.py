#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发言指标配对分析脚本
进行配对t检验、描述性统计、正态性检验和可视化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def create_paired_data(df):
    """创建配对数据，计算AI - NoAI的差值"""
    ai_groups = df[df['condition'] == 'AI'].copy()
    noai_groups = df[df['condition'] == 'NoAI'].copy()
    
    paired_data = []
    for _, ai_row in ai_groups.iterrows():
        group_id = ai_row['group_id']
        noai_row = noai_groups[noai_groups['group_id'] == group_id]
        
        if not noai_row.empty:
            diff_total = ai_row['speak_total_s'] - noai_row.iloc[0]['speak_total_s']
            diff_count = ai_row['speak_count'] - noai_row.iloc[0]['speak_count']
            diff_gini = ai_row['gini_speak'] - noai_row.iloc[0]['gini_speak']
            
            paired_data.append({
                'group_id': group_id,
                'ai_total': ai_row['speak_total_s'],
                'noai_total': noai_row.iloc[0]['speak_total_s'],
                'ai_count': ai_row['speak_count'],
                'noai_count': noai_row.iloc[0]['speak_count'],
                'ai_gini': ai_row['gini_speak'],
                'noai_gini': noai_row.iloc[0]['gini_speak'],
                'diff_total': diff_total,
                'diff_count': diff_count,
                'diff_gini': diff_gini
            })
    
    paired_df = pd.DataFrame(paired_data)
    diff_df = paired_df[['group_id', 'diff_total', 'diff_count', 'diff_gini']]
    
    return paired_df, diff_df


def descriptive_statistics(paired_df):
    """计算描述性统计"""
    ai_stats = paired_df[['ai_total', 'ai_count', 'ai_gini']].describe()
    noai_stats = paired_df[['noai_total', 'noai_count', 'noai_gini']].describe()
    diff_stats = paired_df[['diff_total', 'diff_count', 'diff_gini']].describe()
    
    desc_stats = pd.concat({
        'AI组': ai_stats,
        'NoAI组': noai_stats,
        '差值(AI-NoAI)': diff_stats
    }, axis=1)
    
    return desc_stats


def normality_analysis(diff_df):
    """对差值进行正态性检验"""
    normality_results = {}
    
    for col in ['diff_total', 'diff_count', 'diff_gini']:
        data = diff_df[col].dropna()
        
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        stats.probplot(data, dist="norm", plot=ax1)
        ax1.set_title(f'{col.replace("diff_", "").replace("_", " ").title()} Q-Q图')
        
        ax2.hist(data, bins=10, alpha=0.7, edgecolor='black')
        ax2.set_title(f'{col.replace("diff_", "").replace("_", " ").title()} 差值分布图')
        ax2.set_xlabel('差值')
        ax2.set_ylabel('频数')
        
        plt.tight_layout()
        plt.savefig(f'speech/normality_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        normality_results[col] = {
            'shapiro_statistic': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }
    
    return normality_results


def paired_ttest_analysis(paired_df):
    """执行配对t检验"""
    ttest_results = {}
    
    metrics = [
        ('speak_total_s', 'ai_total', 'noai_total'),
        ('speak_count', 'ai_count', 'noai_count'),
        ('gini_speak', 'ai_gini', 'noai_gini')
    ]
    
    for metric_name, ai_col, noai_col in metrics:
        ai_data = paired_df[ai_col]
        noai_data = paired_df[noai_col]
        
        t_stat, p_value = stats.ttest_rel(ai_data, noai_data)
        
        diff = ai_data - noai_data
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        n = len(diff)
        t_critical = stats.t.ppf(0.975, n-1)
        ci_lower = mean_diff - t_critical * (std_diff / np.sqrt(n))
        ci_upper = mean_diff + t_critical * (std_diff / np.sqrt(n))
        
        ttest_results[metric_name] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_difference': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05
        }
    
    return ttest_results


def create_visualizations(paired_df, diff_df):
    """创建可视化图表"""
    # 配对线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['speak_total_s', 'speak_count', 'gini_speak']
    ai_cols = ['ai_total', 'ai_count', 'ai_gini']
    noai_cols = ['noai_total', 'noai_count', 'noai_gini']
    metric_names = ['发言总时长(秒)', '发言次数(段落数)', '发言均衡性(Gini系数)']
    
    for i, (metric, ai_col, noai_col, metric_name) in enumerate(zip(metrics, ai_cols, noai_cols, metric_names)):
        ax = axes[i]
        
        for _, row in paired_df.iterrows():
            ax.plot([1, 2], [row[noai_col], row[ai_col]], 'o-', alpha=0.6)
        
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['NoAI', 'AI'])
        ax.set_title(f'{metric_name} 配对对比')
        ax.set_ylabel(metric_name)
    
    plt.tight_layout()
    plt.savefig('speech/paired_lines_speech_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 箱线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, ai_col, noai_col, metric_name) in enumerate(zip(metrics, ai_cols, noai_cols, metric_names)):
        ax = axes[i]
        
        data_to_plot = [paired_df[noai_col], paired_df[ai_col]]
        ax.boxplot(data_to_plot, labels=['NoAI', 'AI'])
        ax.set_title(f'{metric_name} 分布对比')
        ax.set_ylabel(metric_name)
    
    plt.tight_layout()
    plt.savefig('speech/boxplots_speech_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 效应量森林图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    effect_sizes = []
    metric_labels = []
    
    for metric, ai_col, noai_col in zip(metrics, ai_cols, noai_cols):
        ai_data = paired_df[ai_col]
        noai_data = paired_df[noai_col]
        diff = ai_data - noai_data
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        effect_sizes.append(cohens_d)
        metric_labels.append(metric.replace('_', ' ').title())
    
    y_pos = np.arange(len(metric_labels))
    colors = ['red' if es < 0 else 'blue' for es in effect_sizes]
    
    bars = ax.barh(y_pos, effect_sizes, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Cohen's d 效应量")
    ax.set_title('各指标效应量森林图')
    
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='小效应')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='中等效应')
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='大效应')
    
    for i, (bar, es) in enumerate(zip(bars, effect_sizes)):
        ax.text(es + (0.05 if es >= 0 else -0.15), i, f'{es:.3f}', 
                va='center', ha='left' if es >= 0 else 'right')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('speech/effect_size_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def equilibrium_analysis(paired_df):
    """分析发言均衡性"""
    gini_diff = paired_df['ai_gini'] - paired_df['noai_gini']
    
    t_stat, p_value = stats.ttest_rel(paired_df['ai_gini'], paired_df['noai_gini'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.boxplot([paired_df['noai_gini'], paired_df['ai_gini']], 
                labels=['NoAI', 'AI'])
    ax1.set_title('Gini系数条件间对比')
    ax1.set_ylabel('Gini系数')
    
    ax2.hist(gini_diff, bins=10, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='无差异线')
    ax2.set_title('Gini系数差值分布 (AI - NoAI)')
    ax2.set_xlabel('差值')
    ax2.set_ylabel('频数')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('speech/gini_equilibrium_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'gini_difference_mean': np.mean(gini_diff),
        'gini_difference_std': np.std(gini_diff),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def save_results_to_csv(paired_df, diff_df, desc_stats, normality_results, ttest_results, equilibrium_results):
    """保存所有结果到CSV文件"""
    paired_df.to_csv('speech/paired_speech_data.csv', index=False, encoding='utf-8')
    diff_df.to_csv('speech/speech_differences.csv', index=False, encoding='utf-8')
    desc_stats.to_csv('speech/descriptive_statistics.csv', encoding='utf-8')
    
    normality_df = pd.DataFrame(normality_results).T
    normality_df.to_csv('speech/normality_test_results.csv', encoding='utf-8')
    
    ttest_df = pd.DataFrame(ttest_results).T
    ttest_df.to_csv('speech/paired_ttest_results.csv', encoding='utf-8')
    
    equilibrium_df = pd.DataFrame([equilibrium_results])
    equilibrium_df.to_csv('speech/equilibrium_analysis_results.csv', index=False, encoding='utf-8')
    
    print("所有CSV结果已保存完成！")


def generate_html_report(paired_df, diff_df, desc_stats, normality_results, ttest_results, equilibrium_results):
    """生成完整的HTML分析报告"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>发言指标配对分析报告</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background-color: #f8f9fa; }}
            .result-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            .result-table th, .result-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .result-table th {{ background-color: #3498db; color: white; }}
            .significant {{ color: #e74c3c; font-weight: bold; }}
            .not-significant {{ color: #27ae60; }}
            .metric {{ font-weight: bold; color: #8e44ad; }}
        </style>
    </head>
    <body>
        <h1>发言指标配对分析报告</h1>
        
        <div class="section">
            <h2>1. 研究概述</h2>
            <p>本报告分析了AI辅助与无AI辅助条件下，小组发言行为的差异。通过配对t检验比较了发言总时长、发言次数和发言均衡性三个关键指标。</p>
            <p><strong>样本量：</strong> {len(paired_df)} 个配对小组</p>
            <p><strong>分析指标：</strong> 发言总时长(秒)、发言次数(段落数)、发言均衡性(Gini系数)</p>
        </div>
        
        <div class="section">
            <h2>2. 描述性统计</h2>
            <h3>2.1 基本统计量</h3>
            {desc_stats.to_html(classes='result-table')}
        </div>
        
        <div class="section">
            <h2>3. 正态性检验结果</h2>
            <p>使用Shapiro-Wilk检验评估差值分布的正态性：</p>
            <table class="result-table">
                <tr>
                    <th>指标</th>
                    <th>Shapiro统计量</th>
                    <th>p值</th>
                    <th>是否正态</th>
                </tr>
    """
    
    for metric, results in normality_results.items():
        metric_name = metric.replace('diff_', '').replace('_', ' ').title()
        is_normal = "是" if results['is_normal'] else "否"
        normal_class = "not-significant" if results['is_normal'] else "significant"
        
        html_content += f"""
                <tr>
                    <td class="metric">{metric_name}</td>
                    <td>{results['shapiro_statistic']:.4f}</td>
                    <td>{results['shapiro_pvalue']:.4f}</td>
                    <td class="{normal_class}">{is_normal}</td>
                </tr>
        """
    
    html_content += """
            </table>
            <p><em>注：p > 0.05表示数据符合正态分布</em></p>
        </div>
        
        <div class="section">
            <h2>4. 配对t检验结果</h2>
            <p>比较AI组与NoAI组在各指标上的差异：</p>
            <table class="result-table">
                <tr>
                    <th>指标</th>
                    <th>t值</th>
                    <th>p值</th>
                    <th>效应量(Cohen's d)</th>
                    <th>均值差值</th>
                    <th>95%置信区间</th>
                    <th>显著性</th>
                </tr>
    """
    
    metric_names = {
        'speak_total_s': '发言总时长(秒)',
        'speak_count': '发言次数(段落数)',
        'gini_speak': '发言均衡性(Gini系数)'
    }
    
    for metric, results in ttest_results.items():
        metric_name = metric_names.get(metric, metric)
        significance = "显著" if results['significant'] else "不显著"
        sig_class = "significant" if results['significant'] else "not-significant"
        
        html_content += f"""
                <tr>
                    <td class="metric">{metric_name}</td>
                    <td>{results['t_statistic']:.4f}</td>
                    <td>{results['p_value']:.4f}</td>
                    <td>{results['cohens_d']:.4f}</td>
                    <td>{results['mean_difference']:.4f}</td>
                    <td>[{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]</td>
                    <td class="{sig_class}">{significance}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            <p><em>注：* p < 0.05表示差异显著</em></p>
        </div>
        
        <div class="section">
            <h2>5. 均衡性分析结果</h2>
            <p>重点分析AI对发言均衡性的影响：</p>
            <table class="result-table">
                <tr>
                    <th>分析项目</th>
                    <th>结果</th>
                </tr>
                <tr>
                    <td>Gini系数差值均值</td>
                    <td>{equilibrium_results['gini_difference_mean']:.4f}</td>
                </tr>
                <tr>
                    <td>Gini系数差值标准差</td>
                    <td>{equilibrium_results['gini_difference_std']:.4f}</td>
                </tr>
                <tr>
                    <td>t统计量</td>
                    <td>{equilibrium_results['t_statistic']:.4f}</td>
                </tr>
                <tr>
                    <td>p值</td>
                    <td>{equilibrium_results['p_value']:.4f}</td>
                </tr>
                <tr>
                    <td>差异显著性</td>
                    <td class="{'significant' if equilibrium_results['significant'] else 'not-significant'}">
                        {'显著' if equilibrium_results['significant'] else '不显著'}
                    </td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>6. 可视化图表</h2>
            <p>以下是分析过程中生成的可视化图表：</p>
            
            <h3>6.1 配对线图</h3>
            <img src="paired_lines_speech_metrics.png" alt="配对线图" style="width:100%; max-width:800px; height:auto;">
            
            <h3>6.2 箱线图</h3>
            <img src="boxplots_speech_metrics.png" alt="箱线图" style="width:100%; max-width:800px; height:auto;">
            
            <h3>6.3 正态性检验图</h3>
            <h4>发言总时长差值正态性检验</h4>
            <img src="normality_diff_total.png" alt="总时长差值正态性检验" style="width:100%; max-width:800px; height:auto;">
            
            <h4>发言次数差值正态性检验</h4>
            <img src="normality_diff_count.png" alt="次数差值正态性检验" style="width:100%; max-width:800px; height:auto;">
            
            <h4>发言均衡性差值正态性检验</h4>
            <img src="normality_diff_gini.png" alt="Gini系数差值正态性检验" style="width:100%; max-width:800px; height:auto;">
            
            <h3>6.4 效应量森林图</h3>
            <img src="effect_size_forest_plot.png" alt="效应量森林图" style="width:100%; max-width:800px; height:auto;">
            
            <h3>6.5 均衡性分析图</h3>
            <img src="gini_equilibrium_analysis.png" alt="均衡性分析图" style="width:100%; max-width:800px; height:auto;">
        </div>
        
        <div class="section">
            <h2>7. 结论与建议</h2>
            <p>基于以上分析结果，可以得出以下结论：</p>
            <ul>
                <li>发言总时长：{'AI组显著高于NoAI组' if ttest_results['speak_total_s']['significant'] else '两组间无显著差异'}</li>
                <li>发言次数：{'AI组显著高于NoAI组' if ttest_results['speak_count']['significant'] else '两组间无显著差异'}</li>
                <li>发言均衡性：{'AI组显著影响发言均衡性' if ttest_results['gini_speak']['significant'] else 'AI对发言均衡性无显著影响'}</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>8. 数据文件说明</h2>
            <p>本分析生成了以下数据文件：</p>
            <ul>
                <li><strong>paired_speech_data.csv：</strong> 完整的配对数据</li>
                <li><strong>speech_differences.csv：</strong> 差值数据</li>
                <li><strong>descriptive_statistics.csv：</strong> 描述性统计</li>
                <li><strong>normality_test_results.csv：</strong> 正态性检验结果</li>
                <li><strong>paired_ttest_results.csv：</strong> 配对t检验结果</li>
                <li><strong>equilibrium_analysis_results.csv：</strong> 均衡性分析结果</li>
            </ul>
        </div>
        
        <hr>
        <p><em>报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </body>
    </html>
    """
    
    with open('speech/speech_paired_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("HTML报告已生成完成！")


def main():
    """主函数"""
    print("=" * 60)
    print("发言指标配对分析开始")
    print("=" * 60)
    
    print("1. 正在读取数据...")
    df = pd.read_csv('speech/speech_metrics_results.csv')
    
    print("2. 正在创建配对数据...")
    paired_df, diff_df = create_paired_data(df)
    
    print("3. 正在计算描述性统计...")
    desc_stats = descriptive_statistics(paired_df)
    
    print("4. 正在进行正态性检验...")
    normality_results = normality_analysis(diff_df)
    
    print("5. 正在执行配对t检验...")
    ttest_results = paired_ttest_analysis(paired_df)
    
    print("6. 正在分析发言均衡性...")
    equilibrium_results = equilibrium_analysis(paired_df)
    
    print("7. 正在生成可视化图表...")
    create_visualizations(paired_df, diff_df)
    
    print("8. 正在保存CSV结果...")
    save_results_to_csv(paired_df, diff_df, desc_stats, normality_results, ttest_results, equilibrium_results)
    
    print("9. 正在生成HTML报告...")
    generate_html_report(paired_df, diff_df, desc_stats, normality_results, ttest_results, equilibrium_results)
    
    print("\n" + "=" * 60)
    print("分析完成！所有结果已保存")
    print("=" * 60)
    print(f"生成的CSV文件：")
    print("- paired_speech_data.csv (配对数据)")
    print("- speech_differences.csv (差值数据)")
    print("- descriptive_statistics.csv (描述性统计)")
    print("- normality_test_results.csv (正态性检验)")
    print("- paired_ttest_results.csv (配对t检验)")
    print("- equilibrium_analysis_results.csv (均衡性分析)")
    print(f"\n生成的HTML报告：speech_paired_analysis_report.html")
    print(f"生成的可视化图表：6个PNG文件")


if __name__ == "__main__":
    main()
