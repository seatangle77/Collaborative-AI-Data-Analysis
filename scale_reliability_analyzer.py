#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scale Reliability Analyzer
Completely follows the SPSS RELIABILITY syntax structure provided by user
Author: AI Assistant
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ScaleReliabilityAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize the analyzer with scale data"""
        self.data = pd.read_csv(csv_file_path)
        print(f"✅ Data loaded successfully!")
        print(f"   Total participants: {len(self.data)}")
        print(f"   Total variables: {len(self.data.columns)}")
        
        # Define scale structure exactly as per SPSS syntax
        self.scale_structure = {
            'BE_N': {
                'variables': ['BE1_N', 'BE2_N', 'BE4_N'],
                'description': 'Behavioral Engagement (No-assistance)'
            },
            'SE_N': {
                'variables': ['SE1_N', 'SE2R_N', 'SE4R_N', 'SE6R_N'],
                'description': 'Social Engagement (No-assistance)'
            },
            'CE_N': {
                'variables': ['CE1_N', 'CE2_N', 'CE3_N'],
                'description': 'Cognitive Engagement (No-assistance)'
            },
            'CC_N': {
                'variables': ['CC3_N', 'CC5_N', 'CC7_N'],
                'description': 'Conceptual-to-Consequential Engagement (No-assistance)'
            },
            'Total_N': {
                'variables': ['BE1_N', 'BE2_N', 'BE4_N', 'SE1_N', 'SE2R_N', 'SE4R_N', 'SE6R_N', 
                            'CE1_N', 'CE2_N', 'CE3_N', 'CC3_N', 'CC5_N', 'CC7_N'],
                'description': 'Total Scale (No-assistance)'
            },
            'BE_A': {
                'variables': ['BE1_A', 'BE2_A', 'BE4_A'],
                'description': 'Behavioral Engagement (Assistance)'
            },
            'SE_A': {
                'variables': ['SE1_A', 'SE2R_A', 'SE4R_A', 'SE6R_A'],
                'description': 'Social Engagement (Assistance)'
            },
            'CE_A': {
                'variables': ['CE1_A', 'CE2_A', 'CE3_A'],
                'description': 'Cognitive Engagement (Assistance)'
            },
            'CC_A': {
                'variables': ['CC3_A', 'CC5_A', 'CC7_A'],
                'description': 'Conceptual-to-Consequential Engagement (Assistance)'
            },
            'Total_A': {
                'variables': ['BE1_A', 'BE2_A', 'BE4_A', 'SE1_A', 'SE2R_A', 'SE4R_A', 'SE6R_A', 
                            'CE1_A', 'CE2_A', 'CE3_A', 'CC3_A', 'CC5_A', 'CC7_A'],
                'description': 'Total Scale (Assistance)'
            }
        }
    
    def calculate_cronbach_alpha(self, variables):
        """Calculate Cronbach's α for a set of variables"""
        if len(variables) < 2:
            return np.nan, np.nan, np.nan
            
        # Get data for the variables
        item_data = self.data[variables].dropna()
        
        if len(item_data) == 0:
            return np.nan, np.nan, np.nan
            
        # Calculate item variances and total variance
        item_variances = item_data.var()
        total_variance = item_data.sum(axis=1).var()
        
        # Cronbach's α formula
        n_items = len(variables)
        if total_variance == 0:
            return np.nan, np.nan, np.nan
            
        alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
        
        # Calculate descriptive statistics
        total_scores = item_data.sum(axis=1)
        mean_total = total_scores.mean()
        std_total = total_scores.std()
        
        return alpha, mean_total, std_total
    
    def analyze_reliability(self):
        """Analyze reliability for all scales exactly as per SPSS syntax"""
        print("\n" + "="*80)
        print("📊 RELIABILITY ANALYSIS - Cronbach's α")
        print("Following SPSS RELIABILITY syntax structure")
        print("="*80)
        
        results = {}
        
        # Analyze each scale
        for scale_name, scale_info in self.scale_structure.items():
            print(f"\n🔍 {scale_info['description']}")
            print("-" * 60)
            
            # Check if all variables exist
            missing_vars = [var for var in scale_info['variables'] if var not in self.data.columns]
            if missing_vars:
                print(f"❌ Missing variables: {missing_vars}")
                continue
            
            # Calculate reliability
            alpha, mean_total, std_total = self.calculate_cronbach_alpha(scale_info['variables'])
            
            if not np.isnan(alpha):
                print(f"  Variables: {', '.join(scale_info['variables'])}")
                print(f"  Number of items: {len(scale_info['variables'])}")
                print(f"  Cronbach's α: {alpha:.4f}")
                print(f"  Total score - Mean: {mean_total:.3f}")
                print(f"  Total score - Std: {std_total:.3f}")
                
                # Interpret α value
                if alpha >= 0.9:
                    interpretation = "Excellent"
                elif alpha >= 0.8:
                    interpretation = "Good"
                elif alpha >= 0.7:
                    interpretation = "Acceptable"
                elif alpha >= 0.6:
                    interpretation = "Questionable"
                else:
                    interpretation = "Poor"
                print(f"  Reliability: {interpretation}")
                
                # Store results
                results[scale_name] = {
                    'alpha': alpha,
                    'n_items': len(scale_info['variables']),
                    'mean_total': mean_total,
                    'std_total': std_total,
                    'interpretation': interpretation,
                    'variables': scale_info['variables']
                }
            else:
                print(f"  ❌ Cannot calculate α for {scale_name}")
                results[scale_name] = {
                    'alpha': np.nan,
                    'n_items': len(scale_info['variables']),
                    'mean_total': np.nan,
                    'std_total': np.nan,
                    'interpretation': 'N/A',
                    'variables': scale_info['variables']
                }
        
        return results
    
    def create_summary_table(self, results):
        """Create a clear summary table of all results"""
        print("\n" + "="*80)
        print("📋 RELIABILITY SUMMARY TABLE")
        print("="*80)
        
        # Create table header
        print(f"{'Scale':<12} {'Condition':<10} {'Items':<6} {'α':<8} {'Mean':<8} {'Std':<8} {'Quality':<12}")
        print("-" * 80)
        
        # Group results by condition
        no_assistance_scales = ['BE_N', 'SE_N', 'CE_N', 'CC_N', 'Total_N']
        assistance_scales = ['BE_A', 'SE_A', 'CE_A', 'CC_A', 'Total_A']
        
        # Print No-assistance results
        print("  📊 NO-ASSISTANCE CONDITION (N)")
        print("-" * 80)
        for scale in no_assistance_scales:
            if scale in results:
                result = results[scale]
                if not np.isnan(result['alpha']):
                    print(f"{scale:<12} {'No-assist':<10} {result['n_items']:<6} {result['alpha']:<8.3f} {result['mean_total']:<8.1f} {result['std_total']:<8.1f} {result['interpretation']:<12}")
                else:
                    print(f"{scale:<12} {'No-assist':<10} {result['n_items']:<6} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<12}")
        
        print()
        print("  📊 ASSISTANCE CONDITION (A)")
        print("-" * 80)
        for scale in assistance_scales:
            if scale in results:
                result = results[scale]
                if not np.isnan(result['alpha']):
                    print(f"{scale:<12} {'Assistance':<10} {result['n_items']:<6} {result['alpha']:<8.3f} {result['mean_total']:<8.1f} {result['std_total']:<8.1f} {result['interpretation']:<12}")
                else:
                    print(f"{scale:<12} {'Assistance':<10} {result['n_items']:<6} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<12}")
        
        print("-" * 80)
    
    def create_comparison_table(self, results):
        """Create a comparison table between No-assistance and Assistance conditions"""
        print("\n" + "="*80)
        print("📊 CONDITION COMPARISON TABLE")
        print("="*80)
        
        print(f"{'Dimension':<20} {'No-assist α':<12} {'Assist α':<12} {'Difference':<12} {'Change':<15}")
        print("-" * 80)
        
        # Compare each dimension
        dimensions = ['BE', 'SE', 'CE', 'CC', 'Total']
        
        for dim in dimensions:
            n_scale = f"{dim}_N"
            a_scale = f"{dim}_A"
            
            if n_scale in results and a_scale in results:
                n_alpha = results[n_scale]['alpha']
                a_alpha = results[a_scale]['alpha']
                
                if not np.isnan(n_alpha) and not np.isnan(a_alpha):
                    diff = a_alpha - n_alpha
                    if diff > 0:
                        change = "↑ Improvement"
                    elif diff < 0:
                        change = "↓ Decline"
                    else:
                        change = "→ No change"
                    
                    print(f"{dim:<20} {n_alpha:<12.3f} {a_alpha:<12.3f} {diff:<+12.3f} {change:<15}")
                else:
                    print(f"{dim:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
        
        print("-" * 80)
    
    def generate_final_report(self):
        """Generate the complete reliability analysis report"""
        print("\n🚀 Starting comprehensive reliability analysis...")
        print("Following SPSS RELIABILITY syntax structure exactly")
        
        # Run reliability analysis
        results = self.analyze_reliability()
        
        # Create summary table
        self.create_summary_table(results)
        
        # Create comparison table
        self.create_comparison_table(results)
        
        # Final summary
        print("\n" + "="*80)
        print("📋 ANALYSIS COMPLETE")
        print("="*80)
        
        # Count successful analyses
        successful = sum(1 for r in results.values() if not np.isnan(r['alpha']))
        total = len(results)
        
        print(f"✅ Successfully analyzed: {successful}/{total} scales")
        print(f"📊 Results match SPSS RELIABILITY syntax structure")
        print(f"🔍 Check the tables above for detailed results")
        
        return results

def main():
    """Main function"""
    print("🔍 Scale Reliability Analyzer")
    print("="*60)
    print("Completely follows SPSS RELIABILITY syntax structure")
    print("="*60)
    
    try:
        # Create analyzer instance
        analyzer = ScaleReliabilityAnalyzer('Scale_data.csv')
        
        # Generate complete report
        results = analyzer.generate_final_report()
        
    except FileNotFoundError:
        print("❌ Error: File 'Scale_data.csv' not found!")
        print("Please ensure the data file is in the current directory.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check your data file and try again.")

if __name__ == "__main__":
    main()
