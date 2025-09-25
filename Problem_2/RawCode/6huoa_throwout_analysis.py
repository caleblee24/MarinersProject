#!/usr/bin/env python3
"""
Analysis of 6Huoa throwouts from outfield on groundouts vs other hit types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_analyze_data():
    """Load and analyze 6Huoa throwout data by hit type"""
    
    print("=== 6HUA THROWOUT ANALYSIS ===")
    print("Analyzing throwouts from outfield for 6Huoa on groundouts vs other hit types\n")
    
    # Load the context table data
    context_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Problem 2/6HuoaRi0_context_table.csv')
    
    # Load training data to get hit trajectory information
    train_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Data/train_data.csv')
    
    # Load outfield position data
    outfield_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Data/outfield_position.csv')
    
    print("Data loaded successfully!")
    print(f"Context table shape: {context_df.shape}")
    print(f"Training data shape: {train_df.shape}")
    print(f"Outfield position data shape: {outfield_df.shape}")
    
    # Analyze the context data
    print("\n=== CONTEXT TABLE ANALYSIS ===")
    print("Context table columns:", context_df.columns.tolist())
    print("\nFirst few rows:")
    print(context_df.head())
    
    # Check if we have hit trajectory data in training data
    if 'hit_trajectory' in train_df.columns:
        print("\n=== HIT TRAJECTORY ANALYSIS ===")
        hit_types = train_df['hit_trajectory'].value_counts()
        print("Hit trajectory distribution:")
        print(hit_types)
        
        # Focus on groundouts vs other types
        groundouts = train_df[train_df['hit_trajectory'] == 'ground_ball']
        other_hits = train_df[train_df['hit_trajectory'] != 'ground_ball']
        
        print(f"\nGroundouts: {len(groundouts)} plays")
        print(f"Other hit types: {len(other_hits)} plays")
        
        # Analyze runner advancement by hit type
        print("\n=== RUNNER ADVANCEMENT BY HIT TYPE ===")
        groundout_advance_rate = groundouts['runner_advance'].mean()
        other_hit_advance_rate = other_hits['runner_advance'].mean()
        
        print(f"Groundout advancement rate: {groundout_advance_rate:.3f}")
        print(f"Other hit advancement rate: {other_hit_advance_rate:.3f}")
        print(f"Difference: {other_hit_advance_rate - groundout_advance_rate:.3f}")
        
        # Statistical test
        chi2, p_value = stats.chi2_contingency([
            [groundouts['runner_advance'].sum(), len(groundouts) - groundouts['runner_advance'].sum()],
            [other_hits['runner_advance'].sum(), len(other_hits) - other_hits['runner_advance'].sum()]
        ])[:2]
        
        print(f"\nChi-square test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Significant difference in advancement rates between hit types")
        else:
            print("No significant difference in advancement rates between hit types")
    
    # Analyze the context table for 6Huoa specifically
    print("\n=== 6HUA CONTEXT ANALYSIS ===")
    
    # Look for patterns in the context data
    print("Context categories:")
    print(context_df['context'].value_counts())
    
    # Analyze player prevention rates by context
    print("\nPlayer prevention rates by context:")
    context_analysis = context_df.groupby('context').agg({
        'player_prevent': 'mean',
        'league_prevent': 'mean', 
        'diff': 'mean',
        'n': 'sum'
    }).round(3)
    
    print(context_analysis)
    
    # Look for groundout-specific contexts
    groundout_contexts = context_df[context_df['context'].str.contains('ground', case=False, na=False)]
    if len(groundout_contexts) > 0:
        print("\n=== GROUNDOUT-SPECIFIC CONTEXTS ===")
        print(groundout_contexts)
    else:
        print("\nNo specific groundout contexts found in the data")
    
    # Analyze by base position and outs
    print("\n=== ANALYSIS BY BASE POSITION AND OUTS ===")
    base_analysis = context_df.groupby(['context']).agg({
        'player_prevent': 'mean',
        'league_prevent': 'mean',
        'diff': 'mean',
        'n': 'sum'
    }).round(3)
    
    print("Performance by context:")
    print(base_analysis.sort_values('diff', ascending=False))
    
    return context_df, train_df, outfield_df

def create_visualizations(context_df, train_df):
    """Create visualizations for the analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('6Huoa Throwout Analysis: Groundouts vs Other Hit Types', fontsize=16, fontweight='bold')
    
    # 1. Context performance comparison
    ax1 = axes[0, 0]
    context_performance = context_df.groupby('context').agg({
        'player_prevent': 'mean',
        'league_prevent': 'mean',
        'diff': 'mean',
        'n': 'sum'
    }).sort_values('diff', ascending=True)
    
    x_pos = np.arange(len(context_performance))
    ax1.barh(x_pos, context_performance['player_prevent'], alpha=0.7, label='6Huoa Performance', color='steelblue')
    ax1.barh(x_pos, context_performance['league_prevent'], alpha=0.7, label='League Average', color='lightcoral')
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(context_performance.index, fontsize=8)
    ax1.set_xlabel('Prevention Rate')
    ax1.set_title('6Huoa vs League Performance by Context')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sample size by context
    ax2 = axes[0, 1]
    context_counts = context_df['context'].value_counts()
    ax2.bar(range(len(context_counts)), context_counts.values, color='lightgreen', alpha=0.7)
    ax2.set_xticks(range(len(context_counts)))
    ax2.set_xticklabels(context_counts.index, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Number of Plays')
    ax2.set_title('Sample Size by Context')
    ax2.grid(True, alpha=0.3)
    
    # 3. Hit trajectory analysis (if available)
    ax3 = axes[1, 0]
    if 'hit_trajectory' in train_df.columns:
        hit_trajectory_counts = train_df['hit_trajectory'].value_counts()
        ax3.pie(hit_trajectory_counts.values, labels=hit_trajectory_counts.index, autopct='%1.1f%%')
        ax3.set_title('Distribution of Hit Types')
    else:
        ax3.text(0.5, 0.5, 'Hit trajectory data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Hit Trajectory Analysis')
    
    # 4. Performance difference by context
    ax4 = axes[1, 1]
    context_df_sorted = context_df.sort_values('diff', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in context_df_sorted['diff']]
    ax4.barh(range(len(context_df_sorted)), context_df_sorted['diff'], color=colors, alpha=0.7)
    ax4.set_yticks(range(len(context_df_sorted)))
    ax4.set_yticklabels(context_df_sorted['context'], fontsize=8)
    ax4.set_xlabel('Performance Difference (6Huoa - League)')
    ax4.set_title('6Huoa Performance vs League by Context')
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main analysis function"""
    
    # Load and analyze data
    context_df, train_df, outfield_df = load_and_analyze_data()
    
    # Create visualizations
    fig = create_visualizations(context_df, train_df)
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total contexts analyzed: {len(context_df)}")
    print(f"Average 6Huoa prevention rate: {context_df['player_prevent'].mean():.3f}")
    print(f"Average league prevention rate: {context_df['league_prevent'].mean():.3f}")
    print(f"Average performance difference: {context_df['diff'].mean():.3f}")
    
    # Best and worst performing contexts
    best_context = context_df.loc[context_df['diff'].idxmax()]
    worst_context = context_df.loc[context_df['diff'].idxmin()]
    
    print(f"\nBest performing context: {best_context['context']} (diff: {best_context['diff']:.3f})")
    print(f"Worst performing context: {worst_context['context']} (diff: {worst_context['diff']:.3f})")
    
    # Check for groundout-specific patterns
    print("\n=== GROUNDOUT ANALYSIS ===")
    if 'ground' in context_df['context'].str.lower().str.cat(sep=' '):
        print("Groundout-related contexts found in the data")
        groundout_contexts = context_df[context_df['context'].str.contains('ground', case=False, na=False)]
        if len(groundout_contexts) > 0:
            print("Groundout-specific contexts:")
            print(groundout_contexts[['context', 'player_prevent', 'league_prevent', 'diff', 'n']])
    else:
        print("No specific groundout contexts found in the context table")
        print("The context table appears to focus on base/out situations rather than hit types")

if __name__ == "__main__":
    main()
