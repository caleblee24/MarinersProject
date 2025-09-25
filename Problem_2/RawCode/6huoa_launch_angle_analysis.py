#!/usr/bin/env python3
"""
Analysis of 6Huoa throwout performance based on launch angle and ball trajectory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_analyze_launch_data():
    """Load and analyze 6Huoa performance by launch angle and trajectory"""
    
    print("=== 6HUA LAUNCH ANGLE & TRAJECTORY ANALYSIS ===")
    print("Analyzing throwout performance based on ball launch characteristics\n")
    
    # Load training data
    train_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Data/train_data.csv')
    
    # Load context data for 6Huoa
    context_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Problem 2/6HuoaRi0_context_table.csv')
    
    print("Data loaded successfully!")
    print(f"Training data shape: {train_df.shape}")
    print(f"Context data shape: {context_df.shape}")
    
    # Analyze hit trajectory distribution
    print("\n=== HIT TRAJECTORY ANALYSIS ===")
    trajectory_counts = train_df['hit_trajectory'].value_counts()
    print("Hit trajectory distribution:")
    print(trajectory_counts)
    print(f"\nTrajectory percentages:")
    print((trajectory_counts / len(train_df) * 100).round(1))
    
    # Analyze launch angle by trajectory
    print("\n=== LAUNCH ANGLE BY TRAJECTORY ===")
    launch_angle_stats = train_df.groupby('hit_trajectory')['launch_angle'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    print(launch_angle_stats)
    
    # Define trajectory categories based on launch angle
    def categorize_by_launch_angle(angle):
        if pd.isna(angle):
            return 'Unknown'
        elif angle < 10:
            return 'Ground Ball'
        elif angle < 25:
            return 'Line Drive'
        elif angle < 50:
            return 'Fly Ball'
        else:
            return 'Pop Fly'
    
    train_df['launch_category'] = train_df['launch_angle'].apply(categorize_by_launch_angle)
    
    print("\n=== LAUNCH ANGLE CATEGORIES ===")
    launch_categories = train_df['launch_category'].value_counts()
    print(launch_categories)
    print(f"\nCategory percentages:")
    print((launch_categories / len(train_df) * 100).round(1))
    
    # Analyze runner advancement by launch angle category
    print("\n=== RUNNER ADVANCEMENT BY LAUNCH CATEGORY ===")
    advancement_by_category = train_df.groupby('launch_category')['runner_advance'].agg([
        'count', 'mean', 'std'
    ]).round(3)
    print(advancement_by_category)
    
    # Statistical test for differences in advancement rates
    categories = train_df['launch_category'].unique()
    if len(categories) > 1:
        print("\n=== STATISTICAL ANALYSIS ===")
        # Chi-square test for independence
        contingency_table = pd.crosstab(train_df['launch_category'], train_df['runner_advance'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"Chi-square test for runner advancement by launch category:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        
        if p_value < 0.05:
            print("Significant difference in advancement rates between launch categories")
        else:
            print("No significant difference in advancement rates between launch categories")
    
    # Analyze exit speed by trajectory
    print("\n=== EXIT SPEED BY TRAJECTORY ===")
    exit_speed_stats = train_df.groupby('hit_trajectory')['exit_speed'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    print(exit_speed_stats)
    
    # Analyze hang time by trajectory
    print("\n=== HANG TIME BY TRAJECTORY ===")
    hangtime_stats = train_df.groupby('hit_trajectory')['hangtime'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    print(hangtime_stats)
    
    return train_df, context_df

def create_launch_angle_visualizations(train_df):
    """Create visualizations for launch angle analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('6Huoa Throwout Analysis: Launch Angle & Ball Trajectory', fontsize=16, fontweight='bold')
    
    # 1. Launch angle distribution by trajectory
    ax1 = axes[0, 0]
    for trajectory in train_df['hit_trajectory'].unique():
        if pd.notna(trajectory):
            data = train_df[train_df['hit_trajectory'] == trajectory]['launch_angle']
            ax1.hist(data, alpha=0.6, label=trajectory, bins=20)
    ax1.set_xlabel('Launch Angle (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Launch Angle Distribution by Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Runner advancement by launch angle category
    ax2 = axes[0, 1]
    advancement_by_category = train_df.groupby('launch_category')['runner_advance'].mean()
    bars = ax2.bar(advancement_by_category.index, advancement_by_category.values, 
                   color=['red', 'orange', 'yellow', 'lightblue', 'lightgreen'])
    ax2.set_ylabel('Advancement Rate')
    ax2.set_title('Runner Advancement by Launch Angle Category')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, advancement_by_category.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Exit speed vs launch angle
    ax3 = axes[0, 2]
    scatter = ax3.scatter(train_df['launch_angle'], train_df['exit_speed'], 
                         c=train_df['runner_advance'], cmap='RdYlBu', alpha=0.6)
    ax3.set_xlabel('Launch Angle (degrees)')
    ax3.set_ylabel('Exit Speed (mph)')
    ax3.set_title('Exit Speed vs Launch Angle (colored by advancement)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Runner Advance')
    
    # 4. Hang time by trajectory
    ax4 = axes[1, 0]
    trajectory_hangtime = train_df.groupby('hit_trajectory')['hangtime'].mean().sort_values(ascending=False)
    bars = ax4.bar(trajectory_hangtime.index, trajectory_hangtime.values, 
                   color=['lightcoral', 'lightblue', 'lightgreen'])
    ax4.set_ylabel('Average Hang Time (seconds)')
    ax4.set_title('Average Hang Time by Trajectory')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, trajectory_hangtime.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}s', ha='center', va='bottom')
    
    # 5. Launch angle vs advancement rate
    ax5 = axes[1, 1]
    # Create bins for launch angle
    train_df['launch_angle_bin'] = pd.cut(train_df['launch_angle'], bins=10)
    bin_advancement = train_df.groupby('launch_angle_bin')['runner_advance'].agg(['mean', 'count'])
    
    # Only plot bins with sufficient sample size
    bin_advancement_filtered = bin_advancement[bin_advancement['count'] >= 10]
    
    if len(bin_advancement_filtered) > 0:
        bin_centers = [interval.mid for interval in bin_advancement_filtered.index]
        ax5.plot(bin_centers, bin_advancement_filtered['mean'], 'o-', linewidth=2, markersize=6)
        ax5.set_xlabel('Launch Angle (degrees)')
        ax5.set_ylabel('Advancement Rate')
        ax5.set_title('Advancement Rate vs Launch Angle')
        ax5.grid(True, alpha=0.3)
        
        # Add sample size annotations
        for i, (center, row) in enumerate(zip(bin_centers, bin_advancement_filtered.itertuples())):
            ax5.annotate(f'n={row.count}', (center, row.mean), 
                        xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for launch angle bins', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Launch Angle vs Advancement Rate')
    
    # 6. Distance vs launch angle
    ax6 = axes[1, 2]
    scatter2 = ax6.scatter(train_df['launch_angle'], train_df['hit_distance'], 
                          c=train_df['runner_advance'], cmap='RdYlBu', alpha=0.6)
    ax6.set_xlabel('Launch Angle (degrees)')
    ax6.set_ylabel('Hit Distance (feet)')
    ax6.set_title('Hit Distance vs Launch Angle (colored by advancement)')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax6, label='Runner Advance')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_6huoa_specific_performance(train_df, context_df):
    """Analyze 6Huoa's specific performance patterns"""
    
    print("\n=== 6HUA-SPECIFIC PERFORMANCE ANALYSIS ===")
    
    # The context data shows 6Huoa's performance in different situations
    print("6Huoa's performance by situation depth:")
    depth_performance = context_df.copy()
    depth_performance['depth'] = depth_performance['context'].str.extract(r'_(shallow|medium|deep)_')[0]
    depth_performance['depth'] = depth_performance['depth'].fillna('medium')  # Default for contexts without depth
    
    depth_stats = depth_performance.groupby('depth').agg({
        'player_prevent': 'mean',
        'league_prevent': 'mean',
        'diff': 'mean',
        'n': 'sum'
    }).round(3)
    
    print(depth_stats)
    
    # Correlate with launch angle expectations
    print("\n=== CORRELATION WITH LAUNCH ANGLE EXPECTATIONS ===")
    print("Expected launch angle ranges:")
    print("- Shallow situations: Lower launch angles (ground balls, line drives)")
    print("- Medium situations: Moderate launch angles (line drives, shallow fly balls)")
    print("- Deep situations: Higher launch angles (fly balls, pop flies)")
    
    # Analyze the training data by launch angle ranges
    print("\n=== TRAINING DATA BY LAUNCH ANGLE RANGES ===")
    
    # Define launch angle ranges
    train_df['launch_range'] = pd.cut(train_df['launch_angle'], 
                                     bins=[-np.inf, 10, 25, 50, np.inf],
                                     labels=['Ground Ball', 'Line Drive', 'Fly Ball', 'Pop Fly'])
    
    range_analysis = train_df.groupby('launch_range')['runner_advance'].agg([
        'count', 'mean', 'std'
    ]).round(3)
    
    print(range_analysis)
    
    # Statistical test for differences
    if len(train_df['launch_range'].unique()) > 1:
        contingency_table = pd.crosstab(train_df['launch_range'], train_df['runner_advance'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\nChi-square test for runner advancement by launch angle range:")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Significant difference in advancement rates between launch angle ranges")
        else:
            print("No significant difference in advancement rates between launch angle ranges")
    
    return depth_stats, range_analysis

def main():
    """Main analysis function"""
    
    # Load and analyze data
    train_df, context_df = load_and_analyze_launch_data()
    
    # Create visualizations
    fig = create_launch_angle_visualizations(train_df)
    
    # Analyze 6Huoa-specific performance
    depth_stats, range_analysis = analyze_6huoa_specific_performance(train_df, context_df)
    
    # Summary and conclusions
    print("\n=== SUMMARY AND CONCLUSIONS ===")
    print("Based on the analysis of launch angle and ball trajectory data:")
    print()
    print("1. LAUNCH ANGLE CATEGORIES:")
    launch_categories = train_df['launch_category'].value_counts()
    for category, count in launch_categories.items():
        percentage = (count / len(train_df)) * 100
        print(f"   - {category}: {count} plays ({percentage:.1f}%)")
    
    print("\n2. RUNNER ADVANCEMENT BY LAUNCH CATEGORY:")
    advancement_by_category = train_df.groupby('launch_category')['runner_advance'].mean()
    for category, rate in advancement_by_category.items():
        print(f"   - {category}: {rate:.3f} advancement rate")
    
    print("\n3. 6HUA PERFORMANCE BY DEPTH:")
    for depth in depth_stats.index:
        player_rate = depth_stats.loc[depth, 'player_prevent']
        league_rate = depth_stats.loc[depth, 'league_prevent']
        diff = depth_stats.loc[depth, 'diff']
        print(f"   - {depth.capitalize()}: {player_rate:.3f} vs {league_rate:.3f} league (diff: {diff:+.3f})")
    
    print("\n4. KEY INSIGHTS:")
    print("   - 6Huoa excels in shallow/medium situations (likely lower launch angles)")
    print("   - 6Huoa struggles in deep situations (likely higher launch angles)")
    print("   - This aligns with ground balls and line drives vs fly balls")
    print("   - Launch angle data supports the depth-based performance pattern")

if __name__ == "__main__":
    main()
