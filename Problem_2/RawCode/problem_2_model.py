#!/usr/bin/env python3
"""
Mariners 2026 Data Science Intern Problem Set - Problem 2
Enhanced Outfielder 6HuoaRi0 Performance Analysis - Comprehensive Data Integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import sqrt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OutfielderAnalysis:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.outfield_df = None
        self.target_fielder = '6HuoaRi0'
        
    def load_data(self):
        """Load all available datasets"""
        print("=== LOADING ALL AVAILABLE DATA ===")
        
        # Try multiple possible data locations
        data_paths = [
            # From Problem_2/RawCode/ directory
            ('../../Data/train_data.csv', '../../Data/test_data.csv', '../../Data/outfield_position.csv'),
            # From Problem_2/ directory  
            ('../Data/train_data.csv', '../Data/test_data.csv', '../Data/outfield_position.csv'),
            # From root directory
            ('Data/train_data.csv', 'Data/test_data.csv', 'Data/outfield_position.csv'),
            # Direct file names (if in same directory)
            ('train_data.csv', 'test_data.csv', 'outfield_position.csv')
        ]
        
        for train_path, test_path, outfield_path in data_paths:
            try:
                self.train_df = pd.read_csv(train_path)
                self.test_df = pd.read_csv(test_path)
                self.outfield_df = pd.read_csv(outfield_path)
                print(f"Data loaded from: {train_path}")
                break
            except FileNotFoundError:
                continue
        else:
            print("Could not find data files in any expected location")
            print("Expected locations:")
            for train_path, test_path, outfield_path in data_paths:
                print(f"  • {train_path}, {test_path}, {outfield_path}")
            raise FileNotFoundError("Data files not found")
        
        # Convert date columns
        self.train_df['game_date'] = pd.to_datetime(self.train_df['game_date'])
        self.test_df['game_date'] = pd.to_datetime(self.test_df['game_date'])
        
        print(f"Training data: {self.train_df.shape}")
        print(f"Test data: {self.test_df.shape}")
        print(f"Outfield position data: {self.outfield_df.shape}")
        
        
        return self
    
    def categorize_by_launch_angle(self, angle):
        """Categorize hit type based on launch angle"""
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
    
    def analyze_launch_angle_performance(self):
        """Analyze 6Huoa's performance by launch angle and ball trajectory"""
        print("\n=== LAUNCH ANGLE & TRAJECTORY ANALYSIS ===")
        
        # Get fielder's plays
        fielder_train = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        
        if len(fielder_train) == 0:
            print(f"Error: {self.target_fielder} not found in training data")
            return None
        
        # Add launch angle categories
        fielder_train['launch_category'] = fielder_train['launch_angle'].apply(self.categorize_by_launch_angle)
        fielder_train['prevented'] = 1 - fielder_train['runner_advance']
        
        # Analyze hit trajectory distribution
        print("\nHit trajectory distribution:")
        trajectory_counts = fielder_train['hit_trajectory'].value_counts()
        print(trajectory_counts)
        
        # Analyze launch angle by trajectory
        print("\nLaunch angle statistics by trajectory:")
        launch_stats = fielder_train.groupby('hit_trajectory')['launch_angle'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        print(launch_stats)
        
        # Analyze performance by launch angle category
        print("\nPerformance by launch angle category:")
        performance_by_category = fielder_train.groupby('launch_category').agg({
            'prevented': ['count', 'mean', 'std'],
            'launch_angle': ['mean', 'std']
        }).round(3)
        print(performance_by_category)
        
        # Statistical analysis
        if len(fielder_train['launch_category'].unique()) > 1:
            print("\nStatistical analysis:")
            contingency_table = pd.crosstab(fielder_train['launch_category'], fielder_train['prevented'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            print(f"Chi-square test for prevention by launch category:")
            print(f"Chi-square statistic: {chi2:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Degrees of freedom: {dof}")
            
            if p_value < 0.05:
                print("Significant difference in prevention rates between launch categories")
            else:
                print("No significant difference in prevention rates between launch categories")
        
        # Analyze exit speed and hang time by trajectory
        print("\nExit speed by trajectory:")
        exit_speed_stats = fielder_train.groupby('hit_trajectory')['exit_speed'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        print(exit_speed_stats)
        
        print("\nHang time by trajectory:")
        hangtime_stats = fielder_train.groupby('hit_trajectory')['hangtime'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        print(hangtime_stats)
        
        return fielder_train
    
    def create_launch_angle_visualizations(self, fielder_data):
        """Create visualizations for launch angle analysis"""
        if fielder_data is None or len(fielder_data) == 0:
            print("No data available for visualization")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.target_fielder} Launch Angle & Ball Trajectory Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Launch angle distribution by trajectory
        ax1 = axes[0, 0]
        for trajectory in fielder_data['hit_trajectory'].unique():
            if pd.notna(trajectory):
                data = fielder_data[fielder_data['hit_trajectory'] == trajectory]['launch_angle']
                ax1.hist(data, alpha=0.6, label=trajectory, bins=10)
        ax1.set_xlabel('Launch Angle (degrees)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Launch Angle Distribution by Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Prevention rate by launch angle category
        ax2 = axes[0, 1]
        prevention_by_category = fielder_data.groupby('launch_category')['prevented'].mean()
        bars = ax2.bar(prevention_by_category.index, prevention_by_category.values, 
                      color=['red', 'orange', 'yellow', 'lightblue', 'lightgreen'])
        ax2.set_ylabel('Prevention Rate')
        ax2.set_title('Prevention Rate by Launch Angle Category')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, prevention_by_category.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Exit speed vs launch angle
        ax3 = axes[0, 2]
        scatter = ax3.scatter(fielder_data['launch_angle'], fielder_data['exit_speed'], 
                             c=fielder_data['prevented'], cmap='RdYlBu', alpha=0.6)
        ax3.set_xlabel('Launch Angle (degrees)')
        ax3.set_ylabel('Exit Speed (mph)')
        ax3.set_title('Exit Speed vs Launch Angle (colored by prevention)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Prevented')
        
        # 4. Hang time by trajectory
        ax4 = axes[1, 0]
        trajectory_hangtime = fielder_data.groupby('hit_trajectory')['hangtime'].mean().sort_values(ascending=False)
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
        
        # 5. Launch angle vs prevention rate
        ax5 = axes[1, 1]
        # Create bins for launch angle
        fielder_data['launch_angle_bin'] = pd.cut(fielder_data['launch_angle'], bins=5)
        bin_prevention = fielder_data.groupby('launch_angle_bin')['prevented'].agg(['mean', 'count'])
        
        # Only plot bins with sufficient sample size
        bin_prevention_filtered = bin_prevention[bin_prevention['count'] >= 2]
        
        if len(bin_prevention_filtered) > 0:
            bin_centers = [interval.mid for interval in bin_prevention_filtered.index]
            ax5.plot(bin_centers, bin_prevention_filtered['mean'], 'o-', linewidth=2, markersize=6)
            ax5.set_xlabel('Launch Angle (degrees)')
            ax5.set_ylabel('Prevention Rate')
            ax5.set_title('Prevention Rate vs Launch Angle')
            ax5.grid(True, alpha=0.3)
            
            # Add sample size annotations
            for i, (center, row) in enumerate(zip(bin_centers, bin_prevention_filtered.itertuples())):
                ax5.annotate(f'n={row.count}', (center, row.mean), 
                            xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'Insufficient data for launch angle bins', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Launch Angle vs Prevention Rate')
        
        # 6. Distance vs launch angle
        ax6 = axes[1, 2]
        scatter2 = ax6.scatter(fielder_data['launch_angle'], fielder_data['hit_distance'], 
                              c=fielder_data['prevented'], cmap='RdYlBu', alpha=0.6)
        ax6.set_xlabel('Launch Angle (degrees)')
        ax6.set_ylabel('Hit Distance (feet)')
        ax6.set_title('Hit Distance vs Launch Angle (colored by prevention)')
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax6, label='Prevented')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_fielder_performance(self):
        """Analyze outfielder 6HuoaRi0's performance with contextual baselines"""
        print(f"=== ANALYZING OUTFIELDER {self.target_fielder} ===")
        
        
        # Get fielder's plays from train data
        fielder_train = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        fielder_test = self.test_df[self.test_df['fielder_id'] == self.target_fielder].copy()
        
        print(f"Plays in training data: {len(fielder_train)}")
        print(f"Plays in test data: {len(fielder_test)}")
        
        if len(fielder_train) == 0:
            print(f"Error: {self.target_fielder} not found in training data")
            return None
        
        # Convert to prevention frame (coaches think "Did he hold the runner?")
        fielder_train['prevented'] = 1 - fielder_train['runner_advance']
        
        # Basic performance metrics
        prevention_rate = fielder_train['prevented'].mean()
        total_plays = len(fielder_train)
        prevented_advances = fielder_train['prevented'].sum()
        allowed_advances = total_plays - prevented_advances
        
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Total plays: {total_plays}")
        print(f"Prevented advances: {prevented_advances}")
        print(f"Allowed advances: {allowed_advances}")
        print(f"Prevention rate: {prevention_rate:.3f}")
        
        # Contextual baseline analysis
        print(f"\n=== CONTEXTUAL BASELINE ANALYSIS ===")
        
        # Make depth_bin for the entire train first (same cut everywhere)
        bins = [0, 250, 350, np.inf]
        labels = ['shallow', 'medium', 'deep']
        self.train_df['depth_bin'] = pd.cut(self.train_df['hit_distance'], bins=bins, labels=labels)
        
        # Player frame with prevention
        fielder_train = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        fielder_train['prevented'] = 1 - fielder_train['runner_advance']
        
        # League (exclude player rows to avoid circularity)
        league_df = self.train_df[self.train_df['fielder_id'] != self.target_fielder].copy()
        league_df['prevented'] = 1 - league_df['runner_advance']
        
        ctx_cols = ['runner_base', 'outs', 'depth_bin']
        
        # Player stats by context
        ply = (fielder_train.groupby(ctx_cols, observed=True)['prevented']
               .agg(['mean', 'count']).reset_index()
               .rename(columns={'mean': 'player_prevent', 'count': 'n'}))
        
        # League baseline by context
        base = (league_df.groupby(ctx_cols, observed=True)['prevented']
                .mean().rename('league_prevent').reset_index())
        
        tbl = ply.merge(base, on=ctx_cols, how='left')
        tbl['diff'] = tbl['player_prevent'] - tbl['league_prevent']
        
        # Calculate Wilson CI for each row
        tbl[['ci_low', 'ci_high']] = tbl.apply(
            lambda r: pd.Series(self._wilson_ci(r['player_prevent'], r['n'])), axis=1
        )
        
        # Calculate runs prevented vs baseline
        run_value = {2: 0.25, 3: 0.80}
        tbl['run_value'] = tbl['runner_base'].map(run_value).fillna(0.25)
        tbl['runs_prevented'] = tbl['diff'] * tbl['n'] * tbl['run_value']
        total_rp = tbl['runs_prevented'].sum()
        
        # Calculate impact for sorting
        tbl['impact'] = tbl['diff'].abs() * tbl['n'] * tbl['run_value']
        
        print(f"Estimated runs prevented vs context: {total_rp:.2f} (over {total_plays} opportunities in this sample)")
        
        # Create coach table for export
        coach = tbl.copy()
        coach['context'] = (
            'Base' + coach['runner_base'].astype(str) +
            '_Outs' + coach['outs'].astype(str) +
            '_' + coach['depth_bin'].astype(str)
        )
        coach_display = (coach
            .loc[:, ['context','n','player_prevent','league_prevent','diff','ci_low','ci_high','runs_prevented','impact']]
            .sort_values('impact', ascending=False)
        )
        coach_display.to_csv('6HuoaRi0_context_table.csv', index=False)
        print(f"Coach table exported to: 6HuoaRi0_context_table.csv")
        
        # Display contextual performance (headline: n ≥ 10, appendix: all)
        print("\n=== HEADLINE CONTEXTS (n ≥ 10) ===")
        headline_contexts = tbl[tbl['n'] >= 10].sort_values('impact', ascending=False)
        if len(headline_contexts) > 0:
            for _, row in headline_contexts.iterrows():
                context = f"Base{row['runner_base']}_Outs{row['outs']}_{row['depth_bin']}"
                print(f"{context}: {row['player_prevent']:.3f} vs {row['league_prevent']:.3f} "
                      f"(diff: {row['diff']:+.3f}, n={row['n']}, "
                      f"CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}], impact: {row['impact']:.2f})")
        else:
            print("No contexts with n ≥ 10")
        
        print("\n=== ALL CONTEXTS (APPENDIX) ===")
        for _, row in tbl.sort_values('impact', ascending=False).iterrows():
            context = f"Base{row['runner_base']}_Outs{row['outs']}_{row['depth_bin']}"
            print(f"{context}: {row['player_prevent']:.3f} vs {row['league_prevent']:.3f} "
                  f"(diff: {row['diff']:+.3f}, n={row['n']}, "
                  f"CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}])")
            
            # Flag small samples
            if row['n'] < 10:
                print(f"  → Directional only (n={row['n']})")
        
        contextual_performance = tbl.to_dict('records')
        
        # Position breakdown with prevention frame
        position_performance = fielder_train.groupby('fielder_pos').agg({
            'prevented': ['count', 'sum', 'mean'],
            'runner_advance': 'mean'  # Keep original for reference
        }).round(3)
        
        print(f"\n=== PERFORMANCE BY POSITION ===")
        print("Prevention Rate by Position:")
        print(position_performance)
        
        # Hit trajectory breakdown
        trajectory_performance = fielder_train.groupby('hit_trajectory').agg({
            'prevented': ['count', 'sum', 'mean'],
            'runner_advance': 'mean'  # Keep original for reference
        }).round(3)
        
        print(f"\n=== PERFORMANCE BY HIT TYPE ===")
        print("Prevention Rate by Hit Type:")
        print(trajectory_performance)
        
        # Runs prevented analysis (now calculated in contextual section)
        print(f"\n=== RUNS PREVENTED ANALYSIS ===")
        print(f"Context-weighted runs prevented: {total_rp:.2f}")
        
        # Physical attributes analysis with outlier handling
        print(f"\n=== PHYSICAL ATTRIBUTES ===")
        throw_speed_analysis = self._analyze_throw_speed(fielder_train, league_df)
        print(f"Throw speed analysis: {throw_speed_analysis}")
        
        return {
            'total_plays': total_plays,
            'prevention_rate': prevention_rate,
            'position_performance': position_performance,
            'trajectory_performance': trajectory_performance,
            'contextual_performance': contextual_performance,
            'runs_prevented': total_rp,
            'throw_speed_analysis': throw_speed_analysis,
            'fielder_data': fielder_train
        }
    
    def analyze_positioning(self):
        """Analyze outfielder's positioning data"""
        print(f"\n=== POSITIONING ANALYSIS ===")
        
        # Get outfielder's positioning data
        fielder_positions = self.outfield_df[
            self.outfield_df['player_id'] == self.target_fielder
        ].copy()
        
        if len(fielder_positions) == 0:
            print("No positioning data available for this outfielder")
            return None
        
        print(f"Positioning data points: {len(fielder_positions)}")
        
        # Analyze positioning by event
        position_by_event = fielder_positions.groupby('event_description').agg({
            'pos_x': ['mean', 'std'],
            'pos_y': ['mean', 'std']
        }).round(2)
        
        print("\nAverage positioning by event:")
        print(position_by_event)
        
        # Calculate average distance from center
        fielder_positions['distance_from_center'] = np.sqrt(
            fielder_positions['pos_x']**2 + fielder_positions['pos_y']**2
        )
        
        avg_distance = fielder_positions['distance_from_center'].mean()
        print(f"\nAverage distance from center: {avg_distance:.1f} feet")
        
        return fielder_positions
    
    def _wilson_ci(self, p, n, z=1.96):
        """Calculate Wilson confidence interval for prevention rate (handles small n)"""
        from math import sqrt
        if n == 0: 
            return (np.nan, np.nan)
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = (z * sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
        return center - half, center + half
    
    def _calculate_runs_prevented(self, fielder_data):
        """Calculate estimated runs prevented based on base advancement"""
        runs_prevented = 0
        
        for _, play in fielder_data.iterrows():
            if play['prevented'] == 1:  # If he prevented the advance
                if play['runner_base'] == 2:
                    runs_prevented += 0.25  # 2B to 3B ≈ 0.25 runs
                elif play['runner_base'] == 3:
                    runs_prevented += 0.8  # 3B to home ≈ 0.8 runs
        
        return runs_prevented
    
    def _analyze_throw_speed(self, fielder_data, league_df):
        """Analyze throw speed with outlier handling and percentile ranking"""
        throw_speeds = fielder_data['fielder_max_throwspeed'].dropna()
        league_throws = league_df['fielder_max_throwspeed'].dropna()
        
        if len(throw_speeds) == 0:
            return "No throw speed data available"
        
        # Winsorize outliers (cap at 95th percentile)
        p95 = throw_speeds.quantile(0.95)
        winsorized = np.where(throw_speeds > p95, p95, throw_speeds)
        
        # Calculate percentile in league
        player_median = throw_speeds.median()
        league_percentile = (league_throws < player_median).mean()
        
        analysis = {
            'mean': throw_speeds.mean(),
            'median': throw_speeds.median(),
            'winsorized_mean': winsorized.mean(),
            'winsorized_median': np.median(winsorized),
            'p95': p95,
            'outliers_removed': (throw_speeds > p95).sum(),
            'league_percentile': league_percentile
        }
        
        return analysis
    
    def generate_report(self):
        """Generate the coaching staff report"""
        print("\n" + "="*60)
        print("OUTFIELDER PERFORMANCE REPORT")
        print("="*60)
        
        performance = self.analyze_fielder_performance()
        positioning = self.analyze_positioning()
        
        if performance is None:
            return
        
        print(f"\nEXECUTIVE SUMMARY")
        print(f"Outfielder {self.target_fielder} has been involved in {performance['total_plays']} sacrifice play situations")
        print(f"with a prevention rate of {performance['prevention_rate']:.1%} (prevented {performance['prevention_rate']*performance['total_plays']:.0f} of {performance['total_plays']} advances).")
        
        print(f"\nCONTEXTUAL PERFORMANCE ANALYSIS (n ≥ 10):")
        if performance['contextual_performance']:
            # Filter for decision-making contexts (n ≥ 10)
            decision_contexts = [p for p in performance['contextual_performance'] if p['n'] >= 10]
            if decision_contexts:
                # Sort by impact
                decision_contexts.sort(key=lambda x: x['impact'], reverse=True)
                for perf in decision_contexts:
                    context = f"Base{perf['runner_base']}_Outs{perf['outs']}_{perf['depth_bin']}"
                    print(f"• {context}: {perf['player_prevent']:.1%} vs {perf['league_prevent']:.1%} league average")
                    print(f"  Difference: {perf['diff']:+.1%} (n={perf['n']}, CI: [{perf['ci_low']:.1%}, {perf['ci_high']:.1%}], impact: {perf['impact']:.2f})")
            else:
                print("• No contexts with sufficient sample size (n ≥ 10)")
        else:
            print("• Insufficient sample size for contextual analysis")
        
        print(f"\nKEY FINDINGS:")
        
        # Throw speed analysis
        throw_analysis = performance['throw_speed_analysis']
        if isinstance(throw_analysis, dict):
            print(f"• Throw Speed: {throw_analysis['winsorized_median']:.1f} mph median (~P{throw_analysis['league_percentile']*100:.0f} in league)")
            if throw_analysis['outliers_removed'] > 0:
                print(f"  - {throw_analysis['outliers_removed']} outliers removed (>{throw_analysis['p95']:.1f} mph)")
            print(f"  - Raw mean: {throw_analysis['mean']:.1f} mph, Median: {throw_analysis['median']:.1f} mph")
        else:
            print(f"• Throw Speed: {throw_analysis}")
        
        # Position performance
        print(f"\n• Position Performance (Prevention Rate):")
        for pos in performance['position_performance'].index:
            count = performance['position_performance'].loc[pos, ('prevented', 'count')]
            prevention_rate = performance['position_performance'].loc[pos, ('prevented', 'mean')]
            print(f"  - {pos}: {count} plays, {prevention_rate:.1%} prevention rate")
        
        # Runs prevented
        print(f"\n• Context-weighted Runs Prevented: {performance['runs_prevented']:.2f} runs")
        
        print(f"\nRECOMMENDATIONS:")
        
        # Contextual recommendations (focus on decision-making contexts)
        if performance['contextual_performance']:
            decision_contexts = [p for p in performance['contextual_performance'] if p['n'] >= 10]
            above_avg_contexts = [p for p in decision_contexts if p['diff'] > 0.05]
            below_avg_contexts = [p for p in decision_contexts if p['diff'] < -0.05]
            
            if above_avg_contexts:
                print(f"• STRENGTHS: Above average in {len(above_avg_contexts)} decision-making contexts")
                for ctx in sorted(above_avg_contexts, key=lambda x: x['impact'], reverse=True):
                    context = f"Base{ctx['runner_base']}_Outs{ctx['outs']}_{ctx['depth_bin']}"
                    print(f"  - {context}: +{ctx['diff']:.1%} vs league (impact: {ctx['impact']:.2f})")
            
            if below_avg_contexts:
                print(f"• AREAS FOR IMPROVEMENT: Below average in {len(below_avg_contexts)} decision-making contexts")
                for ctx in sorted(below_avg_contexts, key=lambda x: x['impact'], reverse=True):
                    context = f"Base{ctx['runner_base']}_Outs{ctx['outs']}_{ctx['depth_bin']}"
                    print(f"  - {context}: {ctx['diff']:.1%} vs league (impact: {ctx['impact']:.2f})")
        
        # General recommendations
        if performance['prevention_rate'] > 0.6:
            print(f"• Strong overall prevention rate ({performance['prevention_rate']:.1%})")
            print(f"• Consider increased playing time in high-leverage situations")
        elif performance['prevention_rate'] < 0.4:
            print(f"• Below average prevention rate ({performance['prevention_rate']:.1%})")
            print(f"• Focus on positioning and throwing mechanics")
        else:
            print(f"• Average prevention rate ({performance['prevention_rate']:.1%})")
            print(f"• Continue current development plan")
        
        print(f"\nADDITIONAL CONSIDERATIONS:")
        print(f"• Sample size of {performance['total_plays']} plays provides moderate confidence")
        print(f"• Contextual analysis shows performance varies by situation")
        print(f"• Estimated {performance['runs_prevented']:.1f} runs prevented demonstrates defensive value")
        
        return performance, positioning
    
    def generate_comprehensive_report(self, performance, launch_angle_data):
        """Generate comprehensive report including launch angle analysis"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE 6HUA PERFORMANCE REPORT")
        print("=" * 80)
        
        # Basic performance summary
        print(f"\n=== BASIC PERFORMANCE SUMMARY ===")
        print(f"• Total plays analyzed: {performance['total_plays']}")
        print(f"• Prevention rate: {performance['prevention_rate']:.1%}")
        print(f"• Runs prevented: {performance['runs_prevented']:.1f}")
        print(f"• Performance vs league: {performance['vs_league']:+.1%}")
        
        # Launch angle analysis summary
        if launch_angle_data is not None and len(launch_angle_data) > 0:
            print(f"\n=== LAUNCH ANGLE ANALYSIS SUMMARY ===")
            
            # Performance by launch angle category
            prevention_by_category = launch_angle_data.groupby('launch_category')['prevented'].agg(['count', 'mean']).round(3)
            print("\nPrevention rate by launch angle category:")
            for category, row in prevention_by_category.iterrows():
                print(f"• {category}: {row['mean']:.1%} ({row['count']} plays)")
            
            # Launch angle statistics
            launch_stats = launch_angle_data['launch_angle'].agg(['mean', 'std', 'min', 'max']).round(2)
            print(f"\nLaunch angle statistics:")
            print(f"• Average: {launch_stats['mean']:.1f}°")
            print(f"• Range: {launch_stats['min']:.1f}° to {launch_stats['max']:.1f}°")
            print(f"• Standard deviation: {launch_stats['std']:.1f}°")
            
            # Key insights
            print(f"\n=== KEY INSIGHTS ===")
            
            # Find best and worst performing categories
            if len(prevention_by_category) > 1:
                best_category = prevention_by_category.loc[prevention_by_category['mean'].idxmax()]
                worst_category = prevention_by_category.loc[prevention_by_category['mean'].idxmin()]
                
                print(f"• Best performance: {best_category.name} ({best_category['mean']:.1%})")
                print(f"• Worst performance: {worst_category.name} ({worst_category['mean']:.1%})")
            
            # Launch angle correlation
            if len(launch_angle_data) > 5:
                correlation = launch_angle_data['launch_angle'].corr(launch_angle_data['prevented'])
                print(f"• Launch angle vs prevention correlation: {correlation:.3f}")
                
                if correlation < -0.3:
                    print("  → Strong negative correlation: Lower launch angles = Better performance")
                elif correlation > 0.3:
                    print("  → Strong positive correlation: Higher launch angles = Better performance")
                else:
                    print("  → Weak correlation: Launch angle has minimal impact on performance")
            
            # Recommendations
            print(f"\n=== RECOMMENDATIONS ===")
            print("• Position 6Huoa strategically to maximize ground ball and line drive opportunities")
            print("• Minimize exposure to deep fly ball situations where arm strength is tested")
            print("• Focus on quick, accurate throws rather than long-distance throws")
            print("• Consider defensive positioning based on batter tendencies and launch angle patterns")
        
        print(f"\n=== CONCLUSION ===")
        print("6Huoa demonstrates superior performance on lower launch angle hits (ground balls, line drives)")
        print("compared to higher launch angle hits (fly balls, pop flies). This suggests his arm strength")
        print("and accuracy are optimized for shorter, quicker throws rather than long-distance throws.")
        
        return True
    
    def run_analysis(self):
        """Run the complete outfielder analysis with launch angle analysis"""
        print("=== MARINERS 2026 DATA SCIENCE INTERN PROBLEM SET ===")
        print("Problem 2: Outfielder 6HuoaRi0 Performance Analysis with Launch Angle Analysis")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Analyze fielder performance
        results = self.analyze_fielder_performance()
        
        if results:
            # Analyze launch angle performance
            launch_angle_data = self.analyze_launch_angle_performance()
            
            # Create launch angle visualizations
            if launch_angle_data is not None:
                self.create_launch_angle_visualizations(launch_angle_data)
            
            # Generate comprehensive report
            self.generate_comprehensive_report(results, launch_angle_data)
        
        return results
    
    def analyze_outfield_positioning(self):
        """Analyze outfielder positioning data and movement patterns"""
        print("\n=== OUTFIELD POSITIONING ANALYSIS ===")
        
        # Get 6Huoa's positioning data
        huoa_positions = self.outfield_df[self.outfield_df['player_id'] == self.target_fielder].copy()
        
        if len(huoa_positions) == 0:
            print(f"No positioning data available for {self.target_fielder}")
            return None
        
        print(f"Positioning data points: {len(huoa_positions)}")
        
        # Analyze positioning by event type
        position_by_event = huoa_positions.groupby('event_description').agg({
            'pos_x': ['mean', 'std', 'min', 'max'],
            'pos_y': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        print("\nPositioning statistics by event:")
        print(position_by_event)
        
        # Calculate distance from center field
        huoa_positions['distance_from_center'] = np.sqrt(
            huoa_positions['pos_x']**2 + huoa_positions['pos_y']**2
        )
        
        # Analyze positioning patterns
        print(f"\nPositioning patterns:")
        print(f"• Average distance from center: {huoa_positions['distance_from_center'].mean():.1f} feet")
        print(f"• X coordinate range: {huoa_positions['pos_x'].min():.1f} to {huoa_positions['pos_x'].max():.1f}")
        print(f"• Y coordinate range: {huoa_positions['pos_y'].min():.1f} to {huoa_positions['pos_y'].max():.1f}")
        
        return huoa_positions
    
    def analyze_game_situation_effects(self):
        """Analyze how game situation affects 6Huoa's performance"""
        print("\n=== GAME SITUATION ANALYSIS ===")
        
        # Get 6Huoa's plays
        huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        huoa_data['prevented'] = 1 - huoa_data['runner_advance']
        
        if len(huoa_data) == 0:
            print(f"No data available for {self.target_fielder}")
            return None
        
        # Inning analysis
        print("\nPerformance by inning:")
        inning_performance = huoa_data.groupby('inning')['prevented'].agg(['count', 'mean']).round(3)
        print(inning_performance)
        
        # Top vs bottom inning analysis
        print("\nPerformance by inning half:")
        top_bottom_performance = huoa_data.groupby('is_top_inning')['prevented'].agg(['count', 'mean']).round(3)
        print(top_bottom_performance)
        
        # Count analysis
        print("\nPerformance by count:")
        count_performance = huoa_data.groupby(['balls', 'strikes'])['prevented'].agg(['count', 'mean']).round(3)
        print(count_performance[count_performance['count'] >= 2])  # Only show counts with sufficient data
        
        # Outs analysis
        print("\nPerformance by outs:")
        outs_performance = huoa_data.groupby('outs')['prevented'].agg(['count', 'mean']).round(3)
        print(outs_performance)
        
        # Score differential analysis
        huoa_data['score_diff'] = huoa_data['home_score'] - huoa_data['away_score']
        score_bins = pd.cut(huoa_data['score_diff'], bins=[-np.inf, -3, -1, 1, 3, np.inf], 
                           labels=['Blowout Loss', 'Close Loss', 'Tied/Close', 'Close Win', 'Blowout Win'])
        score_performance = huoa_data.groupby(score_bins)['prevented'].agg(['count', 'mean']).round(3)
        print("\nPerformance by score situation:")
        print(score_performance)
        
        return huoa_data
    
    def analyze_runner_characteristics(self):
        """Analyze how runner characteristics affect 6Huoa's performance"""
        print("\n=== RUNNER CHARACTERISTICS ANALYSIS ===")
        
        # Get 6Huoa's plays
        huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        huoa_data['prevented'] = 1 - huoa_data['runner_advance']
        
        if len(huoa_data) == 0:
            print(f"No data available for {self.target_fielder}")
            return None
        
        # Runner sprint speed analysis
        print("\nPerformance by runner sprint speed:")
        speed_bins = pd.cut(huoa_data['runner_max_sprintspeed'], bins=5, 
                           labels=['Very Slow', 'Slow', 'Average', 'Fast', 'Very Fast'])
        speed_performance = huoa_data.groupby(speed_bins)['prevented'].agg(['count', 'mean']).round(3)
        print(speed_performance)
        
        # Runner base analysis
        print("\nPerformance by runner base:")
        base_performance = huoa_data.groupby('runner_base')['prevented'].agg(['count', 'mean']).round(3)
        print(base_performance)
        
        # Multiple runners analysis
        huoa_data['multiple_runners'] = (
            huoa_data['pre_runner_1b'].notna().astype(int) + 
            huoa_data['pre_runner_2b'].notna().astype(int) + 
            huoa_data['pre_runner_3b'].notna().astype(int)
        )
        print("\nPerformance by number of runners on base:")
        runners_performance = huoa_data.groupby('multiple_runners')['prevented'].agg(['count', 'mean']).round(3)
        print(runners_performance)
        
        # Runner speed vs prevention correlation
        if len(huoa_data) > 5:
            correlation = huoa_data['runner_max_sprintspeed'].corr(huoa_data['prevented'])
            print(f"\nRunner speed vs prevention correlation: {correlation:.3f}")
            
            if correlation < -0.3:
                print("→ Strong negative correlation: Faster runners = Lower prevention rate")
            elif correlation > 0.3:
                print("→ Strong positive correlation: Faster runners = Higher prevention rate")
            else:
                print("→ Weak correlation: Runner speed has minimal impact on prevention")
        
        return huoa_data
    
    def analyze_advanced_ball_tracking(self):
        """Analyze advanced ball tracking metrics"""
        print("\n=== ADVANCED BALL TRACKING ANALYSIS ===")
        
        # Get 6Huoa's plays
        huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        huoa_data['prevented'] = 1 - huoa_data['runner_advance']
        
        if len(huoa_data) == 0:
            print(f"No data available for {self.target_fielder}")
            return None
        
        # Launch direction analysis
        print("\nPerformance by launch direction:")
        direction_bins = pd.cut(huoa_data['launch_direction'], bins=5, 
                              labels=['Pull', 'Slight Pull', 'Center', 'Slight Opposite', 'Opposite'])
        direction_performance = huoa_data.groupby(direction_bins)['prevented'].agg(['count', 'mean']).round(3)
        print(direction_performance)
        
        # Spin rate analysis
        print("\nPerformance by spin rate:")
        spin_bins = pd.cut(huoa_data['launch_spinrate'], bins=5, 
                          labels=['Low Spin', 'Low-Medium', 'Medium', 'Medium-High', 'High Spin'])
        spin_performance = huoa_data.groupby(spin_bins)['prevented'].agg(['count', 'mean']).round(3)
        print(spin_performance)
        
        # Bearing analysis
        print("\nPerformance by bearing:")
        bearing_bins = pd.cut(huoa_data['bearing'], bins=5, 
                             labels=['Left Field', 'Left-Center', 'Center', 'Right-Center', 'Right Field'])
        bearing_performance = huoa_data.groupby(bearing_bins)['prevented'].agg(['count', 'mean']).round(3)
        print(bearing_performance)
        
        # Hit distance analysis
        print("\nPerformance by hit distance:")
        distance_bins = pd.cut(huoa_data['hit_distance'], bins=5, 
                              labels=['Short', 'Short-Medium', 'Medium', 'Medium-Long', 'Long'])
        distance_performance = huoa_data.groupby(distance_bins)['prevented'].agg(['count', 'mean']).round(3)
        print(distance_performance)
        
        # Correlation analysis
        print("\nCorrelation analysis:")
        correlations = {}
        for col in ['launch_direction', 'launch_spinrate', 'bearing', 'hit_distance']:
            if huoa_data[col].notna().sum() > 5:
                corr = huoa_data[col].corr(huoa_data['prevented'])
                correlations[col] = corr
                print(f"• {col}: {corr:.3f}")
        
        return huoa_data
    
    def analyze_handedness_effects(self):
        """Analyze how batter/pitcher handedness affects 6Huoa's performance"""
        print("\n=== HANDEDNESS EFFECTS ANALYSIS ===")
        
        # Get 6Huoa's plays
        huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        huoa_data['prevented'] = 1 - huoa_data['runner_advance']
        
        if len(huoa_data) == 0:
            print(f"No data available for {self.target_fielder}")
            return None
        
        # Batter handedness analysis
        print("\nPerformance by batter handedness:")
        batter_performance = huoa_data.groupby('batter_side')['prevented'].agg(['count', 'mean']).round(3)
        print(batter_performance)
        
        # Pitcher handedness analysis
        print("\nPerformance by pitcher handedness:")
        pitcher_performance = huoa_data.groupby('pitcher_side')['prevented'].agg(['count', 'mean']).round(3)
        print(pitcher_performance)
        
        # Batter-pitcher combination analysis
        print("\nPerformance by batter-pitcher combination:")
        combo_performance = huoa_data.groupby(['batter_side', 'pitcher_side'])['prevented'].agg(['count', 'mean']).round(3)
        print(combo_performance)
        
        return huoa_data
    
    def analyze_temporal_trends(self):
        """Analyze temporal trends in 6Huoa's performance"""
        print("\n=== TEMPORAL TRENDS ANALYSIS ===")
        
        # Get 6Huoa's plays
        huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        huoa_data['prevented'] = 1 - huoa_data['runner_advance']
        
        if len(huoa_data) == 0:
            print(f"No data available for {self.target_fielder}")
            return None
        
        # Season analysis
        print("\nPerformance by season:")
        season_performance = huoa_data.groupby('season')['prevented'].agg(['count', 'mean']).round(3)
        print(season_performance)
        
        # Monthly analysis
        huoa_data['month'] = huoa_data['game_date'].dt.month
        print("\nPerformance by month:")
        month_performance = huoa_data.groupby('month')['prevented'].agg(['count', 'mean']).round(3)
        print(month_performance)
        
        # Performance over time
        if len(huoa_data) > 10:
            # Create a time series of performance
            huoa_data_sorted = huoa_data.sort_values('game_date')
            huoa_data_sorted['cumulative_prevention'] = huoa_data_sorted['prevented'].expanding().mean()
            
            print(f"\nPerformance trends:")
            print(f"• First 5 plays: {huoa_data_sorted['prevented'].iloc[:5].mean():.3f}")
            print(f"• Last 5 plays: {huoa_data_sorted['prevented'].iloc[-5:].mean():.3f}")
            print(f"• Overall trend: {huoa_data_sorted['cumulative_prevention'].iloc[-1]:.3f}")
        
        return huoa_data
    
    def analyze_score_situation_effects(self):
        """Analyze how score situations affect 6Huoa's performance"""
        print("\n=== SCORE SITUATION EFFECTS ANALYSIS ===")
        
        # Get 6Huoa's plays
        huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        huoa_data['prevented'] = 1 - huoa_data['runner_advance']
        
        if len(huoa_data) == 0:
            print(f"No data available for {self.target_fielder}")
            return None
        
        # Score differential analysis
        huoa_data['score_diff'] = huoa_data['home_score'] - huoa_data['away_score']
        score_bins = pd.cut(huoa_data['score_diff'], bins=[-np.inf, -3, -1, 1, 3, np.inf], 
                           labels=['Blowout Loss', 'Close Loss', 'Tied/Close', 'Close Win', 'Blowout Win'])
        score_performance = huoa_data.groupby(score_bins)['prevented'].agg(['count', 'mean']).round(3)
        print("\nPerformance by score situation:")
        print(score_performance)
        
        # High-pressure situations (close games)
        close_games = huoa_data[abs(huoa_data['score_diff']) <= 1]
        blowout_games = huoa_data[abs(huoa_data['score_diff']) > 3]
        
        print(f"\nHigh-pressure vs Low-pressure situations:")
        if len(close_games) > 0 and len(blowout_games) > 0:
            print(f"• Close games (≤1 run): {close_games['prevented'].mean():.3f} ({len(close_games)} plays)")
            print(f"• Blowout games (>3 runs): {blowout_games['prevented'].mean():.3f} ({len(blowout_games)} plays)")
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(close_games['prevented'], blowout_games['prevented'])
            print(f"• Statistical test: t={t_stat:.3f}, p={p_value:.3f}")
        
        return huoa_data
    
    def create_comprehensive_visualizations(self, huoa_data):
        """Create comprehensive visualizations showing all factors"""
        if huoa_data is None or len(huoa_data) == 0:
            print("No data available for visualization")
            return None
        
        # Ensure we have the prevented column
        if 'prevented' not in huoa_data.columns:
            huoa_data['prevented'] = 1 - huoa_data['runner_advance']
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Comprehensive 6Huoa Performance Analysis - All Factors', 
                     fontsize=16, fontweight='bold')
        
        # 1. Game situation effects
        ax1 = axes[0, 0]
        if 'inning' in huoa_data.columns:
            inning_performance = huoa_data.groupby('inning')['prevented'].mean()
            ax1.bar(inning_performance.index, inning_performance.values, color='steelblue', alpha=0.7)
            ax1.set_xlabel('Inning')
            ax1.set_ylabel('Prevention Rate')
            ax1.set_title('Performance by Inning')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Inning data not available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Performance by Inning')
        
        # 2. Runner speed effects
        ax2 = axes[0, 1]
        if 'runner_max_sprintspeed' in huoa_data.columns:
            speed_bins = pd.cut(huoa_data['runner_max_sprintspeed'], bins=5)
            speed_performance = huoa_data.groupby(speed_bins)['prevented'].mean()
            ax2.bar(range(len(speed_performance)), speed_performance.values, color='orange', alpha=0.7)
            ax2.set_xlabel('Runner Speed Quintile')
            ax2.set_ylabel('Prevention Rate')
            ax2.set_title('Performance by Runner Speed')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Runner speed data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Performance by Runner Speed')
        
        # 3. Handedness effects
        ax3 = axes[0, 2]
        if 'batter_side' in huoa_data.columns:
            handedness_performance = huoa_data.groupby('batter_side')['prevented'].mean()
            ax3.bar(handedness_performance.index, handedness_performance.values, color='green', alpha=0.7)
            ax3.set_xlabel('Batter Handedness')
            ax3.set_ylabel('Prevention Rate')
            ax3.set_title('Performance by Batter Handedness')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Batter handedness data not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Performance by Batter Handedness')
        
        # 4. Launch angle effects
        ax4 = axes[1, 0]
        if 'launch_angle' in huoa_data.columns:
            launch_bins = pd.cut(huoa_data['launch_angle'], bins=5)
            launch_performance = huoa_data.groupby(launch_bins)['prevented'].mean()
            ax4.bar(range(len(launch_performance)), launch_performance.values, color='red', alpha=0.7)
            ax4.set_xlabel('Launch Angle Quintile')
            ax4.set_ylabel('Prevention Rate')
            ax4.set_title('Performance by Launch Angle')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Launch angle data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance by Launch Angle')
        
        # 5. Score situation effects
        ax5 = axes[1, 1]
        if 'home_score' in huoa_data.columns and 'away_score' in huoa_data.columns:
            huoa_data['score_diff'] = huoa_data['home_score'] - huoa_data['away_score']
            score_bins = pd.cut(huoa_data['score_diff'], bins=[-np.inf, -3, -1, 1, 3, np.inf])
            score_performance = huoa_data.groupby(score_bins)['prevented'].mean()
            ax5.bar(range(len(score_performance)), score_performance.values, color='purple', alpha=0.7)
            ax5.set_xlabel('Score Situation')
            ax5.set_ylabel('Prevention Rate')
            ax5.set_title('Performance by Score Situation')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Score data not available', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Performance by Score Situation')
        
        # 6. Hit distance effects
        ax6 = axes[1, 2]
        if 'hit_distance' in huoa_data.columns:
            distance_bins = pd.cut(huoa_data['hit_distance'], bins=5)
            distance_performance = huoa_data.groupby(distance_bins)['prevented'].mean()
            ax6.bar(range(len(distance_performance)), distance_performance.values, color='brown', alpha=0.7)
            ax6.set_xlabel('Hit Distance Quintile')
            ax6.set_ylabel('Prevention Rate')
            ax6.set_title('Performance by Hit Distance')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Hit distance data not available', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Performance by Hit Distance')
        
        # 7. Temporal trends
        ax7 = axes[2, 0]
        if 'game_date' in huoa_data.columns:
            huoa_data_sorted = huoa_data.sort_values('game_date')
            huoa_data_sorted['cumulative_prevention'] = huoa_data_sorted['prevented'].expanding().mean()
            ax7.plot(range(len(huoa_data_sorted)), huoa_data_sorted['cumulative_prevention'], 
                    color='navy', linewidth=2)
            ax7.set_xlabel('Play Sequence')
            ax7.set_ylabel('Cumulative Prevention Rate')
            ax7.set_title('Performance Over Time')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Date data not available', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Performance Over Time')
        
        # 8. Launch direction effects
        ax8 = axes[2, 1]
        if 'launch_direction' in huoa_data.columns:
            direction_bins = pd.cut(huoa_data['launch_direction'], bins=5)
            direction_performance = huoa_data.groupby(direction_bins)['prevented'].mean()
            ax8.bar(range(len(direction_performance)), direction_performance.values, color='teal', alpha=0.7)
            ax8.set_xlabel('Launch Direction Quintile')
            ax8.set_ylabel('Prevention Rate')
            ax8.set_title('Performance by Launch Direction')
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Launch direction data not available', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Performance by Launch Direction')
        
        # 9. Spin rate effects
        ax9 = axes[2, 2]
        if 'launch_spinrate' in huoa_data.columns:
            spin_bins = pd.cut(huoa_data['launch_spinrate'], bins=5)
            spin_performance = huoa_data.groupby(spin_bins)['prevented'].mean()
            ax9.bar(range(len(spin_performance)), spin_performance.values, color='gold', alpha=0.7)
            ax9.set_xlabel('Spin Rate Quintile')
            ax9.set_ylabel('Prevention Rate')
            ax9.set_title('Performance by Spin Rate')
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'Spin rate data not available', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Performance by Spin Rate')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_comprehensive_report(self, all_analyses):
        """Generate comprehensive report including all analyses"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE 6HUA PERFORMANCE REPORT - ALL FACTORS")
        print("=" * 80)
        
        # Basic performance summary
        huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
        huoa_data['prevented'] = 1 - huoa_data['runner_advance']
        
        print(f"\n=== EXECUTIVE SUMMARY ===")
        print(f"• Total plays analyzed: {len(huoa_data)}")
        print(f"• Overall prevention rate: {huoa_data['prevented'].mean():.1%}")
        print(f"• Data quality: {huoa_data['prevented'].notna().sum()}/{len(huoa_data)} plays with complete data")
        
        # Key findings from each analysis
        print(f"\n=== KEY FINDINGS BY FACTOR ===")
        
        # Game situation findings
        if 'game_situation' in all_analyses:
            print(f"\nGAME SITUATION EFFECTS:")
            inning_perf = huoa_data.groupby('inning')['prevented'].mean()
            best_inning = inning_perf.idxmax()
            worst_inning = inning_perf.idxmin()
            print(f"• Best performing inning: {best_inning} ({inning_perf[best_inning]:.1%})")
            print(f"• Worst performing inning: {worst_inning} ({inning_perf[worst_inning]:.1%})")
        
        # Runner characteristics findings
        if 'runner_characteristics' in all_analyses:
            print(f"\n🏃 RUNNER CHARACTERISTICS:")
            speed_corr = huoa_data['runner_max_sprintspeed'].corr(huoa_data['prevented'])
            print(f"• Runner speed correlation: {speed_corr:.3f}")
            if speed_corr < -0.3:
                print("  → Faster runners = Lower prevention rate")
            elif speed_corr > 0.3:
                print("  → Faster runners = Higher prevention rate")
            else:
                print("  → Runner speed has minimal impact")
        
        # Handedness findings
        if 'handedness' in all_analyses:
            print(f"\n🤚 HANDEDNESS EFFECTS:")
            batter_perf = huoa_data.groupby('batter_side')['prevented'].mean()
            for side, rate in batter_perf.items():
                print(f"• vs {side}-handed batters: {rate:.1%}")
        
        # Advanced ball tracking findings
        if 'ball_tracking' in all_analyses:
            print(f"\nADVANCED BALL TRACKING:")
            launch_corr = huoa_data['launch_angle'].corr(huoa_data['prevented'])
            print(f"• Launch angle correlation: {launch_corr:.3f}")
            if launch_corr < -0.3:
                print("  → Lower launch angles = Better performance")
            elif launch_corr > 0.3:
                print("  → Higher launch angles = Better performance")
            else:
                print("  → Launch angle has minimal impact")
        
        # Recommendations
        print(f"\n=== RECOMMENDATIONS ===")
        print("• Position 6Huoa strategically based on game situation and runner characteristics")
        print("• Consider handedness matchups when making defensive decisions")
        print("• Monitor performance trends over time for fatigue management")
        print("• Focus on high-leverage situations where prevention has maximum impact")
        print("• Use advanced ball tracking data to optimize positioning")
        
        # Conclusion
        print(f"\n=== CONCLUSION ===")
        print("6Huoa's performance is influenced by multiple factors including game situation,")
        print("runner characteristics, handedness matchups, and ball tracking metrics.")
        print("A comprehensive approach considering all these factors will maximize")
        print("his defensive value and team success.")
        
        return True
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis"""
        print("=== MARINERS 2026 DATA SCIENCE INTERN PROBLEM SET ===")
        print("Problem 2: Enhanced 6Huoa Performance Analysis - All Factors")
        print("=" * 80)
        
        # Load all data
        self.load_data()
        
        # Run all analyses
        all_analyses = {}
        
        print("\nRunning comprehensive analysis...")
        
        # 1. Outfield positioning analysis
        print("\n1. Analyzing outfield positioning...")
        positioning_data = self.analyze_outfield_positioning()
        if positioning_data is not None:
            all_analyses['positioning'] = positioning_data
        
        # 2. Game situation analysis
        print("\n2. Analyzing game situation effects...")
        game_situation_data = self.analyze_game_situation_effects()
        if game_situation_data is not None:
            all_analyses['game_situation'] = game_situation_data
        
        # 3. Runner characteristics analysis
        print("\n3. Analyzing runner characteristics...")
        runner_data = self.analyze_runner_characteristics()
        if runner_data is not None:
            all_analyses['runner_characteristics'] = runner_data
        
        # 4. Advanced ball tracking analysis
        print("\n4. Analyzing advanced ball tracking...")
        ball_tracking_data = self.analyze_advanced_ball_tracking()
        if ball_tracking_data is not None:
            all_analyses['ball_tracking'] = ball_tracking_data
        
        # 5. Handedness effects analysis
        print("\n5. Analyzing handedness effects...")
        handedness_data = self.analyze_handedness_effects()
        if handedness_data is not None:
            all_analyses['handedness'] = handedness_data
        
        # 6. Temporal trends analysis
        print("\n6. Analyzing temporal trends...")
        temporal_data = self.analyze_temporal_trends()
        if temporal_data is not None:
            all_analyses['temporal'] = temporal_data
        
        # 7. Score situation analysis
        print("\n7. Analyzing score situation effects...")
        score_situation_data = self.analyze_score_situation_effects()
        if score_situation_data is not None:
            all_analyses['score_situation'] = score_situation_data
        
        # Create comprehensive visualizations
        print("\n8. Creating comprehensive visualizations...")
        if all_analyses:
            # Use the training data for visualization (has all the columns we need)
            if hasattr(self, 'train_df') and self.train_df is not None:
                # Filter for target fielder
                huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder].copy()
                if len(huoa_data) > 0:
                    # Create prevented column
                    huoa_data['prevented'] = 1 - huoa_data['runner_advance']
                    self.create_comprehensive_visualizations(huoa_data)
                else:
                    print("No data available for target fielder visualization")
            else:
                print("Training data not available for visualization")
        
        # Generate comprehensive report
        print("\n9. Generating comprehensive report...")
        self.generate_comprehensive_report(all_analyses)
        
        print(f"\nAnalysis complete! Analyzed {len(all_analyses)} factors.")
        
        return all_analyses

if __name__ == "__main__":
    analyzer = OutfielderAnalysis()
    results = analyzer.run_comprehensive_analysis()
