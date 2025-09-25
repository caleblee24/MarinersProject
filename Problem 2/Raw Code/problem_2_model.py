#!/usr/bin/env python3
"""
Mariners 2026 Data Science Intern Problem Set - Problem 2
Outfielder 6HuoaRi0 Performance Analysis Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class OutfielderAnalysis:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.outfield_df = None
        self.target_fielder = '6HuoaRi0'
        
    def load_data(self):
        """Load the datasets"""
        print("Loading data for outfielder analysis...")
        self.train_df = pd.read_csv('train_data.csv')
        self.test_df = pd.read_csv('test_data.csv')
        self.outfield_df = pd.read_csv('outfield_position.csv')
        return self
    
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
    
    def run_analysis(self):
        """Run the complete outfielder analysis"""
        self.load_data()
        return self.generate_report()

if __name__ == "__main__":
    analyzer = OutfielderAnalysis()
    performance, positioning = analyzer.run_analysis()
