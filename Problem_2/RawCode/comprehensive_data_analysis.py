#!/usr/bin/env python3
"""
Comprehensive analysis of all available data for 6Huoa performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ComprehensiveDataAnalysis:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.outfield_df = None
        self.context_df = None
        self.target_fielder = '6HuoaRi0'
        
    def load_all_data(self):
        """Load all available datasets"""
        print("=== LOADING ALL AVAILABLE DATA ===")
        
        # Load main datasets
        self.train_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Data/train_data.csv')
        self.test_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Data/test_data.csv')
        self.outfield_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Data/outfield_position.csv')
        self.context_df = pd.read_csv('/Users/klebb24/Desktop/MarinersProject/Problem 2/6HuoaRi0_context_table.csv')
        
        print(f"Training data: {self.train_df.shape}")
        print(f"Test data: {self.test_df.shape}")
        print(f"Outfield position data: {self.outfield_df.shape}")
        print(f"Context data: {self.context_df.shape}")
        
        return self
    
    def analyze_missing_data(self):
        """Analyze missing data patterns"""
        print("\n=== MISSING DATA ANALYSIS ===")
        
        missing_data = self.train_df.isnull().sum()
        missing_pct = (missing_data / len(self.train_df) * 100).round(1)
        missing_summary = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_pct
        }).sort_values('Missing Count', ascending=False)
        
        print("Missing data by column:")
        print(missing_summary[missing_summary['Missing Count'] > 0])
        
        return missing_summary
    
    def analyze_6huoa_data_availability(self):
        """Analyze 6Huoa's data availability across all datasets"""
        print("\n=== 6HUA DATA AVAILABILITY ===")
        
        # As fielder
        huoa_as_fielder = self.train_df[self.train_df['fielder_id'] == self.target_fielder]
        print(f"6Huoa as fielder in training data: {len(huoa_as_fielder)} plays")
        
        # As runner
        huoa_as_runner = self.train_df[self.train_df['runner_id'] == self.target_fielder]
        print(f"6Huoa as runner in training data: {len(huoa_as_runner)} plays")
        
        # In outfield position data
        huoa_positions = self.outfield_df[self.outfield_df['player_id'] == self.target_fielder]
        print(f"6Huoa in outfield position data: {len(huoa_positions)} position records")
        
        # In test data
        huoa_test_fielder = self.test_df[self.test_df['fielder_id'] == self.target_fielder]
        huoa_test_runner = self.test_df[self.test_df['runner_id'] == self.target_fielder]
        print(f"6Huoa in test data (as fielder): {len(huoa_test_fielder)} plays")
        print(f"6Huoa in test data (as runner): {len(huoa_test_runner)} plays")
        
        return {
            'fielder_train': len(huoa_as_fielder),
            'runner_train': len(huoa_as_runner),
            'positions': len(huoa_positions),
            'fielder_test': len(huoa_test_fielder),
            'runner_test': len(huoa_test_runner)
        }
    
    def analyze_additional_features(self):
        """Analyze additional features we haven't fully utilized"""
        print("\n=== ADDITIONAL FEATURES ANALYSIS ===")
        
        # Game context features
        print("Game context features:")
        print(f"• Unique games: {self.train_df['game_id'].nunique()}")
        print(f"• Date range: {self.train_df['game_date'].min()} to {self.train_df['game_date'].max()}")
        print(f"• Seasons: {sorted(self.train_df['season'].unique())}")
        print(f"• Levels: {self.train_df['level'].unique()}")
        
        # Count features
        print("\nCount features:")
        print(f"• Balls: {self.train_df['balls'].value_counts().sort_index()}")
        print(f"• Strikes: {self.train_df['strikes'].value_counts().sort_index()}")
        print(f"• Outs: {self.train_df['outs'].value_counts().sort_index()}")
        
        # Score features
        print("\nScore features:")
        print(f"• Home score range: {self.train_df['home_score'].min()} to {self.train_df['home_score'].max()}")
        print(f"• Away score range: {self.train_df['away_score'].min()} to {self.train_df['away_score'].max()}")
        
        # Handedness features
        print("\nHandedness features:")
        print(f"• Batter side: {self.train_df['batter_side'].value_counts()}")
        print(f"• Pitcher side: {self.train_df['pitcher_side'].value_counts()}")
        
        # Advanced ball tracking features
        print("\nAdvanced ball tracking features:")
        print(f"• Launch direction range: {self.train_df['launch_direction'].min():.1f} to {self.train_df['launch_direction'].max():.1f}")
        print(f"• Spin axis range: {self.train_df['launch_spinaxis'].min():.1f} to {self.train_df['launch_spinaxis'].max():.1f}")
        print(f"• Spin rate range: {self.train_df['launch_spinrate'].min():.1f} to {self.train_df['launch_spinrate'].max():.1f}")
        print(f"• Bearing range: {self.train_df['bearing'].min():.1f} to {self.train_df['bearing'].max():.1f}")
        
        # Player performance features
        print("\nPlayer performance features:")
        print(f"• Fielder positions: {self.train_df['fielder_pos'].value_counts()}")
        print(f"• Throw speed range: {self.train_df['fielder_max_throwspeed'].min():.1f} to {self.train_df['fielder_max_throwspeed'].max():.1f} mph")
        print(f"• Sprint speed range: {self.train_df['runner_max_sprintspeed'].min():.1f} to {self.train_df['runner_max_sprintspeed'].max():.1f} mph")
        
        return True
    
    def analyze_outfield_position_data(self):
        """Analyze the outfield position tracking data"""
        print("\n=== OUTFIELD POSITION DATA ANALYSIS ===")
        
        print(f"Total position records: {len(self.outfield_df)}")
        print(f"Unique plays: {self.outfield_df['play_id'].nunique()}")
        print(f"Unique players: {self.outfield_df['player_id'].nunique()}")
        
        # Event codes and descriptions
        print("\nEvent codes and descriptions:")
        event_summary = self.outfield_df.groupby(['event_code', 'event_description']).size().reset_index(name='count')
        print(event_summary)
        
        # Position analysis
        print("\nPosition analysis:")
        print(f"• X coordinate range: {self.outfield_df['pos_x'].min():.1f} to {self.outfield_df['pos_x'].max():.1f}")
        print(f"• Y coordinate range: {self.outfield_df['pos_y'].min():.1f} to {self.outfield_df['pos_y'].max():.1f}")
        
        # 6Huoa specific position data
        huoa_positions = self.outfield_df[self.outfield_df['player_id'] == self.target_fielder]
        if len(huoa_positions) > 0:
            print(f"\n6Huoa position data: {len(huoa_positions)} records")
            print(f"• X coordinate range: {huoa_positions['pos_x'].min():.1f} to {huoa_positions['pos_x'].max():.1f}")
            print(f"• Y coordinate range: {huoa_positions['pos_y'].min():.1f} to {huoa_positions['pos_y'].max():.1f}")
            
            # Event breakdown for 6Huoa
            huoa_events = huoa_positions.groupby(['event_code', 'event_description']).size().reset_index(name='count')
            print("\n6Huoa event breakdown:")
            print(huoa_events)
        
        return huoa_positions
    
    def analyze_contextual_factors(self):
        """Analyze contextual factors that might affect performance"""
        print("\n=== CONTEXTUAL FACTORS ANALYSIS ===")
        
        # Game situation factors
        print("Game situation factors:")
        
        # Inning analysis
        inning_advancement = self.train_df.groupby('inning')['runner_advance'].mean()
        print(f"\nAdvancement rate by inning:")
        for inning, rate in inning_advancement.items():
            print(f"  Inning {inning}: {rate:.3f}")
        
        # Top vs bottom inning
        top_bottom_advancement = self.train_df.groupby('is_top_inning')['runner_advance'].mean()
        print(f"\nAdvancement rate by inning half:")
        print(f"  Top of inning: {top_bottom_advancement[1]:.3f}")
        print(f"  Bottom of inning: {top_bottom_advancement[0]:.3f}")
        
        # Count analysis
        count_advancement = self.train_df.groupby(['balls', 'strikes'])['runner_advance'].mean().round(3)
        print(f"\nAdvancement rate by count (sample):")
        print(count_advancement.head(10))
        
        # Outs analysis
        outs_advancement = self.train_df.groupby('outs')['runner_advance'].mean()
        print(f"\nAdvancement rate by outs:")
        for outs, rate in outs_advancement.items():
            print(f"  {outs} outs: {rate:.3f}")
        
        # Score differential analysis
        self.train_df['score_diff'] = self.train_df['home_score'] - self.train_df['away_score']
        score_advancement = self.train_df.groupby(pd.cut(self.train_df['score_diff'], bins=5))['runner_advance'].mean()
        print(f"\nAdvancement rate by score differential:")
        print(score_advancement)
        
        return True
    
    def analyze_runner_characteristics(self):
        """Analyze runner characteristics and their impact"""
        print("\n=== RUNNER CHARACTERISTICS ANALYSIS ===")
        
        # Sprint speed analysis
        sprint_speed_advancement = self.train_df.groupby(pd.cut(self.train_df['runner_max_sprintspeed'], bins=5))['runner_advance'].mean()
        print("Advancement rate by runner sprint speed:")
        print(sprint_speed_advancement)
        
        # Runner base analysis
        base_advancement = self.train_df.groupby('runner_base')['runner_advance'].mean()
        print(f"\nAdvancement rate by runner base:")
        for base, rate in base_advancement.items():
            print(f"  Base {base}: {rate:.3f}")
        
        # Runner count analysis
        runner_count = self.train_df.groupby(['pre_runner_1b', 'pre_runner_2b', 'pre_runner_3b']).size().sort_values(ascending=False)
        print(f"\nMost common runner situations:")
        print(runner_count.head(10))
        
        return True
    
    def analyze_fielder_characteristics(self):
        """Analyze fielder characteristics and their impact"""
        print("\n=== FIELDER CHARACTERISTICS ANALYSIS ===")
        
        # Throw speed analysis
        throw_speed_advancement = self.train_df.groupby(pd.cut(self.train_df['fielder_max_throwspeed'], bins=5))['runner_advance'].mean()
        print("Advancement rate by fielder throw speed:")
        print(throw_speed_advancement)
        
        # Position analysis
        position_advancement = self.train_df.groupby('fielder_pos')['runner_advance'].mean()
        print(f"\nAdvancement rate by fielder position:")
        for pos, rate in position_advancement.items():
            print(f"  {pos}: {rate:.3f}")
        
        # 6Huoa specific analysis
        huoa_data = self.train_df[self.train_df['fielder_id'] == self.target_fielder]
        if len(huoa_data) > 0:
            print(f"\n6Huoa specific analysis:")
            print(f"  Throw speed: {huoa_data['fielder_max_throwspeed'].iloc[0]:.1f} mph")
            print(f"  Position: {huoa_data['fielder_pos'].iloc[0]}")
            print(f"  Prevention rate: {(1 - huoa_data['runner_advance']).mean():.3f}")
        
        return True
    
    def identify_missing_analyses(self):
        """Identify analyses we haven't performed yet"""
        print("\n=== MISSING ANALYSES IDENTIFICATION ===")
        
        missing_analyses = []
        
        # 1. Outfield positioning analysis
        if len(self.outfield_df) > 0:
            missing_analyses.append("Outfield positioning analysis - How does 6Huoa's position affect performance?")
        
        # 2. Game situation analysis
        missing_analyses.append("Game situation analysis - How do inning, count, outs affect 6Huoa's performance?")
        
        # 3. Runner characteristics analysis
        missing_analyses.append("Runner characteristics analysis - How do runner speed and base affect 6Huoa's performance?")
        
        # 4. Advanced ball tracking analysis
        missing_analyses.append("Advanced ball tracking analysis - Spin rate, spin axis, bearing effects")
        
        # 5. Handedness analysis
        missing_analyses.append("Handedness analysis - How do batter/pitcher handedness affect 6Huoa's performance?")
        
        # 6. Score situation analysis
        missing_analyses.append("Score situation analysis - How do game score and pressure situations affect performance?")
        
        # 7. Temporal analysis
        missing_analyses.append("Temporal analysis - How does performance change over time/season?")
        
        # 8. Weather/field conditions (if available)
        missing_analyses.append("Environmental factors analysis - Field conditions, weather effects")
        
        print("Analyses we haven't performed yet:")
        for i, analysis in enumerate(missing_analyses, 1):
            print(f"{i}. {analysis}")
        
        return missing_analyses
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis of all available data"""
        print("=== COMPREHENSIVE DATA ANALYSIS FOR 6HUA ===")
        print("=" * 60)
        
        # Load all data
        self.load_all_data()
        
        # Analyze missing data
        missing_summary = self.analyze_missing_data()
        
        # Analyze 6Huoa data availability
        huoa_availability = self.analyze_6huoa_data_availability()
        
        # Analyze additional features
        self.analyze_additional_features()
        
        # Analyze outfield position data
        huoa_positions = self.analyze_outfield_position_data()
        
        # Analyze contextual factors
        self.analyze_contextual_factors()
        
        # Analyze runner characteristics
        self.analyze_runner_characteristics()
        
        # Analyze fielder characteristics
        self.analyze_fielder_characteristics()
        
        # Identify missing analyses
        missing_analyses = self.identify_missing_analyses()
        
        print("\n=== SUMMARY ===")
        print(f"Total data points available: {len(self.train_df)} training plays")
        print(f"6Huoa data points: {huoa_availability['fielder_train']} as fielder, {huoa_availability['runner_train']} as runner")
        print(f"Position tracking records: {huoa_availability['positions']}")
        print(f"Missing analyses identified: {len(missing_analyses)}")
        
        return {
            'missing_summary': missing_summary,
            'huoa_availability': huoa_availability,
            'huoa_positions': huoa_positions,
            'missing_analyses': missing_analyses
        }

if __name__ == "__main__":
    analyzer = ComprehensiveDataAnalysis()
    results = analyzer.run_comprehensive_analysis()
