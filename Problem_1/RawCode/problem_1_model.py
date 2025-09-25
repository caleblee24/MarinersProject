"""
Mariners 2026 Data Science Intern Problem Set - Problem 1
Enhanced Runner Advancement Prediction Model - Comprehensive Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class EnhancedRunnerAdvancementModel:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.outfield_df = None
        self.feature_columns = []
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load all available datasets with flexible path detection"""
        print("=== LOADING ALL AVAILABLE DATA ===")
        
        # Try multiple possible data locations
        data_paths = [
            ('Data/train_data.csv', 'Data/test_data.csv', 'Data/outfield_position.csv'),
            ('../Data/train_data.csv', '../Data/test_data.csv', '../Data/outfield_position.csv'),
            ('../../Data/train_data.csv', '../../Data/test_data.csv', '../../Data/outfield_position.csv'),
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
            raise FileNotFoundError("Data files not found")
        
        # Convert date columns
        self.train_df['game_date'] = pd.to_datetime(self.train_df['game_date'])
        self.test_df['game_date'] = pd.to_datetime(self.test_df['game_date'])
        
        print(f"Training data: {self.train_df.shape}")
        print(f"Test data: {self.test_df.shape}")
        print(f"Outfield position data: {self.outfield_df.shape}")
        
        return self
    
    
    def oof_contextual_baseline(self, df, ctx_cols, groups, n_splits=5):
        """Calculate out-of-fold contextual baseline"""
        from sklearn.model_selection import GroupKFold
        
        gkf = GroupKFold(n_splits=n_splits)
        oof_preds = np.zeros(len(df))
        
        for train_idx, val_idx in gkf.split(df, groups=groups):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            
            # Calculate contextual baseline for validation fold
            for _, row in val_fold.iterrows():
                context_mask = True
                for col in ctx_cols:
                    context_mask &= (train_fold[col] == row[col])
                
                if context_mask.any():
                    oof_preds[val_idx[val_fold.index == row.name]] = train_fold[context_mask]['runner_advance'].mean()
                else:
                    oof_preds[val_idx[val_fold.index == row.name]] = train_fold['runner_advance'].mean()
        
        return oof_preds
    
    def merge_outfield_tracking(self):
        """Merge outfield tracking data with main datasets"""
        print("\n=== MERGING OUTFIELD TRACKING DATA ===")
        
        # Build tracking aggregations
        tracking_agg = self.build_tracking_agg(self.outfield_df)
        
        # Merge with train and test data
        self.train_df = self.train_df.merge(tracking_agg, on='play_id', how='left')
        self.test_df = self.test_df.merge(tracking_agg, on='play_id', how='left')
        
        print(f"Outfield tracking data merged")
        return self.train_df, self.test_df
    
    def build_tracking_agg(self, ofdf):
        """Build outfield tracking aggregations"""
        # Group by play_id and calculate aggregations
        agg_dict = {
            'pos_x': ['mean', 'std', 'min', 'max'],
            'pos_y': ['mean', 'std', 'min', 'max']
        }
        
        tracking_agg = ofdf.groupby('play_id').agg(agg_dict).reset_index()
        
        # Flatten column names
        tracking_agg.columns = ['play_id'] + [f"{col[0]}_{col[1]}" for col in tracking_agg.columns[1:]]
        
        # Calculate distance from center (0, 0) for each position
        tracking_agg['distance_from_center_mean'] = np.sqrt(tracking_agg['pos_x_mean']**2 + tracking_agg['pos_y_mean']**2)
        tracking_agg['distance_from_center_std'] = np.sqrt(tracking_agg['pos_x_std']**2 + tracking_agg['pos_y_std']**2)
        
        return tracking_agg
    
    def engineer_comprehensive_features(self):
        """Create comprehensive feature set using all available data"""
        print("\n=== COMPREHENSIVE FEATURE ENGINEERING ===")
        
        for df in [self.train_df, self.test_df]:
            # 1. GAME CONTEXT FEATURES
            df['inning_phase'] = df['inning'].apply(lambda x: 'early' if x <= 3 else 'middle' if x <= 6 else 'late')
            df['score_diff'] = df['home_score'] - df['away_score']
            df['close_game'] = (df['score_diff'].abs() <= 2).astype(int)
            df['blowout'] = (df['score_diff'].abs() >= 5).astype(int)
            df['late_inning'] = (df['inning'] >= 7).astype(int)
            df['extra_inning'] = (df['inning'] >= 10).astype(int)
            df['high_leverage'] = ((df['inning'] >= 8) & (df['score_diff'].abs() <= 2)).astype(int)
            
            # 2. COUNT AND PRESSURE FEATURES
            df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
            df['three_oh_count'] = ((df['balls'] == 3) & (df['strikes'] == 0)).astype(int)
            df['oh_two_count'] = ((df['balls'] == 0) & (df['strikes'] == 2)).astype(int)
            df['two_strike_count'] = (df['strikes'] == 2).astype(int)
            df['no_outs'] = (df['outs'] == 0).astype(int)
            df['two_outs'] = (df['outs'] == 2).astype(int)
            
            # 3. RUNNER CONTEXT FEATURES
            df['runners_on'] = df[['pre_runner_1b', 'pre_runner_2b', 'pre_runner_3b']].notna().sum(axis=1)
            df['runners_in_scoring'] = df[['pre_runner_2b', 'pre_runner_3b']].notna().sum(axis=1)
            df['bases_loaded'] = (df['runners_on'] == 3).astype(int)
            df['runner_on_first'] = df['pre_runner_1b'].notna().astype(int)
            df['runner_on_second'] = df['pre_runner_2b'].notna().astype(int)
            df['runner_on_third'] = df['pre_runner_3b'].notna().astype(int)
            
            # 4. BALL TRACKING FEATURES
            df['launch_angle_category'] = pd.cut(df['launch_angle'], 
                                               bins=[-np.inf, 0, 15, 30, 45, np.inf],
                                               labels=['ground', 'line_drive', 'fly_ball', 'high_fly', 'pop_up'])
            df['hit_distance_category'] = pd.cut(df['hit_distance'],
                                               bins=[0, 200, 300, 400, 500, np.inf],
                                               labels=['short', 'medium', 'long', 'very_long', 'extreme'])
            df['exit_velocity_category'] = pd.cut(df['exit_speed'],
                                                bins=[0, 80, 90, 100, 110, np.inf],
                                                labels=['slow', 'medium', 'fast', 'very_fast', 'extreme'])
            df['spin_rate_category'] = pd.cut(df['launch_spinrate'],
                                            bins=[0, 1000, 2000, 3000, 4000, np.inf],
                                            labels=['low', 'medium', 'high', 'very_high', 'extreme'])
            
            # Convert categorical features to numeric for modeling
            df['launch_angle_category_num'] = df['launch_angle_category'].cat.codes
            df['hit_distance_category_num'] = df['hit_distance_category'].cat.codes
            df['exit_velocity_category_num'] = df['exit_velocity_category'].cat.codes
            df['spin_rate_category_num'] = df['spin_rate_category'].cat.codes
            
            # 5. ADVANCED BALL TRACKING
            df['hit_speed_per_distance'] = df['exit_speed'] / (df['hit_distance'] + 1)
            df['launch_efficiency'] = df['exit_speed'] * np.cos(np.radians(df['launch_angle']))
            df['carry_distance'] = df['hit_distance'] * np.sin(np.radians(df['launch_angle']))
            df['spin_axis_effect'] = df['launch_spinaxis'] * df['launch_spinrate'] / 1000
            
            # 6. FIELDER CONTEXT FEATURES
            df['fielder_position_category'] = df['fielder_pos'].apply(
                lambda x: 'infield' if x in ['C', '1B', '2B', '3B', 'SS'] 
                else 'outfield' if x in ['LF', 'CF', 'RF'] 
                else 'other'
            )
            df['fielder_throw_speed_category'] = pd.cut(df['fielder_max_throwspeed'],
                                                      bins=[0, 80, 90, 100, np.inf],
                                                      labels=['slow', 'medium', 'fast', 'very_fast'])
            
            # 7. RUNNER INTELLIGENCE FEATURES
            df['runner_speed_category'] = pd.cut(df['runner_max_sprintspeed'],
                                               bins=[0, 26, 28, 30, np.inf],
                                               labels=['slow', 'medium', 'fast', 'very_fast'])
            df['fast_runner'] = (df['runner_max_sprintspeed'] >= 28).astype(int)
            df['slow_runner'] = (df['runner_max_sprintspeed'] < 26).astype(int)
            
            # 8. BATTER/PITCHER MATCHUP FEATURES
            df['same_handedness'] = (df['batter_side'] == df['pitcher_side']).astype(int)
            df['batter_left_pitcher_right'] = ((df['batter_side'] == 'L') & (df['pitcher_side'] == 'R')).astype(int)
            df['batter_right_pitcher_left'] = ((df['batter_side'] == 'R') & (df['pitcher_side'] == 'L')).astype(int)
            
            # 9. TEMPORAL FEATURES
            df['month'] = df['game_date'].dt.month
            df['day_of_year'] = df['game_date'].dt.dayofyear
            df['early_season'] = (df['day_of_year'] <= 60).astype(int)
            df['late_season'] = (df['day_of_year'] >= 200).astype(int)
            
            # 10. SITUATIONAL PRESSURE FEATURES
            df['pressure_situation'] = (
                (df['late_inning'] == 1) & 
                (df['close_game'] == 1) & 
                (df['runners_in_scoring'] > 0)
            ).astype(int)
            df['must_score_situation'] = (
                (df['late_inning'] == 1) & 
                (df['score_diff'] < 0) & 
                (df['runners_in_scoring'] > 0)
            ).astype(int)
            
            # 11. INTERACTION FEATURES
            df['fast_runner_close_game'] = df['fast_runner'] * df['close_game']
            df['high_leverage_fast_runner'] = df['high_leverage'] * df['fast_runner']
            df['late_inning_pressure'] = df['late_inning'] * df['pressure_situation']
            
        print(f"Created comprehensive feature set with {len([col for col in self.train_df.columns if col not in ['play_id', 'game_id', 'runner_advance']])} features")
        
        return self.train_df, self.test_df
    
    def create_statistical_baselines(self):
        """Create statistical baseline features"""
        print("\n=== CREATING STATISTICAL BASELINES ===")
        
        # Player-specific baselines
        fielder_baselines = self.train_df.groupby('fielder_id')['runner_advance'].mean()
        runner_baselines = self.train_df.groupby('runner_id')['runner_advance'].mean()
        
        # Situational baselines
        situation_baselines = self.train_df.groupby(['inning', 'outs', 'runners_on'])['runner_advance'].mean()
        
        for df in [self.train_df, self.test_df]:
            # Add player baselines
            df['fielder_baseline'] = df['fielder_id'].map(fielder_baselines).fillna(df['runner_advance'].mean())
            df['runner_baseline'] = df['runner_id'].map(runner_baselines).fillna(df['runner_advance'].mean())
            
            # Add situational baselines
            df['situation_baseline'] = df.set_index(['inning', 'outs', 'runners_on']).index.map(
                lambda x: situation_baselines.get(x, df['runner_advance'].mean())
            )
            
            # Fill any remaining NaNs
            df['fielder_baseline'] = df['fielder_baseline'].fillna(df['runner_advance'].mean())
            df['runner_baseline'] = df['runner_baseline'].fillna(df['runner_advance'].mean())
            df['situation_baseline'] = df['situation_baseline'].fillna(df['runner_advance'].mean())
        
        return self.train_df, self.test_df
    
    def prepare_features(self):
        """Prepare final feature set for modeling"""
        print("\n=== PREPARING FEATURES FOR MODELING ===")
        
        # Select feature columns (exclude target, ID columns, and original categorical columns)
        exclude_cols = ['play_id', 'game_id', 'runner_advance', 'game_date', 'season', 'level',
                       'launch_angle_category', 'hit_distance_category', 'exit_velocity_category', 
                       'spin_rate_category', 'fielder_position_category', 'fielder_throw_speed_category',
                       'runner_speed_category', 'inning_phase']
        self.feature_columns = [col for col in self.train_df.columns if col not in exclude_cols]
        
        # Handle categorical variables
        categorical_cols = [col for col in self.feature_columns if self.train_df[col].dtype == 'object']
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on combined data to handle unseen categories
            combined_data = pd.concat([self.train_df[col], self.test_df[col]]).fillna('Unknown')
            le.fit(combined_data)
            
            self.train_df[col] = le.transform(self.train_df[col].fillna('Unknown'))
            self.test_df[col] = le.transform(self.test_df[col].fillna('Unknown'))
            self.label_encoders[col] = le
        
        # Handle missing values - separate numeric and categorical
        X_train = self.train_df[self.feature_columns]
        X_test = self.test_df[self.feature_columns]
        
        # Identify numeric and categorical columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns
        
        # Impute numeric columns with median
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])
        
        # Impute categorical columns with mode
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])
        
        print(f"Final feature set: {len(self.feature_columns)} features")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        return X_train, X_test
    
    def train_enhanced_models(self, X_train, y_train):
        """Train enhanced ensemble of models"""
        print("\n=== TRAINING ENHANCED MODELS ===")
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'HistGradientBoosting': HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.1, max_depth=10,
                min_samples_leaf=20, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                min_samples_split=20, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=42, solver='liblinear'
            )
        }
        
        # Train and evaluate models
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use GroupKFold for proper validation
            groups = self.train_df['game_id']
            gkf = GroupKFold(n_splits=5)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=gkf.split(X_train, groups=groups), 
                                      scoring='neg_log_loss', n_jobs=-1)
            
            # Train on full data
            model.fit(X_train, y_train)
            
            # Calibrate the model
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(X_train, y_train)
            
            model_scores[name] = -cv_scores.mean()
            trained_models[name] = calibrated_model
            
            print(f"{name} CV Log Loss: {model_scores[name]:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model
        best_model_name = min(model_scores, key=model_scores.get)
        best_model = trained_models[best_model_name]
        best_score = model_scores[best_model_name]
        
        print(f"\nBest model: {best_model_name} with CV Log Loss: {best_score:.4f}")
        
        return best_model, best_score, trained_models
    
    def run_comprehensive_analysis(self):
        """Run the complete enhanced analysis"""
        print("=== MARINERS 2026 DATA SCIENCE INTERN PROBLEM SET ===")
        print("Problem 1: Enhanced Runner Advancement Prediction - Comprehensive Analysis")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Calculate constant baseline first
        p = self.train_df['runner_advance'].mean()
        baseline_ll = -p * np.log(p) - (1-p) * np.log(1-p)
        print(f"\n=== CALCULATING BASELINES ===")
        print(f"Constant baseline (always predict {p:.3f}): {baseline_ll:.4f}")
        
        # Merge outfield tracking data
        self.merge_outfield_tracking()
        
        # Engineer comprehensive features
        self.engineer_comprehensive_features()
        
        # Calculate contextual baseline after feature engineering
        ctx_cols = ['inning', 'outs', 'runners_on']
        groups = self.train_df['game_id'].values
        
        oof_preds = self.oof_contextual_baseline(self.train_df, ctx_cols, groups)
        contextual_ll = log_loss(self.train_df['runner_advance'], oof_preds)
        print(f"Contextual baseline: {contextual_ll:.4f}")
        
        # Create statistical baselines
        self.create_statistical_baselines()
        
        # Prepare features
        X_train, X_test = self.prepare_features()
        y_train = self.train_df['runner_advance']
        groups = self.train_df['game_id'].values
        
        # Train models
        best_model, best_score, all_models = self.train_enhanced_models(X_train, y_train)
        
        # Calibrate the best model
        print("\n=== CALIBRATING BEST MODEL ===")
        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate calibrated model
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(calibrated_model, X_train, y_train, cv=5, scoring='neg_log_loss')
        calibrated_score = -cv_scores.mean()
        print(f"Calibrated model CV Log Loss: {calibrated_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Make predictions
        predictions = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Create submission
        submission = pd.DataFrame({
            'play_id': self.test_df['play_id'],
            'runner_advance': predictions
        })
        
        # Save predictions
        submission.to_csv('enhanced_runner_advance_predictions.csv', index=False)
        print(f"\nEnhanced predictions saved to enhanced_runner_advance_predictions.csv")
        print(f"Prediction statistics:")
        print(f"Mean probability: {predictions.mean():.3f}")
        print(f"Std probability: {predictions.std():.3f}")
        
        # Final results summary
        print("\n=== FINAL RESULTS ===")
        print(f"Constant baseline: {baseline_ll:.4f}")
        print(f"Contextual baseline: {contextual_ll:.4f}")
        print(f"Final model performance: {calibrated_score:.4f}")
        improvement_over_constant = baseline_ll - calibrated_score
        improvement_over_contextual = contextual_ll - calibrated_score
        print(f"Improvement over constant baseline: {improvement_over_constant:+.4f}")
        print(f"Improvement over contextual baseline: {improvement_over_contextual:+.4f}")
        
        return calibrated_model, predictions, submission, best_model

if __name__ == "__main__":
    model = EnhancedRunnerAdvancementModel()
    best_model, predictions, submission = model.run_comprehensive_analysis()
