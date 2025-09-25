"""
Mariners 2026 Data Science Intern Problem Set - Problem 1
Runner Advancement Prediction Model (Core Functions Only)

This file contains only the core data processing and modeling functions.
All visualization functions have been moved to the Jupyter notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the datasets"""
    print("Loading data...")
    train_df = pd.read_csv('../Data/train_data.csv')
    test_df = pd.read_csv('../Data/test_data.csv')
    outfield_df = pd.read_csv('../Data/outfield_position.csv')
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Outfield data shape: {outfield_df.shape}")
    
    return train_df, test_df, outfield_df

def calculate_baselines(train_df):
    """Calculate baseline performance metrics"""
    print("\n=== CALCULATING BASELINES ===")
    
    # Constant baseline (always predict the mean)
    p = train_df['runner_advance'].mean()
    baseline_ll = -p * np.log(p) - (1-p) * np.log(1-p)
    
    print(f"Constant baseline (always predict {p:.3f}): {baseline_ll:.4f}")
    
    return p, baseline_ll

def oof_contextual_baseline(df, ctx_cols, groups, n_splits=5):
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

def engineer_features(train_df, test_df):
    """Create enhanced features from the raw data"""
    print("\n=== FEATURE ENGINEERING ===")
    
    for df in [train_df, test_df]:
        # Game context features
        df['inning_phase'] = df['inning'].apply(lambda x: 'early' if x <= 3 else 'middle' if x <= 6 else 'late')
        df['score_diff'] = df['home_score'] - df['away_score']
        df['close_game'] = (df['score_diff'].abs() <= 2).astype(int)
        
        # Batted ball features
        df['hit_distance_bin'] = pd.cut(df['hit_distance'], bins=[0, 200, 300, 400, np.inf], 
                                        labels=['short', 'medium', 'long', 'very_long'])
        df['hit_angle_bin'] = pd.cut(df['hit_angle'], bins=[-np.inf, 0, 15, 30, np.inf], 
                                    labels=['ground', 'line_drive', 'fly_ball', 'pop_up'])
        
        # Situational features
        df['runners_on'] = df['runner_1b'].fillna(0) + df['runner_2b'].fillna(0) + df['runner_3b'].fillna(0)
        df['runners_in_scoring'] = df['runner_2b'].fillna(0) + df['runner_3b'].fillna(0)
        df['late_inning'] = (df['inning'] >= 7).astype(int)
        df['high_leverage'] = ((df['inning'] >= 8) & (df['score_diff'].abs() <= 2)).astype(int)
        
        # Pitcher features
        df['pitcher_hand_vs_batter'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
        
        # Advanced features
        df['hit_speed_per_distance'] = df['hit_speed'] / (df['hit_distance'] + 1)
        df['total_bases'] = df['hit_distance'] * df['hit_angle'].abs() / 1000
        
    print(f"Added {len([col for col in train_df.columns if col not in ['play_id', 'game_id', 'runner_advance']])} features")
    
    return train_df, test_df

def build_tracking_agg(ofdf):
    """Build tracking data aggregations"""
    agg_dict = {
        'pos_x': ['mean', 'std', 'min', 'max'],
        'pos_y': ['mean', 'std', 'min', 'max'],
        'velo': ['mean', 'std', 'max'],
        'accel': ['mean', 'std', 'max']
    }
    
    tracking_agg = ofdf.groupby('play_id').agg(agg_dict).round(2)
    tracking_agg.columns = [f'track_{col[0]}_{col[1]}' for col in tracking_agg.columns]
    tracking_agg = tracking_agg.reset_index()
    
    return tracking_agg

def merge_outfield_tracking(train_df, test_df, outfield_df):
    """Merge outfield tracking data"""
    print("\n=== MERGING TRACKING DATA ===")
    
    # Build tracking aggregations
    tracking_agg = build_tracking_agg(outfield_df)
    
    # Merge with train and test data
    train_df = train_df.merge(tracking_agg, on='play_id', how='left')
    test_df = test_df.merge(tracking_agg, on='play_id', how='left')
    
    print(f"Added {len(tracking_agg.columns)-1} tracking features")
    
    return train_df, test_df

def safe_fill(df, cols, val=0, kind='num'):
    """Safely fill missing values"""
    for col in cols:
        if col in df.columns:
            if kind == 'num':
                df[col] = df[col].fillna(val)
            else:
                df[col] = df[col].fillna(val)

def prepare_features(train_df, test_df):
    """Prepare features for modeling"""
    print("\n=== PREPARING FEATURES ===")
    
    # Identify feature columns
    exclude_cols = ['play_id', 'game_id', 'runner_advance']
    available_features = []
    
    for col in train_df.columns:
        if col not in exclude_cols:
            if train_df[col].dtype == 'object':
                # Categorical features
                le = LabelEncoder()
                train_df[col] = train_df[col].fillna('Unknown')
                test_df[col] = test_df[col].fillna('Unknown')
                
                # Fit on train, transform both
                le.fit(train_df[col])
                train_df[col] = le.transform(train_df[col])
                test_df[col] = le.transform(test_df[col])
                
                available_features.append(col)
            else:
                # Numerical features
                train_df[col] = train_df[col].fillna(train_df[col].median())
                test_df[col] = test_df[col].fillna(train_df[col].median())
                available_features.append(col)
    
    feature_columns = available_features
    
    print(f"Using {len(feature_columns)} features for modeling")
    
    return train_df, test_df, feature_columns

class RunnerAdvancementModel:
    """Main model class for runner advancement prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
        # Define models
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'HistGradientBoosting': HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=5000,
                class_weight='balanced',
                n_jobs=-1,
                solver='liblinear'
            )
        }
    
    def train(self, X, y, groups):
        """Train models with cross-validation"""
        print("\n=== TRAINING MODELS ===")
        
        results = {}
        gkf = GroupKFold(n_splits=5)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, groups=groups, cv=gkf, 
                                      scoring='neg_log_loss', n_jobs=-1)
            cv_score = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"{name} CV Log Loss: {cv_score:.4f} (+/- {cv_std:.4f})")
            
            # Train on full data
            model.fit(X, y)
            
            results[name] = {
                'model': model,
                'cv_score': cv_score,
                'cv_std': cv_std
            }
        
        # Find best model
        best_name = min(results.keys(), key=lambda x: results[x]['cv_score'])
        best_score = results[best_name]['cv_score']
        
        print(f"\nBest model: {best_name} with CV Log Loss: {best_score:.4f}")
        
        self.model = results[best_name]['model']
        
        return results, best_name, best_score
    
    def calibrate(self, X, y, groups, best_model, best_name, best_score):
        """Calibrate the best model"""
        print(f"\n=== CALIBRATING {best_name.upper()} ===")
        
        # Use CalibratedClassifierCV for better probability estimates
        calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
        calibrated_model.fit(X, y)
        
        # Cross-validation on calibrated model
        gkf = GroupKFold(n_splits=5)
        cv_scores = cross_val_score(calibrated_model, X, y, groups=groups, cv=gkf, 
                                  scoring='neg_log_loss', n_jobs=-1)
        calibrated_score = -cv_scores.mean()
        
        print(f"Calibrated {best_name} CV Log Loss: {calibrated_score:.4f}")
        
        if calibrated_score < best_score:
            print(f"Calibration improved performance by {best_score - calibrated_score:.4f}")
            self.model = calibrated_model
            final_score = calibrated_score
        else:
            print("Calibration did not improve performance, using original model")
            final_score = best_score
        
        return final_score
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict_proba(X)[:, 1]
    
    def set_features(self, feature_columns):
        """Set the feature columns for this model"""
        self.feature_columns = feature_columns

def run_analysis():
    """Run the complete analysis pipeline"""
    print("=== MARINERS RUNNER ADVANCEMENT PREDICTION ===")
    
    # Load data
    train_df, test_df, outfield_df = load_data()
    
    # Calculate baselines
    p, baseline_ll = calculate_baselines(train_df)
    
    # Feature engineering
    train_df, test_df = engineer_features(train_df, test_df)
    train_df, test_df = merge_outfield_tracking(train_df, test_df, outfield_df)
    train_df, test_df, feature_columns, label_encoders = prepare_features(train_df, test_df)
    
    # Train model
    model = RunnerAdvancementModel()
    model.set_features(feature_columns)
    
    X = train_df[feature_columns]
    y = train_df['runner_advance']
    groups = train_df['game_id'].values
    
    results, best_name, best_score = model.train(X, y, groups)
    best_model = results[best_name]['model']
    
    # Calibrate model
    calibrated_score = model.calibrate(X, y, groups, best_model, best_name, best_score)
    
    # Generate predictions
    print("\n=== MAKING FINAL PREDICTIONS ===")
    X_test = test_df[feature_columns]
    predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'play_id': test_df['play_id'],
        'runner_advance': predictions
    })
    
    # Save predictions
    submission.to_csv('runner_advance_predictions.csv', index=False)
    print(f"Predictions saved to 'runner_advance_predictions.csv'")
    print(f"Prediction statistics:")
    print(f"Mean probability: {predictions.mean():.3f}")
    print(f"Std probability: {predictions.std():.3f}")
    
    print("\n=== FINAL RESULTS ===")
    print(f"Constant baseline: {baseline_ll:.4f}")
    print(f"Final model performance: {calibrated_score:.4f}")
    improvement = baseline_ll - calibrated_score
    print(f"Improvement over baseline: {improvement:+.4f}")
    
    return submission, results, model

if __name__ == "__main__":
    submission, results, model = run_analysis()
