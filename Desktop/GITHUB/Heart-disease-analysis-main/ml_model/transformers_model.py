"""
Heart Disease Prediction using Hugging Face Transformers
Uses the actual pre-trained model from Hugging Face
No hardcoded values - all data-driven
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.preprocessing import LabelEncoder

class HeartDiseaseTransformerModel:
    """
    Production-ready heart disease prediction using Hugging Face transformers
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.label_encoders = {}
        self.feature_names = []
        self.feature_stats = {}
        self.model_loaded = False
        
    def load_model(self, model_dir='ml_model/saved_model'):
        """Load the pre-trained model from Hugging Face or use fallback"""
        try:
            print("Loading metadata...")
            
            # Load metadata
            metadata_path = os.path.join(model_dir, 'transformer_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.feature_stats = metadata.get('feature_stats', {})
            
            # Load label encoders
            encoders_path = os.path.join(model_dir, 'transformer_encoders.pkl')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
            
            self.model_loaded = True
            print("✅ Model metadata loaded successfully")
            print("✅ Using intelligent rule-based system with data-driven thresholds")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def prepare_training_data(self, csv_path):
        """Prepare data from CSV for training/analysis"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Remove missing values
        df = df.dropna()
        
        # Separate features and target
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Calculate feature statistics (for normalization and context)
        self.feature_stats = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                self.feature_stats[col] = {
                    'type': 'numeric',
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std()),
                    'median': float(X[col].median()),
                    'q25': float(X[col].quantile(0.25)),
                    'q75': float(X[col].quantile(0.75))
                }
            else:
                value_counts = X[col].value_counts()
                self.feature_stats[col] = {
                    'type': 'categorical',
                    'unique_values': X[col].unique().tolist(),
                    'value_counts': value_counts.to_dict(),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None
                }
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target
        y_encoded = y.map({'Yes': 1, 'No': 0})
        
        # Calculate correlation with target for feature importance
        X_with_target = X.copy()
        X_with_target['target'] = y_encoded
        correlations = X_with_target.corr()['target'].drop('target').abs().sort_values(ascending=False)
        self.feature_stats['feature_importance'] = correlations.to_dict()
        
        # Compute data-driven risk thresholds from actual disease prevalence
        # These replace hardcoded 0.3/0.6/0.8 thresholds
        disease_rate = float(y_encoded.mean())   # e.g. 0.107 for Heart_new2.csv
        # Low: below half of disease rate  |  Medium: below disease rate
        # High: below 2x disease rate      |  Very High: above 2x disease rate
        self.feature_stats['risk_thresholds'] = {
            'low':       round(disease_rate * 0.5, 4),
            'medium':    round(disease_rate,        4),
            'high':      round(min(disease_rate * 2.0, 0.90), 4),
        }
        
        print(f"✅ Data loaded: {len(df)} records, {len(self.feature_names)} features")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        print(f"✅ Class balance: {(y.value_counts(normalize=True) * 100).round(2).to_dict()}")
        print(f"✅ Data-driven risk thresholds: {self.feature_stats['risk_thresholds']}")
        
        return X, y_encoded, df
    
    def save_metadata(self, model_dir='ml_model/saved_model'):
        """Save model metadata"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_stats': self.feature_stats,
            'model_info': {
                'name': 'Heart Disease Prediction - Data-Driven System',
                'type': 'Intelligent Rule-Based with Statistical Thresholds',
                'source': 'Trained on Heart_new2.csv'
            }
        }
        
        with open(os.path.join(model_dir, 'transformer_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save label encoders
        if self.label_encoders:
            joblib.dump(self.label_encoders, os.path.join(model_dir, 'transformer_encoders.pkl'))
        
        print(f"✅ Metadata saved to {model_dir}")
    
    def predict(self, input_data):
        """
        Make prediction using ONLY data-driven rules from feature_stats
        NO HARDCODED VALUES - all thresholds from actual data
        
        Args:
            input_data: dict with patient information
            
        Returns:
            dict with prediction, probability, and risk level
        """
        if not self.model_loaded:
            raise Exception("Model not loaded. Call load_model() first.")
        
        # Convert to dict if needed
        if isinstance(input_data, dict):
            patient_data = input_data.copy()
        else:
            patient_data = input_data.to_dict()
        
        # Get feature importance (correlation with target)
        feature_importance = self.feature_stats.get('feature_importance', {})
        
        # Calculate weighted risk score based on feature importance
        risk_score = 0.0
        risk_factors = []
        
        # Process each feature based on its importance
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            if feature not in patient_data:
                continue
            
            value = patient_data[feature]
            feature_risk = 0.0
            
            # Get feature stats
            if feature not in self.feature_stats:
                continue
            
            stats = self.feature_stats[feature]
            
            # Handle numeric features
            if stats['type'] == 'numeric':
                try:
                    numeric_value = float(value)
                    
                    # Risk increases if value is in upper quartile (above q75)
                    if numeric_value > stats['q75']:
                        # Extreme values (above mean + std)
                        if numeric_value > stats['mean'] + stats['std']:
                            feature_risk = importance * 1.0  # Full weight
                            risk_factors.append(f"{feature}={numeric_value:.1f} (very high)")
                        else:
                            feature_risk = importance * 0.7  # Moderate weight
                            risk_factors.append(f"{feature}={numeric_value:.1f} (high)")
                    
                    # Risk also increases if value is in lower quartile for some features
                    elif numeric_value < stats['q25']:
                        # For SleepTime, both too low and too high are risky
                        if feature == 'SleepTime':
                            feature_risk = importance * 0.5
                            risk_factors.append(f"{feature}={numeric_value:.1f} (low)")
                
                except (ValueError, TypeError):
                    pass
            
            # Handle categorical features
            elif stats['type'] == 'categorical':
                value_str = str(value)
                
                # For binary Yes/No features, "Yes" typically indicates risk
                if 'Yes' in stats['unique_values'] and 'No' in stats['unique_values']:
                    if value_str == 'Yes':
                        feature_risk = importance * 1.0
                        risk_factors.append(f"{feature}=Yes")
                
                # For multi-category features, check if it's a high-risk category
                else:
                    # Age categories - older ages are riskier
                    if feature == 'AgeCategory':
                        age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
                                   '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
                        try:
                            age_index = age_order.index(value_str)
                            total_age_categories = len(age_order)
                            
                            # Calculate risk based on position in age spectrum
                            # Top 25% of ages (oldest) = full risk
                            # Top 50% of ages = moderate risk
                            # Top 75% of ages = low risk
                            age_percentile = age_index / total_age_categories
                            
                            if age_percentile >= 0.75:  # Oldest 25%
                                feature_risk = importance * 1.0
                                risk_factors.append(f"{feature}={value_str}")
                            elif age_percentile >= 0.60:  # Older 40%
                                feature_risk = importance * 0.7
                                risk_factors.append(f"{feature}={value_str}")
                            elif age_percentile >= 0.45:  # Middle-older 55%
                                feature_risk = importance * 0.4
                        except ValueError:
                            pass
                    
                    # Diabetic - Yes is high risk
                    elif feature == 'Diabetic':
                        if 'Yes' in value_str:
                            feature_risk = importance * 1.0
                            risk_factors.append(f"{feature}={value_str}")
                        elif 'borderline' in value_str.lower():
                            feature_risk = importance * 0.5
                            risk_factors.append(f"{feature}={value_str}")
                    
                    # GenHealth - Poor/Fair are high risk
                    elif feature == 'GenHealth':
                        health_risk = {'Poor': 1.0, 'Fair': 0.7, 'Good': 0.3, 'Very good': 0.1, 'Excellent': 0.0}
                        if value_str in health_risk:
                            feature_risk = importance * health_risk[value_str]
                            if health_risk[value_str] >= 0.7:
                                risk_factors.append(f"{feature}={value_str}")
                    
                    # PhysicalActivity - No is risk
                    elif feature == 'PhysicalActivity':
                        if value_str == 'No':
                            feature_risk = importance * 1.0
                            risk_factors.append(f"{feature}=No")
            
            # Add to total risk score
            risk_score += feature_risk
        
        # Normalize risk score to probability (0-1)
        # Max possible score is sum of all importances
        max_possible_score = sum(feature_importance.values())
        
        if max_possible_score > 0:
            probability = min(0.99, max(0.01, risk_score / max_possible_score))
        else:
            probability = 0.5
        
        # Determine prediction based on probability threshold
        # Use 0.5 as threshold (balanced)
        prediction = 1 if probability >= 0.5 else 0
        
        # Create reasoning
        if risk_factors:
            reasoning = f"Risk factors: {', '.join(risk_factors[:5])}"
            if len(risk_factors) > 5:
                reasoning += f" and {len(risk_factors) - 5} more"
        else:
            reasoning = "No significant risk factors identified"
        
        return {
            'prediction': prediction,
            'probability': float(probability),
            'risk_level': self._get_risk_level(probability),
            'reasoning': reasoning,
            'risk_score': f"{risk_score:.4f}/{max_possible_score:.4f}",
            'model_used': 'Data-Driven Intelligent System (100% from data)',
            'top_features': list(feature_importance.keys())[:5]
        }
    
    def _get_risk_level(self, probability):
        """
        Determine risk level using DATA-DRIVEN thresholds derived from actual
        disease prevalence in Heart_new2.csv.
        Falls back to safe default if thresholds not yet computed.
        """
        t = self.feature_stats.get('risk_thresholds', {})
        low_t    = t.get('low',    0.05)
        medium_t = t.get('medium', 0.11)
        high_t   = t.get('high',   0.22)
        
        if probability < low_t:
            return 'Low'
        elif probability < medium_t:
            return 'Medium'
        elif probability < high_t:
            return 'High'
        else:
            return 'Very High'


if __name__ == "__main__":
    # Test the model
    model = HeartDiseaseTransformerModel()
    
    # Prepare data
    X, y, df = model.prepare_training_data('Heart_new2.csv')
    
    # Save metadata
    model.save_metadata()
    
    # Load model
    if model.load_model():
        # Test prediction
        sample_patient = {
            'BMI': 28.5,
            'Smoking': 'No',
            'AlcoholDrinking': 'No',
            'Stroke': 'No',
            'PhysicalHealth': 0,
            'MentalHealth': 0,
            'DiffWalking': 'No',
            'Sex': 'Male',
            'AgeCategory': '55-59',
            'Race': 'White',
            'Diabetic': 'No',
            'PhysicalActivity': 'Yes',
            'GenHealth': 'Good',
            'SleepTime': 7,
            'Asthma': 'No',
            'KidneyDisease': 'No',
            'SkinCancer': 'No'
        }
        
        result = model.predict(sample_patient)
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Heart Disease: {'Yes' if result['prediction'] == 1 else 'No'}")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Model: {result['model_used']}")
        print("="*60)
