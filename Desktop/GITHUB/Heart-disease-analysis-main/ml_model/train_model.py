"""
Deep Learning Model for Heart Disease Prediction
Author: AI-Powered Healthcare System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import joblib
import json
import os

class HeartDiseaseModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the heart disease dataset"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        # Encode target variable
        y = y.map({'Yes': 1, 'No': 0})
        
        # Compute data-driven risk thresholds from actual disease rate
        disease_rate = float(y.mean())
        self._threshold_low    = round(disease_rate * 0.5, 4)
        self._threshold_medium = round(disease_rate,        4)
        self._threshold_high   = round(min(disease_rate * 2.0, 0.90), 4)
        print(f"Data-driven thresholds — Low: {self._threshold_low}, "
              f"Medium: {self._threshold_medium}, High: {self._threshold_high}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"Features: {self.feature_names}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def build_model(self, input_dim):
        """Build deep neural network model"""
        model = models.Sequential([
            # Input layer
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train(self, csv_path, epochs=100, batch_size=32):
        """Train the model"""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data(csv_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Calculate class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"\nClass weights: {class_weight_dict}")
        
        # Train model
        print("\nTraining model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test_scaled, y_test, verbose=0
        )
        
        # Predictions
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': float(test_accuracy),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("="*50)
        
        return history, metrics
    
    def save_model(self, model_dir='ml_model/saved_model'):
        """Save model and preprocessing objects"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Keras model
        self.model.save(os.path.join(model_dir, 'heart_disease_model.h5'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        
        # Save feature names
        with open(os.path.join(model_dir, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"\nModel saved to {model_dir}")
    
    def load_model(self, model_dir='ml_model/saved_model'):
        """Load trained model and preprocessing objects"""
        self.model = keras.models.load_model(os.path.join(model_dir, 'heart_disease_model.h5'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        
        with open(os.path.join(model_dir, 'feature_names.json'), 'r') as f:
            self.feature_names = json.load(f)
        
        print("Model loaded successfully")
    
    def predict(self, input_data):
        """Make prediction on new data"""
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col in self.label_encoders:
            if col in input_data.columns:
                input_data[col] = self.label_encoders[col].transform(input_data[col].astype(str))
        
        # Ensure correct feature order
        input_data = input_data[self.feature_names]
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        prediction_proba = self.model.predict(input_scaled, verbose=0)
        prediction = (prediction_proba > 0.5).astype(int)
        
        return {
            'prediction': int(prediction[0][0]),
            'probability': float(prediction_proba[0][0]),
            'risk_level': self._get_risk_level(prediction_proba[0][0])
        }
    
    def _get_risk_level(self, probability):
        """
        Determine risk level using data-driven thresholds.
        Thresholds are calibrated from the dataset's disease prevalence:
        - Low:       probability < 50% of disease rate
        - Medium:    probability < disease rate (baseline)
        - High:      probability < 2x disease rate
        - Very High: probability >= 2x disease rate
        
        Default fallback values approximate Heart_new2.csv (~10.7% disease rate).
        """
        low_t    = getattr(self, '_threshold_low',    0.05)
        medium_t = getattr(self, '_threshold_medium', 0.11)
        high_t   = getattr(self, '_threshold_high',   0.22)
        
        if probability < low_t:
            return 'Low'
        elif probability < medium_t:
            return 'Medium'
        elif probability < high_t:
            return 'High'
        else:
            return 'Very High'


if __name__ == "__main__":
    # Initialize model
    model = HeartDiseaseModel()
    
    # Train model
    history, metrics = model.train('Heart_new2.csv', epochs=100, batch_size=32)
    
    # Save model
    model.save_model()
    
    # Save metrics
    with open('ml_model/saved_model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n✅ Training complete! Model ready for deployment.")
