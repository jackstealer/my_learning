"""
Production-Ready Flask Application with Deep Learning
Heart Disease Prediction System
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime
import logging

# Add ml_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml_model'))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None

def load_predictor():
    """Load the ML model predictor"""
    global predictor
    try:
        from predict_transformer import HeartDiseasePredictor
        predictor = HeartDiseasePredictor()
        logger.info("✅ Data-driven model loaded successfully")
        logger.info("✅ All thresholds extracted from Heart_new2.csv (no hardcoding)")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.error("Run: python train_transformer.py")
        return False

# Try to load model on startup
model_loaded = load_predictor()

@app.route('/')
def index():
    """Home page"""
    return render_template('index_professional.html', model_loaded=model_loaded)

@app.route('/predict-page')
def predict_page():
    """Prediction form page"""
    if not model_loaded:
        return render_template('error.html', 
                             message="Model not loaded. Please train the model first.")
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for heart disease prediction"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train the model first'
        }), 503
    
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
            'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race',
            'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
            'Asthma', 'KidneyDisease', 'SkinCancer'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Convert numeric fields
        data['BMI'] = float(data['BMI'])
        data['PhysicalHealth'] = int(data['PhysicalHealth'])
        data['MentalHealth'] = int(data['MentalHealth'])
        data['SleepTime'] = int(data['SleepTime'])
        
        # Make prediction using data-driven model
        result = predictor.predict_single(data)
        
        # Generate detailed AI insights
        from llm_insights import LLMInsightsGenerator
        insights_generator = LLMInsightsGenerator()
        detailed_insights = insights_generator.generate_comprehensive_insights(data, result)
        
        # Add insights to result
        result['ai_insights'] = detailed_insights
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        # Log prediction
        logger.info(f"Prediction made: {result['prediction']}, Risk: {result['risk_level']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        data = request.get_json()
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({
                'error': 'No patient data provided'
            }), 400
        
        # Convert list of patients to predictions
        results = []
        for patient in patients:
            result = predictor.predict_single(patient)
            results.append(result)
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'model_used': 'Data-Driven System'
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    if not model_loaded:
        return jsonify({
            'loaded': False,
            'message': 'Model not loaded'
        })
    
    try:
        # Load metrics if available
        metrics_path = 'ml_model/saved_model/metrics.json'
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        # Get feature stats
        feature_stats = predictor.get_feature_stats()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        feature_stats = convert_to_json_serializable(feature_stats)
        
        return jsonify({
            'loaded': True,
            'features': list(predictor.get_feature_names()),
            'feature_stats': feature_stats,
            'metrics': metrics,
            'model_type': 'Data-Driven Intelligent System',
            'model_source': 'Trained on Heart_new2.csv',
            'framework': 'Statistical Analysis + Rule-Based'
        })
    except Exception as e:
        logger.error(f"Error in model_info: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page with Tableau"""
    tableau_config = {
        'server_url': 'https://public.tableau.com',
        'site_id': 'heart-disease-analysis',
        'dashboard_name': 'HeartDiseaseAnalysis'
    }
    return render_template('dashboard.html', config=tableau_config)

@app.route('/about')
def about():
    """About page"""
    return render_template('about_ml.html', model_loaded=model_loaded)

@app.route('/api/stats')
def get_stats():
    """API endpoint for dashboard statistics"""
    try:
        import pandas as pd
        df = pd.read_csv('Heart_new2.csv')
        
        # Calculate statistics
        total_records = len(df)
        disease_count = df['HeartDisease'].value_counts().get('Yes', 0)
        disease_rate = (disease_count / total_records * 100) if total_records > 0 else 0
        
        # Calculate high risk patients (those with multiple risk factors)
        high_risk = 0
        for _, row in df.iterrows():
            risk_count = 0
            if row.get('Stroke') == 'Yes':
                risk_count += 1
            if row.get('Diabetic') in ['Yes', 'Yes (during pregnancy)']:
                risk_count += 1
            if row.get('DiffWalking') == 'Yes':
                risk_count += 1
            if row.get('KidneyDisease') == 'Yes':
                risk_count += 1
            if row.get('GenHealth') in ['Poor', 'Fair']:
                risk_count += 1
            if risk_count >= 2:
                high_risk += 1
        
        # Get average age from age categories
        age_mapping = {
            '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
            '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
            '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
            '80 or older': 82
        }
        
        avg_age = None  # Will be computed from data
        if 'AgeCategory' in df.columns:
            ages = df['AgeCategory'].map(age_mapping).dropna()
            if len(ages) > 0:
                avg_age = ages.mean()
        
        return jsonify({
            'total_patients': int(total_records),
            'disease_prevalence': round(disease_rate, 1),
            'avg_age': round(avg_age, 1),
            'high_risk_patients': int(high_risk)
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'error': 'Failed to compute statistics from dataset',
            'message': str(e)
        }), 500

@app.route('/api/dataset-stats')
def dataset_stats():
    """Get dataset statistics"""
    try:
        import pandas as pd
        df = pd.read_csv('Heart_new2.csv')
        
        # Calculate statistics
        total_records = len(df)
        disease_count = df['HeartDisease'].value_counts().get('Yes', 0)
        no_disease_count = df['HeartDisease'].value_counts().get('No', 0)
        disease_rate = (disease_count / total_records * 100) if total_records > 0 else 0
        
        # Get feature count (excluding target)
        feature_count = len(df.columns) - 1
        
        # Calculate average age (if age data exists)
        avg_age = None
        if 'Age' in df.columns:
            avg_age = df['Age'].mean()
        
        return jsonify({
            'total_records': int(total_records),
            'disease_count': int(disease_count),
            'no_disease_count': int(no_disease_count),
            'disease_rate': round(disease_rate, 2),
            'feature_count': int(feature_count),
            'avg_age': round(avg_age, 1) if avg_age else None,
            'columns': df.columns.tolist()
        })
    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', 
                         message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', 
                         message="Internal server error"), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print("\n" + "="*60)
    print("🏥 HEART DISEASE PREDICTION SYSTEM")
    print("="*60)
    print(f"Model Status: {'✅ Loaded' if model_loaded else '❌ Not Loaded'}")
    print(f"Server: http://localhost:{port}")
    print(f"Debug Mode: {debug}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
