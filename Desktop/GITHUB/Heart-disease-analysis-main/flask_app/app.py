"""
Flask Web Application for Heart Disease Dashboard
Production-Ready Version with Database Integration
"""

from flask import Flask, render_template, jsonify, request, abort
from flask_cors import CORS
import os
import logging
from logging.handlers import RotatingFileHandler
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from functools import wraps
import secrets

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_NAME = os.getenv('DB_NAME', 'heart_disease_db')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    TABLEAU_SERVER_URL = os.getenv('TABLEAU_SERVER_URL', 'https://public.tableau.com')
    TABLEAU_SITE_ID = os.getenv('TABLEAU_SITE_ID', 'heart-disease-analysis')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    JSON_SORT_KEYS = False

app.config.from_object(Config)

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240000, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Heart Disease Analysis startup')

# Database connection pool
def get_db_connection():
    """Create and return a database connection"""
    try:
        connection = mysql.connector.connect(
            host=app.config['DB_HOST'],
            port=app.config['DB_PORT'],
            database=app.config['DB_NAME'],
            user=app.config['DB_USER'],
            password=app.config['DB_PASSWORD'],
            pool_name='heart_disease_pool',
            pool_size=5,
            pool_reset_session=True
        )
        return connection
    except Error as e:
        app.logger.error(f'Database connection error: {e}')
        return None

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Resource not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f'Server Error: {error}')
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    """Handle 403 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Forbidden'}), 403
    return render_template('403.html'), 403

# Rate limiting decorator (simple implementation)
from collections import defaultdict
from datetime import datetime, timedelta

request_counts = defaultdict(list)

def rate_limit(max_requests=100, window=60):
    """Simple rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            now = datetime.now()
            
            # Clean old requests
            request_counts[client_ip] = [
                req_time for req_time in request_counts[client_ip]
                if now - req_time < timedelta(seconds=window)
            ]
            
            # Check rate limit
            if len(request_counts[client_ip]) >= max_requests:
                app.logger.warning(f'Rate limit exceeded for {client_ip}')
                abort(429)
            
            request_counts[client_ip].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Routes
@app.route('/')
def index():
    """Home page with dashboard overview"""
    app.logger.info('Home page accessed')
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard page with embedded Tableau"""
    tableau_config = {
        'server_url': app.config['TABLEAU_SERVER_URL'],
        'site_id': app.config['TABLEAU_SITE_ID'],
        'dashboard_name': 'HeartDiseaseAnalysis'
    }
    app.logger.info('Dashboard page accessed')
    return render_template('dashboard.html', config=tableau_config)

@app.route('/story')
def story():
    """Story page with embedded Tableau Story"""
    tableau_config = {
        'server_url': app.config['TABLEAU_SERVER_URL'],
        'site_id': app.config['TABLEAU_SITE_ID'],
        'story_name': 'HeartDiseaseStory'
    }
    app.logger.info('Story page accessed')
    return render_template('story.html', config=tableau_config)

@app.route('/about')
def about():
    """About page with project information"""
    app.logger.info('About page accessed')
    return render_template('about.html')

# API Routes
@app.route('/api/stats')
@rate_limit(max_requests=100, window=60)
def get_stats():
    """API endpoint for dashboard statistics"""
    try:
        connection = get_db_connection()
        if not connection:
            # Fallback: compute stats from CSV when DB is unavailable
            app.logger.warning('Database not available, computing stats from CSV')
            try:
                import pandas as pd
                csv_path = os.path.join(os.path.dirname(__file__), '..', 'Heart_new2.csv')
                df = pd.read_csv(csv_path)
                total = len(df)
                disease_count = df['HeartDisease'].value_counts().get('Yes', 0)
                disease_prevalence = round((disease_count / total * 100), 2) if total > 0 else 0
                # Age mapping for average age
                age_mapping = {
                    '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
                    '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
                    '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
                    '80 or older': 82
                }
                avg_age = None
                if 'AgeCategory' in df.columns:
                    ages = df['AgeCategory'].map(age_mapping).dropna()
                    if len(ages) > 0:
                        avg_age = round(float(ages.mean()), 1)
                # High risk: 2+ major risk factors
                high_risk = 0
                for _, row in df.iterrows():
                    risk_count = sum([
                        row.get('Stroke') == 'Yes',
                        row.get('Diabetic') in ['Yes', 'Yes (during pregnancy)'],
                        row.get('DiffWalking') == 'Yes',
                        row.get('KidneyDisease') == 'Yes',
                        row.get('GenHealth') in ['Poor', 'Fair']
                    ])
                    if risk_count >= 2:
                        high_risk += 1
                return jsonify({
                    'total_patients': total,
                    'disease_prevalence': disease_prevalence,
                    'avg_age': avg_age,
                    'high_risk_patients': high_risk,
                    'data_source': 'csv_fallback'
                })
            except Exception as csv_err:
                app.logger.error(f'CSV fallback also failed: {csv_err}')
                return jsonify({'error': 'No data source available'}), 503
        
        cursor = connection.cursor(dictionary=True)
        
        # Get total patients
        cursor.execute("SELECT COUNT(*) as total FROM patients")
        total = cursor.fetchone()['total']
        
        # Get disease prevalence
        cursor.execute("""
            SELECT 
                ROUND(SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as prevalence
            FROM patients
        """)
        prevalence = cursor.fetchone()['prevalence']
        
        # Get average age
        cursor.execute("SELECT ROUND(AVG(age), 1) as avg_age FROM patients")
        avg_age = cursor.fetchone()['avg_age']
        
        # Get high risk patients (cholesterol > 240 AND resting_bp > 140)
        cursor.execute("""
            SELECT COUNT(*) as high_risk 
            FROM patients 
            WHERE cholesterol > 240 AND resting_bp > 140
        """)
        high_risk = cursor.fetchone()['high_risk']
        
        cursor.close()
        connection.close()
        
        stats = {
            'total_patients': total,
            'disease_prevalence': float(prevalence) if prevalence else 0,
            'avg_age': float(avg_age) if avg_age else 0,
            'high_risk_patients': high_risk,
            'data_source': 'database'
        }
        
        app.logger.info('Stats retrieved successfully from database')
        return jsonify(stats)
        
    except Exception as e:
        app.logger.error(f'Error fetching stats: {e}')
        return jsonify({'error': 'Failed to fetch statistics'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        connection = get_db_connection()
        db_status = 'connected' if connection else 'disconnected'
        if connection:
            connection.close()
        
        return jsonify({
            'status': 'healthy',
            'database': db_status,
            'version': '1.0.0'
        })
    except Exception as e:
        app.logger.error(f'Health check failed: {e}')
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/patients/demographics')
@rate_limit(max_requests=50, window=60)
def get_demographics():
    """Get patient demographics data"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database not available'}), 503
        
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                gender,
                COUNT(*) as count,
                ROUND(AVG(age), 1) as avg_age,
                SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) as disease_count
            FROM patients
            GROUP BY gender
        """)
        
        demographics = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return jsonify(demographics)
        
    except Exception as e:
        app.logger.error(f'Error fetching demographics: {e}')
        return jsonify({'error': 'Failed to fetch demographics'}), 500

# Context processors
@app.context_processor
def inject_config():
    """Inject configuration into templates"""
    return {
        'app_name': 'Heart Disease Analysis',
        'version': '1.0.0'
    }

# Before request handler
@app.before_request
def before_request():
    """Log all requests"""
    app.logger.info(f'{request.method} {request.path} from {request.remote_addr}')

if __name__ == '__main__':
    # Production mode check
    is_production = os.getenv('FLASK_ENV') == 'production'
    
    if is_production:
        app.logger.info('Starting in PRODUCTION mode')
        # Use a production WSGI server like gunicorn in production
        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
    else:
        app.logger.info('Starting in DEVELOPMENT mode')
        app.run(host='0.0.0.0', port=5000, debug=True)
