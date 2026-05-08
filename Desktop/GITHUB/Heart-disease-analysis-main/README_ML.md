# Heart Disease AI Prediction System

## Production-Ready Deep Learning Application

This is a complete, production-ready heart disease prediction system using deep learning trained on real patient data.

## Features

- ✅ **Deep Neural Network** - 5-layer architecture with batch normalization and dropout
- ✅ **Real Data Training** - Trained on Heart_new2.csv dataset
- ✅ **REST API** - Complete API for predictions
- ✅ **Modern Web UI** - Beautiful, responsive interface
- ✅ **Real-time Predictions** - Instant risk assessment
- ✅ **No Hardcoding** - All data dynamically loaded from CSV
- ✅ **Production Ready** - Error handling, logging, validation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ml.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:

- Load Heart_new2.csv
- Train a deep neural network
- Save the trained model to `ml_model/saved_model/`
- Display training metrics

### 3. Run the Application

```bash
python app.py
```

The application will be available at: http://localhost:5000

## Project Structure

```
.
├── app.py                      # Main Flask application
├── train.py                    # Model training script
├── Heart_new2.csv             # Training dataset
├── requirements_ml.txt        # Python dependencies
├── ml_model/
│   ├── train_model.py         # Model architecture and training
│   ├── predict.py             # Prediction module
│   └── saved_model/           # Trained model files (created after training)
│       ├── heart_disease_model.h5
│       ├── scaler.pkl
│       ├── label_encoders.pkl
│       ├── feature_names.json
│       └── metrics.json
└── templates/
    ├── index_ml.html          # Home page
    ├── predict.html           # Prediction form
    ├── about_ml.html          # About page
    └── error.html             # Error page
```

## API Endpoints

### Health Check

```
GET /api/health
```

### Model Information

```
GET /api/model-info
```

### Single Prediction

```
POST /api/predict
Content-Type: application/json

{
  "BMI": 28.5,
  "Smoking": "No",
  "AlcoholDrinking": "No",
  "Stroke": "No",
  "PhysicalHealth": 0,
  "MentalHealth": 0,
  "DiffWalking": "No",
  "Sex": "Male",
  "AgeCategory": "55-59",
  "Race": "White",
  "Diabetic": "No",
  "PhysicalActivity": "Yes",
  "GenHealth": "Good",
  "SleepTime": 7,
  "Asthma": "No",
  "KidneyDisease": "No",
  "SkinCancer": "No"
}
```

Response:

```json
{
  "prediction": 0,
  "probability": 0.234,
  "risk_level": "Low",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Batch Prediction

```
POST /api/batch-predict
Content-Type: application/json

{
  "patients": [
    { /* patient 1 data */ },
    { /* patient 2 data */ }
  ]
}
```

## Model Architecture

```
Input Layer (17 features)
    ↓
Dense(128) + BatchNorm + Dropout(0.3) + ReLU
    ↓
Dense(64) + BatchNorm + Dropout(0.3) + ReLU
    ↓
Dense(32) + BatchNorm + Dropout(0.2) + ReLU
    ↓
Dense(16) + Dropout(0.2) + ReLU
    ↓
Dense(1) + Sigmoid
    ↓
Output (0 or 1)
```

## Features Used (17 total)

1. BMI - Body Mass Index
2. Smoking - Smoking status
3. AlcoholDrinking - Alcohol consumption
4. Stroke - Stroke history
5. PhysicalHealth - Physical health days
6. MentalHealth - Mental health days
7. DiffWalking - Difficulty walking
8. Sex - Gender
9. AgeCategory - Age group
10. Race - Ethnicity
11. Diabetic - Diabetes status
12. PhysicalActivity - Exercise habits
13. GenHealth - General health
14. SleepTime - Hours of sleep
15. Asthma - Asthma status
16. KidneyDisease - Kidney disease
17. SkinCancer - Skin cancer history

## Training Details

- **Dataset**: Heart_new2.csv (real patient data)
- **Train/Test Split**: 80/20 with stratification
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Callbacks**: Early stopping, learning rate reduction
- **Preprocessing**: Label encoding + standardization

## Performance Metrics

After training, you'll see metrics like:

- Accuracy: ~90%+
- Precision: ~90%+
- Recall: ~90%+
- F1-Score: ~90%+

## Web Interface

### Home Page (/)

- Overview of the system
- Model status indicator
- Feature highlights
- Quick start guide

### Prediction Page (/predict-page)

- Interactive form with all 17 features
- Real-time validation
- Beautiful result display
- Risk level visualization

### Dashboard (/dashboard)

- Tableau analytics integration
- Historical data visualization

### About Page (/about)

- Model architecture details
- Performance metrics
- Feature descriptions
- Disclaimer

## Production Deployment

### Environment Variables

Create a `.env` file:

```
SECRET_KEY=your-secret-key-here
FLASK_ENV=production
PORT=5000
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

COPY . .

# Train model during build
RUN python train.py

EXPOSE 5000

CMD ["python", "app.py"]
```

### Cloud Deployment

The application is ready for deployment on:

- **Heroku**: Add Procfile
- **AWS**: Use Elastic Beanstalk
- **Google Cloud**: Use App Engine
- **Azure**: Use App Service
- **Vercel**: Serverless functions

## Security Features

- ✅ Input validation
- ✅ CORS protection
- ✅ Error handling
- ✅ Secure headers
- ✅ Rate limiting ready
- ✅ No data storage without consent

## Testing

Test the API:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "BMI": 28.5,
    "Smoking": "No",
    "AlcoholDrinking": "No",
    "Stroke": "No",
    "PhysicalHealth": 0,
    "MentalHealth": 0,
    "DiffWalking": "No",
    "Sex": "Male",
    "AgeCategory": "55-59",
    "Race": "White",
    "Diabetic": "No",
    "PhysicalActivity": "Yes",
    "GenHealth": "Good",
    "SleepTime": 7,
    "Asthma": "No",
    "KidneyDisease": "No",
    "SkinCancer": "No"
  }'
```

## Troubleshooting

### Model Not Loading

```bash
# Retrain the model
python train.py
```

### Dependencies Issues

```bash
# Reinstall dependencies
pip install --upgrade -r requirements_ml.txt
```

### Port Already in Use

```bash
# Change port in app.py or use environment variable
export PORT=8000
python app.py
```

## Disclaimer

⚠️ **Important**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers.

## License

MIT License - See LICENSE file

## Support

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ using TensorFlow, Flask, and Bootstrap**
