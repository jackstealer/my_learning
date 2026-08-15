"""Quick test of the prediction endpoint"""
import requests
import json

# Test data
test_patient = {
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
    "GenHealth": "Very good",
    "SleepTime": 7,
    "Asthma": "No",
    "KidneyDisease": "No",
    "SkinCancer": "No"
}

try:
    # Test prediction endpoint
    response = requests.post(
        'http://localhost:5000/api/predict',
        json=test_patient,
        timeout=10
    )
    
    print("Status Code:", response.status_code)
    print("\nResponse:")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    # Check which model is being used
    if 'model_used' in result:
        print(f"\n✅ Model Used: {result['model_used']}")
    
    if 'risk_factors' in result:
        print(f"\n✅ Using Data-Driven Transformer Model!")
        print(f"Risk Score: {result.get('risk_score', 'N/A')}")
        print(f"Risk Level: {result.get('risk_level', 'N/A')}")
    elif 'probability' in result:
        print(f"\n⚠️ Using OLD Ensemble Model (probability-based)")
        print(f"Probability: {result.get('probability', 'N/A')}")
    
except Exception as e:
    print(f"Error: {e}")
