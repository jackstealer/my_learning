"""
Test all API endpoints to ensure they're working correctly
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_endpoint(name, method, url, data=None, expected_status=200):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"Method: {method}")
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == expected_status:
            print("✅ PASS")
            if response.headers.get('Content-Type', '').startswith('application/json'):
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)[:500]}")
            return True
        else:
            print(f"❌ FAIL - Expected {expected_status}, got {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("HEART DISEASE AI - API ENDPOINT TESTING")
    print("="*60)
    
    results = []
    
    # Test 1: Health Check
    results.append(test_endpoint(
        "Health Check",
        "GET",
        f"{BASE_URL}/api/health"
    ))
    
    # Test 2: Model Info
    results.append(test_endpoint(
        "Model Info",
        "GET",
        f"{BASE_URL}/api/model-info"
    ))
    
    # Test 3: Dataset Stats
    results.append(test_endpoint(
        "Dataset Stats",
        "GET",
        f"{BASE_URL}/api/dataset-stats"
    ))
    
    # Test 4: Dashboard Stats
    results.append(test_endpoint(
        "Dashboard Stats",
        "GET",
        f"{BASE_URL}/api/stats"
    ))
    
    # Test 5: Single Prediction - Low Risk
    low_risk_patient = {
        'BMI': 22.5,
        'Smoking': 'No',
        'AlcoholDrinking': 'No',
        'Stroke': 'No',
        'PhysicalHealth': 0,
        'MentalHealth': 0,
        'DiffWalking': 'No',
        'Sex': 'Female',
        'AgeCategory': '25-29',
        'Race': 'White',
        'Diabetic': 'No',
        'PhysicalActivity': 'Yes',
        'GenHealth': 'Excellent',
        'SleepTime': 8,
        'Asthma': 'No',
        'KidneyDisease': 'No',
        'SkinCancer': 'No'
    }
    results.append(test_endpoint(
        "Single Prediction - Low Risk",
        "POST",
        f"{BASE_URL}/api/predict",
        low_risk_patient
    ))
    
    # Test 6: Single Prediction - High Risk
    high_risk_patient = {
        'BMI': 35.0,
        'Smoking': 'Yes',
        'AlcoholDrinking': 'Yes',
        'Stroke': 'Yes',
        'PhysicalHealth': 20,
        'MentalHealth': 15,
        'DiffWalking': 'Yes',
        'Sex': 'Male',
        'AgeCategory': '70-74',
        'Race': 'White',
        'Diabetic': 'Yes',
        'PhysicalActivity': 'No',
        'GenHealth': 'Poor',
        'SleepTime': 4,
        'Asthma': 'Yes',
        'KidneyDisease': 'Yes',
        'SkinCancer': 'Yes'
    }
    results.append(test_endpoint(
        "Single Prediction - High Risk",
        "POST",
        f"{BASE_URL}/api/predict",
        high_risk_patient
    ))
    
    # Test 7: Batch Prediction
    batch_data = {
        'patients': [low_risk_patient, high_risk_patient]
    }
    results.append(test_endpoint(
        "Batch Prediction",
        "POST",
        f"{BASE_URL}/api/batch-predict",
        batch_data
    ))
    
    # Test 8: Missing Fields (should fail with 400)
    incomplete_patient = {
        'BMI': 28.5,
        'Smoking': 'No'
        # Missing other required fields
    }
    results.append(test_endpoint(
        "Missing Fields (Expected 400)",
        "POST",
        f"{BASE_URL}/api/predict",
        incomplete_patient,
        expected_status=400
    ))
    
    # Test 9: Home Page
    results.append(test_endpoint(
        "Home Page",
        "GET",
        f"{BASE_URL}/"
    ))
    
    # Test 10: Prediction Page
    results.append(test_endpoint(
        "Prediction Page",
        "GET",
        f"{BASE_URL}/predict-page"
    ))
    
    # Test 11: Dashboard Page
    results.append(test_endpoint(
        "Dashboard Page",
        "GET",
        f"{BASE_URL}/dashboard"
    ))
    
    # Test 12: About Page
    results.append(test_endpoint(
        "About Page",
        "GET",
        f"{BASE_URL}/about"
    ))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
