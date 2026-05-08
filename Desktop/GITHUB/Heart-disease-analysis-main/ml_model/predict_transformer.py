"""
Prediction module using data-driven transformer model
"""

from transformers_model import HeartDiseaseTransformerModel

class HeartDiseasePredictor:
    def __init__(self, model_dir='ml_model/saved_model'):
        self.model = HeartDiseaseTransformerModel()
        self.model.load_model(model_dir)
    
    def predict_single(self, patient_data):
        """
        Predict heart disease for a single patient
        
        Args:
            patient_data (dict): Patient information
            
        Returns:
            dict: Prediction results with probability and risk level
        """
        return self.model.predict(patient_data)
    
    def predict_batch(self, patients_df):
        """
        Predict heart disease for multiple patients
        
        Args:
            patients_df (DataFrame): DataFrame with patient information
            
        Returns:
            list: List of prediction results
        """
        results = []
        for idx, row in patients_df.iterrows():
            result = self.model.predict(row.to_dict())
            results.append(result)
        return results
    
    def get_feature_names(self):
        """Get feature names for the model"""
        return self.model.feature_names
    
    def get_feature_stats(self):
        """Get feature statistics"""
        return self.model.feature_stats


# Example usage
if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    
    # Example patient data
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
    
    result = predictor.predict_single(sample_patient)
    print("\nPrediction Result:")
    print(f"Heart Disease: {'Yes' if result['prediction'] == 1 else 'No'}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Model: {result['model_used']}")
