"""
LLM-powered AI Insights Generator
Provides detailed, personalized health insights and recommendations
"""

import json
from typing import Dict, List

class LLMInsightsGenerator:
    """Generate detailed AI insights for cardiovascular risk assessment"""
    
    def __init__(self):
        self.risk_level_descriptions = {
            'Low': 'minimal cardiovascular risk',
            'Medium': 'moderate cardiovascular risk',
            'High': 'elevated cardiovascular risk',
            'Very High': 'significantly elevated cardiovascular risk'
        }
    
    def generate_comprehensive_insights(self, patient_data: Dict, prediction_result: Dict) -> Dict:
        """
        Generate comprehensive AI-powered insights
        
        Args:
            patient_data: Patient information
            prediction_result: Model prediction results
            
        Returns:
            Dictionary with detailed insights
        """
        insights = {
            'executive_summary': self._generate_executive_summary(patient_data, prediction_result),
            'risk_analysis': self._generate_risk_analysis(patient_data, prediction_result),
            'lifestyle_recommendations': self._generate_lifestyle_recommendations(patient_data),
            'medical_considerations': self._generate_medical_considerations(patient_data),
            'preventive_measures': self._generate_preventive_measures(patient_data, prediction_result),
            'follow_up_recommendations': self._generate_follow_up_recommendations(prediction_result),
            'educational_content': self._generate_educational_content(patient_data)
        }
        
        return insights
    
    def _generate_executive_summary(self, patient_data: Dict, prediction_result: Dict) -> str:
        """Generate executive summary of the assessment"""
        risk_level = prediction_result.get('risk_level', 'Unknown')
        probability = prediction_result.get('probability', 0) * 100
        
        summary = f"""Based on comprehensive analysis of {len(patient_data)} clinical and lifestyle factors, 
this assessment indicates {self.risk_level_descriptions.get(risk_level, 'unknown')} with a probability 
of {probability:.1f}%. """
        
        if risk_level in ['High', 'Very High']:
            summary += """This elevated risk profile warrants immediate attention and consultation with 
a healthcare provider to develop a comprehensive cardiovascular risk management plan."""
        elif risk_level == 'Medium':
            summary += """While not immediately critical, this moderate risk level suggests the importance 
of lifestyle modifications and regular monitoring to prevent progression."""
        else:
            summary += """This favorable risk profile indicates good cardiovascular health, though 
continued healthy lifestyle practices remain important for long-term wellness."""
        
        return summary
    
    def _generate_risk_analysis(self, patient_data: Dict, prediction_result: Dict) -> Dict:
        """Generate detailed risk factor analysis"""
        analysis = {
            'primary_risk_factors': [],
            'modifiable_factors': [],
            'non_modifiable_factors': [],
            'protective_factors': []
        }
        
        # Analyze age
        age_cat = patient_data.get('AgeCategory', '')
        if any(x in age_cat for x in ['70', '75', '80']):
            analysis['primary_risk_factors'].append({
                'factor': 'Advanced Age',
                'value': age_cat,
                'impact': 'High',
                'explanation': 'Age is a significant non-modifiable risk factor. Cardiovascular risk increases substantially with age due to cumulative exposure to risk factors and age-related changes in blood vessels.'
            })
            analysis['non_modifiable_factors'].append('Age')
        
        # Analyze smoking
        if patient_data.get('Smoking') == 'Yes':
            analysis['primary_risk_factors'].append({
                'factor': 'Smoking',
                'value': 'Current smoker',
                'impact': 'Very High',
                'explanation': 'Smoking is one of the most significant modifiable risk factors for cardiovascular disease. It damages blood vessel walls, increases blood pressure, reduces oxygen in blood, and promotes blood clot formation.'
            })
            analysis['modifiable_factors'].append('Smoking')
        else:
            analysis['protective_factors'].append({
                'factor': 'Non-smoker',
                'benefit': 'Significantly reduces cardiovascular risk compared to smokers'
            })
        
        # Analyze diabetes
        diabetic_status = patient_data.get('Diabetic', 'No')
        if 'Yes' in diabetic_status:
            analysis['primary_risk_factors'].append({
                'factor': 'Diabetes',
                'value': diabetic_status,
                'impact': 'High',
                'explanation': 'Diabetes significantly increases cardiovascular risk by damaging blood vessels, promoting atherosclerosis, and affecting multiple organ systems. Proper diabetes management is crucial for cardiovascular health.'
            })
            analysis['modifiable_factors'].append('Diabetes Management')
        
        # Analyze physical activity
        if patient_data.get('PhysicalActivity') == 'No':
            analysis['primary_risk_factors'].append({
                'factor': 'Physical Inactivity',
                'value': 'No regular physical activity',
                'impact': 'Moderate',
                'explanation': 'Lack of physical activity is a major modifiable risk factor. Regular exercise strengthens the heart, improves circulation, helps control weight, and reduces other risk factors.'
            })
            analysis['modifiable_factors'].append('Physical Activity')
        else:
            analysis['protective_factors'].append({
                'factor': 'Regular Physical Activity',
                'benefit': 'Helps maintain healthy weight, blood pressure, and cholesterol levels'
            })
        
        # Analyze BMI
        bmi = float(patient_data.get('BMI', 25))
        if bmi > 30:
            analysis['primary_risk_factors'].append({
                'factor': 'Obesity',
                'value': f'BMI: {bmi:.1f}',
                'impact': 'Moderate to High',
                'explanation': 'Obesity increases cardiovascular risk by promoting high blood pressure, diabetes, high cholesterol, and inflammation. Weight management through diet and exercise is crucial.'
            })
            analysis['modifiable_factors'].append('Weight Management')
        elif bmi > 25:
            analysis['modifiable_factors'].append('Weight Optimization')
        
        # Analyze stroke history
        if patient_data.get('Stroke') == 'Yes':
            analysis['primary_risk_factors'].append({
                'factor': 'Previous Stroke',
                'value': 'History of stroke',
                'impact': 'Very High',
                'explanation': 'Previous stroke significantly increases risk of future cardiovascular events. Aggressive risk factor management and preventive therapy are essential.'
            })
        
        # Analyze difficulty walking
        if patient_data.get('DiffWalking') == 'Yes':
            analysis['primary_risk_factors'].append({
                'factor': 'Mobility Limitations',
                'value': 'Difficulty walking or climbing stairs',
                'impact': 'Moderate',
                'explanation': 'Mobility limitations may indicate underlying cardiovascular issues, reduced physical activity, or other health conditions that increase cardiovascular risk.'
            })
        
        # Analyze kidney disease
        if patient_data.get('KidneyDisease') == 'Yes':
            analysis['primary_risk_factors'].append({
                'factor': 'Chronic Kidney Disease',
                'value': 'History of kidney disease',
                'impact': 'High',
                'explanation': 'Kidney disease and cardiovascular disease are closely linked. Kidney dysfunction affects blood pressure, fluid balance, and increases inflammation and cardiovascular risk.'
            })
        
        # Analyze general health
        gen_health = patient_data.get('GenHealth', 'Good')
        if gen_health in ['Poor', 'Fair']:
            analysis['primary_risk_factors'].append({
                'factor': 'Poor General Health',
                'value': gen_health,
                'impact': 'Moderate',
                'explanation': 'Self-reported poor health status correlates with increased cardiovascular risk and may indicate multiple underlying health issues requiring attention.'
            })
        
        # Analyze sleep
        sleep_time = int(patient_data.get('SleepTime', 7))
        if sleep_time < 6 or sleep_time > 9:
            analysis['modifiable_factors'].append('Sleep Quality')
        
        return analysis
    
    def _generate_lifestyle_recommendations(self, patient_data: Dict) -> List[Dict]:
        """Generate personalized lifestyle recommendations"""
        recommendations = []
        
        # Smoking cessation
        if patient_data.get('Smoking') == 'Yes':
            recommendations.append({
                'category': 'Smoking Cessation',
                'priority': 'Critical',
                'recommendation': 'Quit smoking immediately',
                'details': 'Smoking cessation is the single most important step you can take to reduce cardiovascular risk. Benefits begin within hours of quitting.',
                'action_steps': [
                    'Consult with your healthcare provider about smoking cessation programs',
                    'Consider nicotine replacement therapy or prescription medications',
                    'Join a support group or counseling program',
                    'Identify and avoid smoking triggers',
                    'Set a quit date and inform family and friends for support'
                ],
                'expected_benefit': 'Risk reduction of 30-50% within 1-2 years of quitting'
            })
        
        # Physical activity
        if patient_data.get('PhysicalActivity') == 'No':
            recommendations.append({
                'category': 'Physical Activity',
                'priority': 'High',
                'recommendation': 'Begin regular exercise program',
                'details': 'Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week.',
                'action_steps': [
                    'Start with 10-15 minutes of walking daily and gradually increase',
                    'Include both aerobic exercise and strength training',
                    'Find activities you enjoy to maintain consistency',
                    'Consider working with a fitness professional initially',
                    'Track your progress and set achievable goals'
                ],
                'expected_benefit': 'Can reduce cardiovascular risk by 20-30%'
            })
        
        # Weight management
        bmi = float(patient_data.get('BMI', 25))
        if bmi > 25:
            target_bmi = 'below 25' if bmi > 30 else 'within normal range (18.5-24.9)'
            recommendations.append({
                'category': 'Weight Management',
                'priority': 'High' if bmi > 30 else 'Moderate',
                'recommendation': f'Achieve and maintain healthy weight (target BMI {target_bmi})',
                'details': f'Current BMI of {bmi:.1f} indicates {"obesity" if bmi > 30 else "overweight"}. Even modest weight loss of 5-10% can significantly improve cardiovascular health.',
                'action_steps': [
                    'Consult with a registered dietitian for personalized meal planning',
                    'Focus on whole foods, fruits, vegetables, and lean proteins',
                    'Practice portion control and mindful eating',
                    'Combine dietary changes with regular physical activity',
                    'Set realistic weight loss goals (1-2 pounds per week)',
                    'Keep a food diary to track intake and identify patterns'
                ],
                'expected_benefit': 'Each 10% reduction in body weight can lower blood pressure by 5-20 mmHg'
            })
        
        # Diet recommendations
        recommendations.append({
            'category': 'Nutrition',
            'priority': 'High',
            'recommendation': 'Adopt heart-healthy dietary pattern',
            'details': 'Follow a Mediterranean-style or DASH diet emphasizing fruits, vegetables, whole grains, lean proteins, and healthy fats.',
            'action_steps': [
                'Increase intake of fruits and vegetables (5-9 servings daily)',
                'Choose whole grains over refined grains',
                'Include fatty fish (salmon, mackerel) 2-3 times per week',
                'Limit saturated fat, trans fat, and cholesterol',
                'Reduce sodium intake to less than 2,300 mg per day',
                'Limit added sugars and processed foods',
                'Stay well-hydrated with water'
            ],
            'expected_benefit': 'Can reduce cardiovascular events by 25-30%'
        })
        
        # Stress management
        mental_health = int(patient_data.get('MentalHealth', 0))
        if mental_health > 7:
            recommendations.append({
                'category': 'Stress & Mental Health',
                'priority': 'High',
                'recommendation': 'Address mental health and stress management',
                'details': 'Chronic stress and poor mental health significantly impact cardiovascular health through multiple pathways.',
                'action_steps': [
                    'Practice stress-reduction techniques (meditation, deep breathing, yoga)',
                    'Ensure adequate sleep (7-9 hours nightly)',
                    'Consider counseling or therapy if needed',
                    'Maintain social connections and support networks',
                    'Engage in enjoyable hobbies and activities',
                    'Limit alcohol and avoid substance use'
                ],
                'expected_benefit': 'Improved mental health correlates with better cardiovascular outcomes'
            })
        
        # Sleep optimization
        sleep_time = int(patient_data.get('SleepTime', 7))
        if sleep_time < 6 or sleep_time > 9:
            recommendations.append({
                'category': 'Sleep Hygiene',
                'priority': 'Moderate',
                'recommendation': f'Optimize sleep duration (currently {sleep_time} hours)',
                'details': 'Both insufficient and excessive sleep are associated with increased cardiovascular risk. Aim for 7-9 hours of quality sleep nightly.',
                'action_steps': [
                    'Maintain consistent sleep schedule, even on weekends',
                    'Create a relaxing bedtime routine',
                    'Ensure bedroom is dark, quiet, and cool',
                    'Avoid screens 1-2 hours before bedtime',
                    'Limit caffeine and alcohol, especially in evening',
                    'Consider evaluation for sleep disorders if problems persist'
                ],
                'expected_benefit': 'Proper sleep improves blood pressure, metabolism, and overall cardiovascular health'
            })
        
        # Alcohol moderation
        if patient_data.get('AlcoholDrinking') == 'Yes':
            recommendations.append({
                'category': 'Alcohol Consumption',
                'priority': 'Moderate',
                'recommendation': 'Moderate or eliminate alcohol consumption',
                'details': 'Heavy alcohol use increases blood pressure, contributes to weight gain, and can lead to heart failure and stroke.',
                'action_steps': [
                    'Limit to no more than 1 drink per day for women, 2 for men',
                    'Consider alcohol-free days each week',
                    'Choose lower-alcohol options when drinking',
                    'Never binge drink',
                    'Seek help if you have difficulty controlling alcohol use'
                ],
                'expected_benefit': 'Reducing excessive alcohol can lower blood pressure and reduce cardiovascular risk'
            })
        
        return recommendations
    
    def _generate_medical_considerations(self, patient_data: Dict) -> Dict:
        """Generate medical considerations and screening recommendations"""
        considerations = {
            'immediate_actions': [],
            'screening_recommendations': [],
            'medication_considerations': [],
            'specialist_referrals': []
        }
        
        # Check for high-risk conditions
        if patient_data.get('Stroke') == 'Yes':
            considerations['immediate_actions'].append({
                'action': 'Urgent cardiology consultation',
                'reason': 'History of stroke requires comprehensive cardiovascular evaluation and aggressive risk factor management'
            })
            considerations['specialist_referrals'].append('Cardiologist')
            considerations['specialist_referrals'].append('Neurologist')
        
        if patient_data.get('Diabetic') in ['Yes', 'Yes (during pregnancy)']:
            considerations['immediate_actions'].append({
                'action': 'Diabetes management optimization',
                'reason': 'Diabetes significantly increases cardiovascular risk and requires tight glycemic control'
            })
            considerations['screening_recommendations'].append({
                'test': 'HbA1c',
                'frequency': 'Every 3-6 months',
                'purpose': 'Monitor long-term blood sugar control'
            })
            considerations['specialist_referrals'].append('Endocrinologist')
        
        if patient_data.get('KidneyDisease') == 'Yes':
            considerations['screening_recommendations'].append({
                'test': 'Kidney function tests (eGFR, creatinine)',
                'frequency': 'Every 3-6 months',
                'purpose': 'Monitor kidney function and adjust medications accordingly'
            })
            considerations['specialist_referrals'].append('Nephrologist')
        
        # Standard cardiovascular screening
        considerations['screening_recommendations'].extend([
            {
                'test': 'Blood pressure monitoring',
                'frequency': 'At least annually, more often if elevated',
                'purpose': 'Detect and manage hypertension'
            },
            {
                'test': 'Lipid panel (cholesterol, triglycerides)',
                'frequency': 'Every 4-6 years, more often if abnormal',
                'purpose': 'Assess cholesterol levels and cardiovascular risk'
            },
            {
                'test': 'Fasting blood glucose or HbA1c',
                'frequency': 'Every 3 years starting at age 45',
                'purpose': 'Screen for diabetes and prediabetes'
            },
            {
                'test': 'Electrocardiogram (ECG)',
                'frequency': 'Baseline and as clinically indicated',
                'purpose': 'Detect heart rhythm abnormalities and structural issues'
            }
        ])
        
        # Medication considerations
        age_cat = patient_data.get('AgeCategory', '')
        if any(x in age_cat for x in ['60', '65', '70', '75', '80']):
            considerations['medication_considerations'].append({
                'medication': 'Aspirin therapy',
                'consideration': 'Discuss with healthcare provider about daily low-dose aspirin for cardiovascular prevention',
                'note': 'Benefits and bleeding risks must be individually assessed'
            })
        
        if patient_data.get('Diabetic') == 'Yes':
            considerations['medication_considerations'].append({
                'medication': 'Statin therapy',
                'consideration': 'Consider statin for cholesterol management and cardiovascular protection',
                'note': 'Particularly important for diabetic patients with additional risk factors'
            })
        
        return considerations
    
    def _generate_preventive_measures(self, patient_data: Dict, prediction_result: Dict) -> List[Dict]:
        """Generate preventive measures based on risk level"""
        risk_level = prediction_result.get('risk_level', 'Low')
        measures = []
        
        if risk_level in ['High', 'Very High']:
            measures.append({
                'category': 'Immediate Prevention',
                'urgency': 'High',
                'measures': [
                    'Schedule comprehensive cardiovascular evaluation within 1-2 weeks',
                    'Begin daily blood pressure monitoring if not already doing so',
                    'Implement immediate lifestyle modifications (diet, exercise, smoking cessation)',
                    'Discuss preventive medications with healthcare provider',
                    'Create emergency action plan for cardiovascular symptoms'
                ]
            })
        
        measures.append({
            'category': 'Know Warning Signs',
            'urgency': 'Critical',
            'measures': [
                'Chest pain or discomfort (pressure, squeezing, fullness)',
                'Pain or discomfort in arms, back, neck, jaw, or stomach',
                'Shortness of breath with or without chest discomfort',
                'Cold sweat, nausea, or lightheadedness',
                'Sudden numbness or weakness, especially on one side',
                'Sudden confusion, trouble speaking, or understanding',
                'Sudden trouble seeing in one or both eyes',
                'Sudden severe headache with no known cause'
            ],
            'action': 'Call 911 immediately if experiencing any of these symptoms'
        })
        
        measures.append({
            'category': 'Regular Monitoring',
            'urgency': 'Ongoing',
            'measures': [
                'Track blood pressure regularly (daily if high risk)',
                'Monitor weight weekly',
                'Keep a symptom diary',
                'Track medication adherence',
                'Record physical activity and diet',
                'Note any new or worsening symptoms'
            ]
        })
        
        return measures
    
    def _generate_follow_up_recommendations(self, prediction_result: Dict) -> Dict:
        """Generate follow-up recommendations"""
        risk_level = prediction_result.get('risk_level', 'Low')
        
        follow_up = {
            'next_assessment': '',
            'monitoring_frequency': '',
            'key_metrics': [],
            'goals': []
        }
        
        if risk_level == 'Very High':
            follow_up['next_assessment'] = '1-3 months'
            follow_up['monitoring_frequency'] = 'Weekly to monthly'
            follow_up['key_metrics'] = [
                'Blood pressure (target <130/80 mmHg)',
                'LDL cholesterol (target <70 mg/dL)',
                'HbA1c if diabetic (target <7%)',
                'Weight and BMI',
                'Physical activity minutes per week'
            ]
        elif risk_level == 'High':
            follow_up['next_assessment'] = '3-6 months'
            follow_up['monitoring_frequency'] = 'Monthly'
            follow_up['key_metrics'] = [
                'Blood pressure (target <130/80 mmHg)',
                'LDL cholesterol (target <100 mg/dL)',
                'Weight and BMI',
                'Physical activity level'
            ]
        elif risk_level == 'Medium':
            follow_up['next_assessment'] = '6-12 months'
            follow_up['monitoring_frequency'] = 'Every 3-6 months'
            follow_up['key_metrics'] = [
                'Blood pressure (target <120/80 mmHg)',
                'Cholesterol levels',
                'Weight maintenance',
                'Lifestyle adherence'
            ]
        else:
            follow_up['next_assessment'] = '12 months'
            follow_up['monitoring_frequency'] = 'Annually'
            follow_up['key_metrics'] = [
                'Blood pressure',
                'Cholesterol levels',
                'Weight and BMI',
                'Overall health status'
            ]
        
        follow_up['goals'] = [
            'Maintain or improve current risk level',
            'Achieve target values for key health metrics',
            'Sustain healthy lifestyle modifications',
            'Prevent cardiovascular events'
        ]
        
        return follow_up
    
    def _generate_educational_content(self, patient_data: Dict) -> Dict:
        """Generate educational content"""
        content = {
            'understanding_risk': """Cardiovascular disease risk is determined by multiple factors including age, 
genetics, lifestyle choices, and existing health conditions. While some factors like age and genetics cannot 
be changed, many risk factors are modifiable through lifestyle changes and medical management.""",
            
            'importance_of_prevention': """Prevention is far more effective and less costly than treating 
cardiovascular disease after it develops. Even small improvements in risk factors can significantly reduce 
the likelihood of heart attack, stroke, and other cardiovascular events.""",
            
            'role_of_lifestyle': """Lifestyle factors account for approximately 80% of cardiovascular disease 
risk. This means that most cardiovascular disease is preventable through healthy lifestyle choices including 
not smoking, maintaining healthy weight, regular physical activity, and heart-healthy diet.""",
            
            'working_with_healthcare_team': """Effective cardiovascular risk management requires partnership 
between you and your healthcare team. Regular check-ups, open communication about symptoms and concerns, 
medication adherence, and lifestyle modifications all contribute to optimal outcomes.""",
            
            'resources': [
                {
                    'title': 'American Heart Association',
                    'url': 'https://www.heart.org',
                    'description': 'Comprehensive cardiovascular health information and resources'
                },
                {
                    'title': 'CDC Heart Disease Resources',
                    'url': 'https://www.cdc.gov/heartdisease',
                    'description': 'Prevention strategies and educational materials'
                },
                {
                    'title': 'National Heart, Lung, and Blood Institute',
                    'url': 'https://www.nhlbi.nih.gov',
                    'description': 'Research-based health information'
                }
            ]
        }
        
        return content


# Example usage
if __name__ == "__main__":
    generator = LLMInsightsGenerator()
    
    # Example patient data
    patient_data = {
        'BMI': 32.0,
        'Smoking': 'Yes',
        'AlcoholDrinking': 'Yes',
        'Stroke': 'No',
        'PhysicalHealth': 10,
        'MentalHealth': 15,
        'DiffWalking': 'Yes',
        'Sex': 'Male',
        'AgeCategory': '65-69',
        'Race': 'White',
        'Diabetic': 'Yes',
        'PhysicalActivity': 'No',
        'GenHealth': 'Fair',
        'SleepTime': 5,
        'Asthma': 'No',
        'KidneyDisease': 'No',
        'SkinCancer': 'No'
    }
    
    prediction_result = {
        'prediction': 1,
        'probability': 0.75,
        'risk_level': 'High',
        'risk_score': '15.0/20.0'
    }
    
    insights = generator.generate_comprehensive_insights(patient_data, prediction_result)
    
    print("Executive Summary:")
    print(insights['executive_summary'])
    print("\nLifestyle Recommendations:")
    for rec in insights['lifestyle_recommendations'][:2]:
        print(f"\n{rec['category']} ({rec['priority']} priority):")
        print(f"  {rec['recommendation']}")
