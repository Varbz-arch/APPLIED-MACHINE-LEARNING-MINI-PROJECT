from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
import os
from suggestions import DISEASE_SUGGESTIONS, GENERAL_TIPS

app = Flask(__name__)
CORS(app)

# Load encoders (still needed for state/month encoding)
print("Loading encoders...")
state_encoder = joblib.load('state_encoder.pkl')
month_encoder = joblib.load('month_encoder.pkl')
print("✅ Encoders loaded!")

# Disease mapping
disease_list = ['Typhoid', 'Cholera', 'Malaria', 'Dengue', 'Heatstroke']

# Rule-based disease prediction based on climate conditions
def predict_disease_by_rules(temperature, humidity, rainfall, state, month):
    """
    Predict disease based on environmental conditions
    Returns: (disease_name, confidence_percentage)
    """
    
    # HEATSTROKE: Very high temperature
    if temperature > 38:
        return "Heatstroke", 85
    
    # HEATSTROKE: High temperature + low humidity
    if temperature > 35 and humidity < 40:
        return "Heatstroke", 80
    
    # MALARIA: High rainfall + high humidity
    if rainfall > 200 and humidity > 75:
        return "Malaria", 82
    
    # MALARIA: High humidity + monsoon season
    if humidity > 80 and month in ['June', 'July', 'August', 'September']:
        return "Malaria", 75
    
    # CHOLERA: High rainfall + monsoon season
    if rainfall > 150 and month in ['June', 'July', 'August', 'September']:
        return "Cholera", 78
    
    # CHOLERA: Heavy rainfall areas
    if rainfall > 250:
        return "Cholera", 80
    
    # DENGUE: Moderate rainfall + specific months
    if 80 < rainfall < 200 and month in ['July', 'August', 'September', 'October']:
        return "Dengue", 72
    
    # DENGUE: High humidity + moderate temperature
    if humidity > 70 and 25 < temperature < 35 and month in ['July', 'August', 'September']:
        return "Dengue", 70
    
    # State-specific rules
    # Coastal states with high humidity
    if state in ['Kerala', 'West Bengal', 'Goa', 'Tamil Nadu'] and humidity > 75:
        return "Malaria", 73
    
    # Desert/arid states
    if state in ['Rajasthan', 'Gujarat', 'Haryana'] and temperature > 35:
        return "Heatstroke", 75
    
    # High population density states
    if state in ['Delhi', 'Mumbai', 'Kolkata', 'Bangalore'] and rainfall > 100:
        return "Cholera", 70
    
    # Monsoon months in eastern states
    if state in ['West Bengal', 'Assam', 'Odisha', 'Bihar'] and month in ['June', 'July', 'August']:
        return "Cholera", 72
    
    # Summer months in northern states
    if state in ['Punjab', 'Haryana', 'Uttar Pradesh', 'Delhi'] and month in ['April', 'May', 'June']:
        return "Heatstroke", 68
    
    # Default based on season
    if month in ['March', 'April', 'May']:
        return "Typhoid", 65
    elif month in ['June', 'July', 'August', 'September']:
        return "Malaria", 68
    elif month in ['October', 'November']:
        return "Dengue", 62
    else:  # Winter months
        return "Typhoid", 60

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get inputs from frontend
        state = data['state']
        month = data['month']
        temperature = float(data['temp'])
        humidity = float(data['humidity'])
        rainfall = float(data['rainfall'])
        
        print(f"\n📊 Predicting for: {state}, {month}")
        print(f"   Temperature: {temperature}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm")
        
        # Get prediction from rule-based system
        disease, confidence = predict_disease_by_rules(temperature, humidity, rainfall, state, month)
        
        print(f"   → Predicted: {disease} with {confidence}% confidence")
        
        # Create top 3 predictions with different probabilities
        predictions = []
        predictions.append({'disease': disease, 'probability': confidence / 100})
        
        # Add secondary predictions based on primary disease
        if disease == "Heatstroke":
            predictions.append({'disease': 'Typhoid', 'probability': 0.25})
            predictions.append({'disease': 'Malaria', 'probability': 0.15})
        elif disease == "Malaria":
            predictions.append({'disease': 'Dengue', 'probability': 0.28})
            predictions.append({'disease': 'Typhoid', 'probability': 0.22})
        elif disease == "Cholera":
            predictions.append({'disease': 'Typhoid', 'probability': 0.30})
            predictions.append({'disease': 'Malaria', 'probability': 0.25})
        elif disease == "Dengue":
            predictions.append({'disease': 'Malaria', 'probability': 0.32})
            predictions.append({'disease': 'Typhoid', 'probability': 0.28})
        else:  # Typhoid
            predictions.append({'disease': 'Cholera', 'probability': 0.30})
            predictions.append({'disease': 'Malaria', 'probability': 0.20})
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    try:
        data = request.get_json()
        disease = data['disease']
        
        if disease in DISEASE_SUGGESTIONS:
            return jsonify({
                'success': True,
                'suggestions': DISEASE_SUGGESTIONS[disease],
                'general_tips': GENERAL_TIPS
            })
        else:
            return jsonify({
                'success': True,
                'suggestions': DISEASE_SUGGESTIONS.get('Typhoid', {
                    'what_to_do': ['Consult a doctor immediately', 'Get plenty of rest', 'Stay hydrated'],
                    'medical_attention': ['Fever lasting more than 3 days', 'Severe headache', 'Blood in stool'],
                    'prevention': ['Wash hands frequently', 'Drink clean water', 'Avoid street food'],
                    'risk_factors': ['Poor sanitation', 'Contaminated food/water', 'Close contact with infected']
                }),
                'general_tips': GENERAL_TIPS
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
