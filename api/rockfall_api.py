import os
import random
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'best_models')

try:
    binary_model = joblib.load(os.path.join(MODEL_DIR, 'rockfall_binary_model.pkl'))
    print("Binary model loaded successfully")
except Exception as e:
    binary_model = None
    print(f"Failed loading binary model: {e}")

try:
    multiclass_model = joblib.load(os.path.join(MODEL_DIR, 'rockfall_multiclass_model.pkl'))
    print("Multiclass model loaded successfully")
except Exception as e:
    multiclass_model = None
    print(f"Failed loading multiclass model: {e}")

try:
    le = joblib.load(os.path.join(MODEL_DIR, 'rockfall_label_encoder.pkl'))
    print("Label encoder loaded successfully")
except Exception as e:
    le = None
    print(f"Failed loading label encoder: {e}")

FEATURES_BASE = [
    'slope_height_m', 'slope_angle_deg', 'cohesion_kpa', 'friction_angle_deg', 'unit_weight_kn_m3',
    'rqd_percent', 'joint_spacing_m', 'rainfall_mm', 'temperature_range_c', 'groundwater_depth_m',
    'freeze_thaw_cycles', 'blasting_distance_m', 'vibration_intensity', 'days_since_blast',
    'mining_depth_m', 'days_since_rain', 'season_encoded'
]

def calculate_features(input_data):
    try:
        stability_index = (
            input_data['cohesion_kpa'] +
            input_data['unit_weight_kn_m3'] * input_data['slope_height_m'] *
            np.tan(np.radians(input_data['friction_angle_deg']))
        ) / (
            input_data['unit_weight_kn_m3'] * input_data['slope_height_m'] *
            np.sin(np.radians(input_data['slope_angle_deg']))
        )
        weather_risk_score = (input_data['rainfall_mm'] * input_data['temperature_range_c']) / (input_data['days_since_rain'] + 1)
        operational_stress = input_data['vibration_intensity'] / (input_data['blasting_distance_m'] + 1) * (60 - input_data['days_since_blast']) / 60
        geological_weakness = (100 - input_data['rqd_percent']) / input_data['joint_spacing_m']
        slope_steepness_factor = np.tan(np.radians(input_data['slope_angle_deg'])) * input_data['slope_height_m']

        base_features = [
            input_data.get('slope_height_m', 0), input_data.get('slope_angle_deg', 0), input_data.get('cohesion_kpa', 0),
            input_data.get('friction_angle_deg', 0), input_data.get('unit_weight_kn_m3', 0), input_data.get('rqd_percent', 0),
            input_data.get('joint_spacing_m', 0), input_data.get('rainfall_mm', 0), input_data.get('temperature_range_c', 0),
            input_data.get('groundwater_depth_m', 0), input_data.get('freeze_thaw_cycles', 0), input_data.get('blasting_distance_m', 0),
            input_data.get('vibration_intensity', 0), input_data.get('days_since_blast', 0), input_data.get('mining_depth_m', 0),
            input_data.get('days_since_rain', 0), input_data.get('season_encoded', 0),
        ]
        extended_features = [stability_index, weather_risk_score, operational_stress, geological_weakness, slope_steepness_factor]
        features = base_features + extended_features
        return features
    except Exception as ex:
        print(f"Feature calculation error: {ex}")
        return None

def generate_sensor_data():
    now = datetime.now()
    return {
        'timestamp': now.isoformat(),
        'mine_id': 'MINE_001',
        'slope_height_m': round(random.uniform(40, 150), 1),
        'slope_angle_deg': round(random.uniform(30, 80), 1),
        'cohesion_kpa': round(random.uniform(10, 100), 1),
        'friction_angle_deg': round(random.uniform(20, 50), 1),
        'unit_weight_kn_m3': round(random.uniform(20, 30), 1),
        'rqd_percent': round(random.uniform(10, 95), 1),
        'joint_spacing_m': round(random.uniform(0.3, 3.5), 2),
        'rainfall_mm': round(np.random.exponential(10), 1),
        'temperature_range_c': round(random.uniform(5, 40), 1),
        'groundwater_depth_m': round(random.uniform(1, 50), 1),
        'freeze_thaw_cycles': random.randint(0, 30),
        'blasting_distance_m': round(random.uniform(10, 400), 1),
        'vibration_intensity': round(random.uniform(0, 15), 2),
        'days_since_blast': random.randint(0, 60),
        'mining_depth_m': round(random.uniform(5, 150), 1),
        'days_since_rain': random.randint(0, 20),
        'season_encoded': random.randint(0, 3)
    }

@app.route('/')
def home():
    return jsonify({"message": "API is running"})

@app.route('/simulate-and-predict')
def simulate_and_predict():
    if binary_model is None:
        return jsonify({"error": "Binary model not loaded."}), 500

    sensor_data = generate_sensor_data()
    features = calculate_features(sensor_data)
    if features is None:
        return jsonify({'error': 'Feature calculation failed'}), 500

    feature_names_extended = FEATURES_BASE + [
        'stability_index', 'weather_risk_score', 'operational_stress', 'geological_weakness', 'slope_steepness_factor'
    ]
    features_df = pd.DataFrame([features], columns=feature_names_extended)

    binary_result = {'prediction': None, 'confidence': 0.0, 'risk_level': 'UNKNOWN', 'recommendation': ''}
    multiclass_result = {'prediction_label': "N/A", 'confidence': 0.0}

    try:
        binary_pred = binary_model.predict(features_df)[0]
        binary_prob = binary_model.predict_proba(features_df)[0][1]

        if binary_prob > 0.4:
            risk_level = "CRITICAL"
            recommendation = "Immediate evacuation required."
        elif binary_prob > 0.3:
            risk_level = "HIGH"
            recommendation = "High risk detected. Consider evacuation."
        elif binary_prob > 0.1:
            risk_level = "MEDIUM"
            recommendation = "Moderate risk. Increase monitoring."
        else:
            risk_level = "LOW"
            recommendation = "Low risk. Continue operations."

        binary_result = {
            'prediction': int(binary_pred),
            'confidence': float(np.round(binary_prob, 2)),
            'risk_level': risk_level,
            'recommendation': recommendation
        }
    except Exception as e:
        print(f"Binary prediction error: {e}")

    if multiclass_model and le:
        try:
            mc_pred_idx = multiclass_model.predict(features_df)[0]
            mc_probs = multiclass_model.predict_proba(features_df)[0]
            conf_threshold = 0.5
            if max(mc_probs) < conf_threshold:
                import random
                alt_idx = (mc_pred_idx + random.choice([-1, 1])) % len(mc_probs)
                mc_pred_idx = alt_idx

            mc_label = le.inverse_transform([mc_pred_idx])[0]
            mc_conf = float(np.round(max(mc_probs), 2))
            multiclass_result = {
                'prediction_label': str(mc_label),
                'confidence': mc_conf,
                'probabilities': mc_probs.tolist()
            }
        except Exception as e:
            print(f"Multiclass prediction error: {e}")

    result = {
        'sensor_data': sensor_data,
        'prediction': {
            'binary_result': binary_result,
            'multiclass_result': multiclass_result
        },
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
