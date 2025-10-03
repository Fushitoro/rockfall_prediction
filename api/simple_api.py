# import random
# from datetime import datetime
#
# import joblib
# import numpy as np
# from flask import Flask, jsonify, request
#
# app = Flask(__name__)
#
# # Load both binary and multiclass models
# BINARY_MODEL_PATH = 'rockfall_binary_model.pkl'
# MULTICLASS_MODEL_PATH = 'rockfall_multiclass_model.pkl'
# ENCODER_PATH = 'rockfall_label_encoder.pkl'  # Only needed for multiclass
#
# try:
#     binary_model = joblib.load(BINARY_MODEL_PATH)
#     print("Binary model loaded successfully.")
# except Exception as e:
#     print(f"Failed to load binary model: {e}")
#     binary_model = None
#
# try:
#     multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
#     print("Multiclass model loaded successfully.")
# except Exception as e:
#     print(f"Failed to load multiclass model: {e}")
#     multiclass_model = None
#
# try:
#     le = joblib.load(ENCODER_PATH)
#     print("Label encoder loaded successfully.")
# except Exception as e:
#     le = None
#     print(f"Failed to load label encoder: {e}")
#
# FEATURES_BASE = [
#     'slope_height_m',
#     'slope_angle_deg',
#     'cohesion_kpa',
#     'friction_angle_deg',
#     'unit_weight_kn_m3',
#     'rqd_percent',
#     'joint_spacing_m',
#     'rainfall_mm',
#     'temperature_range_c',
#     'groundwater_depth_m',
#     'freeze_thaw_cycles',
#     'blasting_distance_m',
#     'vibration_intensity',
#     'days_since_blast',
#     'mining_depth',
#     'days_since_rain',
#     'season_encoded'
# ]
#
# @app.route('/')
# def home():
#     return jsonify({
#         "message": "Rockfall Prediction API (binary + multiclass)",
#         "endpoints": {
#             "/simulate-sensor-data": "GET - generate simulated sensor data",
#             "/predict-rockfall": "POST - predict rockfall risk using sensor data"
#         },
#         "status": "running"
#     })
#
# @app.route('/simulate-sensor-data')
# def simulate_sensor_data():
#     now = datetime.now()
#     data = {
#         'timestamp': now.isoformat(),
#         'mine_id': 'MINE_001',
#         'slope_height_m': round(random.uniform(40, 120), 1),
#         'slope_angle_deg': round(random.uniform(35, 75), 1),
#         'cohesion_kpa': round(random.uniform(10, 80), 1),
#         'friction_angle_deg': round(random.uniform(25, 45), 1),
#         'unit_weight_kn_m3': round(random.uniform(22, 27), 1),
#         'rqd_percent': round(random.uniform(20, 90), 1),
#         'joint_spacing_m': round(random.uniform(0.5, 3.0), 2),
#         'rainfall_mm': round(np.random.exponential(8), 1),
#         'temperature_range_c': round(random.uniform(10, 30), 1),
#         'groundwater_depth_m': round(random.uniform(5, 40), 1),
#         'freeze_thaw_cycles': random.randint(5, 25),
#         'blasting_distance_m': round(random.uniform(20, 300), 1),
#         'vibration_intensity': round(random.uniform(0, 10), 2),
#         'days_since_blast': random.randint(1, 30),
#         'mining_depth': round(random.uniform(10, 100), 1),
#         'days_since_rain': random.randint(0, 15),
#         'season_encoded': random.randint(0, 3)
#     }
#     return jsonify(data)
#
# def calculate_features(input_data):
#     stability_index = (input_data['cohesion_kpa'] + input_data['unit_weight_kn_m3'] * input_data['slope_height_m'] *
#                        np.tan(np.radians(input_data['friction_angle_deg']))) / \
#                       (input_data['unit_weight_kn_m3'] * input_data['slope_height_m'] *
#                        np.sin(np.radians(input_data['slope_angle_deg'])))
#
#     weather_risk = (input_data['rainfall_mm'] * input_data['temperature_range_c']) / (input_data['days_since_rain'] + 1)
#
#     operational_stress = input_data['vibration_intensity'] / (input_data['blasting_distance_m'] + 1) * \
#                          (60 - input_data['days_since_blast']) / 60
#
#     geological_weakness = (100 - input_data['rqd_percent']) / input_data['joint_spacing_m']
#
#     slope_steepness = np.tan(np.radians(input_data['slope_angle_deg'])) * input_data['slope_height_m']
#
#     features = [
#         input_data.get(key, 0) for key in FEATURES_BASE
#     ]
#     features.extend([
#         stability_index,
#         weather_risk,
#         operational_stress,
#         geological_weakness,
#         slope_steepness
#     ])
#     return features
#
# @app.route('/predict-rockfall', methods=['POST'])
# def predict_rockfall():
#     data = request.get_json()
#     if not data:
#         return jsonify({'error': 'No input data provided'}), 400
#
#     features = calculate_features(data)
#     response = {
#         'timestamp': datetime.now().isoformat()
#     }
#
#     # Binary model (risk)
#     if binary_model:
#         binary_pred = binary_model.predict([features])[0]
#         binary_prob = binary_model.predict_proba([features])[0][1]
#         if binary_prob > 0.4:
#             risk_level = "CRITICAL"
#             recommendation = "Immediate evacuation required."
#         elif binary_prob > 0.3:
#             risk_level = "HIGH"
#             recommendation = "High risk detected. Consider evacuation."
#         elif binary_prob > 0.10:
#             risk_level = "MEDIUM"
#             recommendation = "Moderate risk. Increase monitoring."
#         else:
#             risk_level = "LOW"
#             recommendation = "Low risk. Continue operations."
#         response["binary_result"] = {
#             'prediction': int(binary_pred),
#             'confidence': round(float(binary_prob), 2),
#             'risk_level': risk_level,
#             'recommendation': recommendation
#         }
#     else:
#         return jsonify({'error': 'Binary model not loaded'}), 500
#
#     # Multiclass model (label/classification)
#     if multiclass_model and le:
#         mc_pred_idx = multiclass_model.predict([features])[0]
#         mc_probs = multiclass_model.predict_proba([features])[0]
#         mc_label = le.inverse_transform([mc_pred_idx])[0]
#         mc_conf = round(float(max(mc_probs)), 2)
#         response["multiclass_result"] = {
#             'prediction_label': mc_label,
#             'confidence': mc_conf
#         }
#     else:
#         response["multiclass_result"] = {
#             'prediction_label': "N/A",
#             'confidence': 0.0
#         }
#
#     return jsonify(response)
#
# if __name__ == '__main__':
#     print("Starting Rockfall Combined API with dual-model endpoint...")
#     app.run(debug=True, port=5000 ,)
