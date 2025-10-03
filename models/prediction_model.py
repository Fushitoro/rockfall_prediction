import os

import joblib
import numpy as np
import pandas as pd

print("=== STEP 7: CREATING PREDICTION SYSTEM ===")

# Set base directory relative to this script
base_path = os.path.abspath(os.path.dirname(__file__))
best_models_dir = os.path.join(base_path, "best_models")

# Model files paths (prefer best_models folder)
binary_model_path = os.path.join(best_models_dir, 'rockfall_binary_model.pkl')
multiclass_model_path = os.path.join(best_models_dir, 'rockfall_multiclass_model.pkl')
label_encoder_path = os.path.join(best_models_dir, 'rockfall_label_encoder.pkl')

# Fallback to base_path if files not found in best_models folder
if not os.path.exists(binary_model_path):
    binary_model_path = os.path.join(base_path, 'rockfall_binary_model.pkl')
if not os.path.exists(multiclass_model_path):
    multiclass_model_path = os.path.join(base_path, 'rockfall_multiclass_model.pkl')
if not os.path.exists(label_encoder_path):
    label_encoder_path = os.path.join(base_path, 'rockfall_label_encoder.pkl')

print("Loading binary model from:", binary_model_path)
print("Loading multiclass model from:", multiclass_model_path)
print("Loading label encoder from:", label_encoder_path)

# Load models and encoder
best_binary_model = joblib.load(binary_model_path)
best_multiclass_model = joblib.load(multiclass_model_path)
le = joblib.load(label_encoder_path)

feature_columns = [
    'slope_height_m', 'slope_angle_deg', 'cohesion_kpa', 'friction_angle_deg',
    'unit_weight_kn_m3', 'rqd_percent', 'joint_spacing_m', 'rainfall_mm',
    'temperature_range_c', 'groundwater_depth_m', 'freeze_thaw_cycles',
    'blasting_distance_m', 'vibration_intensity', 'days_since_blast',
    'mining_depth_m', 'days_since_rain', 'season_encoded',
    'stability_index', 'weather_risk_score', 'operational_stress',
    'geological_weakness', 'slope_steepness_factor'
]

def calculate_derived_features(input_features):
    stability_index = (input_features.get('cohesion_kpa', 30) +
                       input_features.get('unit_weight_kn_m3', 24) *
                       input_features.get('slope_height_m', 50) *
                       np.tan(np.radians(input_features.get('friction_angle_deg', 35)))) / \
                      (input_features.get('unit_weight_kn_m3', 24) *
                       input_features.get('slope_height_m', 50) *
                       np.sin(np.radians(input_features.get('slope_angle_deg', 45))))

    weather_risk = (input_features.get('rainfall_mm', 10) *
                    input_features.get('temperature_range_c', 15)) / \
                   (input_features.get('days_since_rain', 5) + 1)

    operational_stress = input_features.get('vibration_intensity', 3) / \
                         (input_features.get('blasting_distance_m', 100) + 1) * \
                         (60 - input_features.get('days_since_blast', 7)) / 60

    geological_weakness = (100 - input_features.get('rqd_percent', 60)) / \
                         input_features.get('joint_spacing_m', 1.5)

    slope_steepness = np.tan(np.radians(input_features.get('slope_angle_deg', 45))) * \
                      input_features.get('slope_height_m', 50)

    return {
        'stability_index': stability_index,
        'weather_risk_score': weather_risk,
        'operational_stress': operational_stress,
        'geological_weakness': geological_weakness,
        'slope_steepness_factor': slope_steepness
    }

def prepare_feature_dataframe(input_features):
    base_features = {k: input_features.get(k, 0) for k in feature_columns[:-5]}
    derived_features = calculate_derived_features(input_features)
    all_features = {**base_features, **derived_features}
    df = pd.DataFrame([all_features], columns=feature_columns)
    return df

def predict_rockfall_risk_binary(input_features):
    feature_df = prepare_feature_dataframe(input_features)
    prediction = best_binary_model.predict(feature_df)[0]
    prob = best_binary_model.predict_proba(feature_df)[0][1]

    if prob > 0.8:
        risk_level = "CRITICAL"
        recommendation = "Immediate evacuation required. Stop all operations."
        color = "red"
    elif prob > 0.6:
        risk_level = "HIGH"
        recommendation = "High risk detected. Consider evacuation."
        color = "orange"
    elif prob > 0.4:
        risk_level = "MEDIUM"
        recommendation = "Moderate risk. Increase monitoring."
        color = "yellow"
    else:
        risk_level = "LOW"
        recommendation = "Low risk. Continue regular monitoring."
        color = "green"

    return {
        'prediction': int(prediction),
        'risk_level': risk_level,
        'confidence': float(prob),
        'recommendation': recommendation,
        'color': color,
        'alert_required': prob > 0.5
    }

def predict_rockfall_risk_multiclass(input_features):
    feature_df = prepare_feature_dataframe(input_features)
    prediction_encoded = best_multiclass_model.predict(feature_df)[0]
    probabilities = best_multiclass_model.predict_proba(feature_df)[0]
    prediction_label = le.inverse_transform([prediction_encoded])[0]

    confidence = float(max(probabilities))

    return {
        'prediction_encoded': int(prediction_encoded),
        'prediction_label': prediction_label,
        'confidence': confidence,
        'probabilities': probabilities.tolist()
    }

# Example test
if __name__ == "__main__":
    test_input = {
        'slope_height_m': 80,
        'slope_angle_deg': 65,
        'cohesion_kpa': 15,
        'friction_angle_deg': 30,
        'unit_weight_kn_m3': 25,
        'rqd_percent': 40,
        'joint_spacing_m': 0.8,
        'rainfall_mm': 25,
        'temperature_range_c': 20,
        'groundwater_depth_m': 10,
        'freeze_thaw_cycles': 20,
        'blasting_distance_m': 50,
        'vibration_intensity': 7,
        'days_since_blast': 2,
        'mining_depth_m': 60,
        'days_since_rain': 1,
        'season_encoded': 2
    }
    print("Binary Prediction:")
    print(predict_rockfall_risk_binary(test_input))
    print("\nMulticlass Prediction:")
    print(predict_rockfall_risk_multiclass(test_input))
