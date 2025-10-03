# Step 4: Data Preprocessing and Feature Selection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
print("=== STEP 4: DATA PREPROCESSING ===")
# Load dataset CSV into df
df = pd.read_csv('rockfall_synthetic_dataset.csv')

# Select features for the model (remove categorical and intermediate features)
feature_columns = [
    'slope_height_m', 'slope_angle_deg', 'cohesion_kpa', 'friction_angle_deg',
    'unit_weight_kn_m3', 'rqd_percent', 'joint_spacing_m', 'rainfall_mm',
    'temperature_range_c', 'groundwater_depth_m', 'freeze_thaw_cycles',
    'blasting_distance_m', 'vibration_intensity', 'days_since_blast',
    'mining_depth_m', 'days_since_rain', 'season_encoded',
    'stability_index', 'weather_risk_score', 'operational_stress',
    'geological_weakness', 'slope_steepness_factor'
]

# Prepare data for modeling
X = df[feature_columns]
y_binary = df['rockfall_binary']
y_multiclass = df['risk_level']

print(f"Features selected: {len(feature_columns)}")
print(f"Feature columns: {feature_columns}")

# Split the data
X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# Split for multiclass (encode labels first)
le = LabelEncoder()
y_multiclass_encoded = le.fit_transform(y_multiclass)
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    X, y_multiclass_encoded, test_size=0.3, random_state=42, stratify=y_multiclass_encoded
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_mc_scaled = scaler.fit_transform(X_train_mc)
X_test_mc_scaled = scaler.transform(X_test_mc)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Binary target distribution in training: {np.bincount(y_train_bin)}")
print("\nData preprocessing complete!")

# Save the scaler and label encoder for later use
joblib.dump(scaler, '../models/best_models/rockfall_scaler.pkl')
joblib.dump(le, '../models/best_models/rockfall_label_encoder.pkl')

# Save data splits and scaled versions
joblib.dump(X_train, '../models/X_train.pkl')
joblib.dump(X_test, '../models/X_test.pkl')
joblib.dump(y_train_bin, '../models/y_train_bin.pkl')
joblib.dump(y_test_bin, '../models/y_test_bin.pkl')
joblib.dump(X_train_scaled, '../models/X_train_scaled.pkl')
joblib.dump(X_test_scaled, '../models/X_test_scaled.pkl')

joblib.dump(X_train_mc, '../models/X_train_mc.pkl')
joblib.dump(X_test_mc, '../models/X_test_mc.pkl')
joblib.dump(y_train_mc, '../models/y_train_mc.pkl')
joblib.dump(y_test_mc, '../models/y_test_mc.pkl')
joblib.dump(X_train_mc_scaled, '../models/X_train_mc_scaled.pkl')
joblib.dump(X_test_mc_scaled, '../models/X_test_mc_scaled.pkl')

print("Scaler, label encoder, and data splits saved!")
