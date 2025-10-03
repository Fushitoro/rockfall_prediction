import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

print("=== STEP 6: HYPERPARAMETER TUNING FOR BINARY AND MULTICLASS ===")

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(project_root, 'dataset', 'rockfall_synthetic_dataset.csv')
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_path)

# Feature columns
feature_columns = [
    'slope_height_m', 'slope_angle_deg', 'cohesion_kpa', 'friction_angle_deg',
    'unit_weight_kn_m3', 'rqd_percent', 'joint_spacing_m', 'rainfall_mm',
    'temperature_range_c', 'groundwater_depth_m', 'freeze_thaw_cycles',
    'blasting_distance_m', 'vibration_intensity', 'days_since_blast',
    'mining_depth_m', 'days_since_rain', 'season_encoded',
    'stability_index', 'weather_risk_score', 'operational_stress',
    'geological_weakness', 'slope_steepness_factor'
]

X = df[feature_columns]

# Prepare binary target
y_binary = df['rockfall_binary']

# Prepare multiclass target and encode
y_multiclass = df['risk_level']
le = LabelEncoder()
y_multiclass_encoded = le.fit_transform(y_multiclass)

# Split binary data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# Split multiclass data
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    X, y_multiclass_encoded, test_size=0.3, random_state=42, stratify=y_multiclass_encoded
)

# Hyperparameter grid
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 7, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Function to tune and evaluate RF model
def tune_rf(X_train, y_train, X_test, y_test, task_name, scoring='accuracy', binary=False):
    print(f"\nTuning Random Forest for {task_name}...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    print(f"Best parameters for {task_name}: {rf_grid.best_params_}")
    print(f"Best CV score for {task_name}: {rf_grid.best_score_:.3f}")

    best_model = rf_grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{task_name} Accuracy: {accuracy:.3f}")

    if binary:
        try:
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"{task_name} AUC: {auc:.3f}")
        except Exception:
            auc = 0.5
            print(f"{task_name} AUC could not be calculated, defaulting to 0.5")
        return best_model, accuracy, auc
    else:
        # Fix applied here: convert class names to strings explicitly
        print(f"{task_name} Classification Report:\n{classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_])}")
        return best_model, accuracy

# Tune binary model
best_rf_tuned_bin, bin_accuracy, bin_auc = tune_rf(X_train_bin, y_train_bin, X_test_bin, y_test_bin,
                                                  'Binary Classification (Rockfall/No Rockfall)', binary=True)

# Tune multiclass model
best_rf_tuned_mc, mc_accuracy = tune_rf(X_train_mc, y_train_mc, X_test_mc, y_test_mc,
                                       'Multiclass Classification (Risk Levels)')

# Save models and label encoder
joblib.dump(best_rf_tuned_bin, os.path.join(models_dir, 'rockfall_binary_model.pkl'))
joblib.dump(best_rf_tuned_mc, os.path.join(models_dir, 'rockfall_multiclass_model.pkl'))
joblib.dump(le, os.path.join(models_dir, 'rockfall_label_encoder.pkl'))

print("\nBest models and label encoder saved successfully.")

# Feature importance from multiclass tuned model
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_rf_tuned_mc.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features (Multiclass Model):")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

feature_importance.to_csv(os.path.join(models_dir, 'feature_importance.csv'), index=False)
print(f"\nFeature importance saved to '{os.path.join(models_dir, 'feature_importance.csv')}'")
