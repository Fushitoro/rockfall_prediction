# Step 5: Train Multiple ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train_bin = joblib.load('y_train_bin.pkl')
y_test_bin = joblib.load('y_test_bin.pkl')
X_train_scaled = joblib.load('X_train_scaled.pkl')
X_test_scaled = joblib.load('X_test_scaled.pkl')
# ... similarly load for multiclass if needed ...
X_train_mc = joblib.load('X_train_mc.pkl')
X_test_mc = joblib.load('X_test_mc.pkl')
y_train_mc = joblib.load('y_train_mc.pkl')
y_test_mc = joblib.load('y_test_mc.pkl')
X_train_mc_scaled = joblib.load('X_train_mc_scaled.pkl')
X_test_mc_scaled = joblib.load('X_test_mc_scaled.pkl')
print("=== STEP 5: TRAINING MACHINE LEARNING MODELS ===")

# Initialize models based on research recommendations
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)  # Based on research finding k=7 optimal
}

# Train and evaluate binary classification models
print("\n1. BINARY CLASSIFICATION (Rockfall/No Rockfall)")
print("=" * 50)

binary_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Use scaled data for KNN and Logistic Regression, original for Random Forest
    if name in ['K-Nearest Neighbors', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train_bin)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train_bin)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test_bin, y_pred)

    # Handle case where only one class is predicted
    try:
        auc = roc_auc_score(y_test_bin, y_pred_proba)
    except:
        auc = 0.5  # Default if AUC can't be calculated

    binary_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC: {auc:.3f}")

# Find best binary model
best_binary_model_name = max(binary_results.keys(), key=lambda k: binary_results[k]['accuracy'])
best_binary_model = binary_results[best_binary_model_name]['model']

print(f"\nBest Binary Model: {best_binary_model_name}")
print(f"Best Binary Accuracy: {binary_results[best_binary_model_name]['accuracy']:.3f}")

# Train multiclass models
print("\n2. MULTICLASS CLASSIFICATION (Low/Medium/High Risk)")
print("=" * 50)

multiclass_results = {}

for name, model in models.items():
    print(f"\nTraining {name} for multiclass...")

    # Clone the model for multiclass
    if name == 'Random Forest':
        mc_model = RandomForestClassifier(random_state=42, n_estimators=100)
    elif name == 'Logistic Regression':
        mc_model = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
    else:  # KNN
        mc_model = KNeighborsClassifier(n_neighbors=7)

    # Use appropriate data
    if name in ['K-Nearest Neighbors', 'Logistic Regression']:
        mc_model.fit(X_train_mc_scaled, y_train_mc)
        y_pred_mc = mc_model.predict(X_test_mc_scaled)
    else:
        mc_model.fit(X_train_mc, y_train_mc)
        y_pred_mc = mc_model.predict(X_test_mc)

    accuracy_mc = accuracy_score(y_test_mc, y_pred_mc)

    multiclass_results[name] = {
        'model': mc_model,
        'accuracy': accuracy_mc,
        'predictions': y_pred_mc
    }

    print(f"Multiclass Accuracy: {accuracy_mc:.3f}")

# Find best multiclass model
best_mc_model_name = max(multiclass_results.keys(), key=lambda k: multiclass_results[k]['accuracy'])
best_mc_model = multiclass_results[best_mc_model_name]['model']

print(f"\nBest Multiclass Model: {best_mc_model_name}")
print(f"Best Multiclass Accuracy: {multiclass_results[best_mc_model_name]['accuracy']:.3f}")

print("\nModel training complete!")

joblib.dump(best_binary_model, 'best_models/rockfall_binary_model.pkl')
joblib.dump(best_mc_model, 'best_models/rockfall_multiclass_model.pkl')

print("Models saved successfully!")