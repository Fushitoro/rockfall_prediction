
# The initial dataset is too skewed - let's rebalance it to be more realistic
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset from CSV file saved after initial creation
df = pd.read_csv('rockfall_synthetic_dataset.csv')

# Ensure 'n_samples' is defined
n_samples = len(df)

# Then run the rebalancing code here...

print("=== STEP 3: REBALANCING DATASET FOR REALISTIC RISK DISTRIBUTION ===")

# Recreate with better risk distribution
np.random.seed(123)  # Different seed for different distribution

# Adjust risk score calculation to get more realistic distribution
risk_score_adjusted = (
    # More aggressive slope geometry impact
        0.35 * (df['slope_steepness_factor'] / df['slope_steepness_factor'].max()) +

        # Higher geological weakness impact
        0.3 * (df['geological_weakness'] / df['geological_weakness'].max()) +

        # Weather conditions
        0.2 * (df['weather_risk_score'] / df['weather_risk_score'].max()) +

        # Operational factors
        0.15 * (df['operational_stress'] / df['operational_stress'].max())
)

# Add controlled noise and bias toward higher risk
risk_score_adjusted += np.random.normal(0.15, 0.2, n_samples)  # Bias toward higher values
risk_score_adjusted = np.clip(risk_score_adjusted, 0, 1)  # Keep in [0,1] range

# Create more balanced risk categories
df['risk_score'] = risk_score_adjusted
df['risk_level'] = pd.cut(risk_score_adjusted,
                          bins=[0, 0.4, 0.7, 1.0],
                          labels=['Low', 'Medium', 'High'])

# Create binary target with better balance
df['rockfall_binary'] = (risk_score_adjusted > 0.55).astype(int)

print("Rebalanced target distribution:")
print(df['risk_level'].value_counts())
print(f"\nBinary target distribution:")
print(df['rockfall_binary'].value_counts())

# Save the rebalanced dataset
df.to_csv('rockfall_synthetic_dataset.csv', index=False)
print("\nRebalanced dataset saved!")