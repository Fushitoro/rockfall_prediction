# Step 2: Create a synthetic dataset for rockfall prediction based on research features
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

# Sample size based on research (220-450 samples for good performance)
n_samples = 350

print("=== STEP 2: CREATING SYNTHETIC ROCKFALL DATASET ===")
print(f"Generating {n_samples} samples with realistic geological features...")

# Define realistic feature ranges based on research
data = {
    # Geological Parameters
    'slope_height_m': np.random.uniform(10, 200, n_samples),  # Slope height in meters
    'slope_angle_deg': np.random.uniform(30, 85, n_samples),  # Slope angle in degrees
    'cohesion_kpa': np.random.uniform(0, 100, n_samples),  # Cohesion in kPa
    'friction_angle_deg': np.random.uniform(20, 45, n_samples),  # Internal friction angle
    'unit_weight_kn_m3': np.random.uniform(20, 28, n_samples),  # Unit weight kN/m³
    'rqd_percent': np.random.uniform(10, 95, n_samples),  # Rock Quality Designation %
    'joint_spacing_m': np.random.uniform(0.1, 3.0, n_samples),  # Joint spacing in meters

    # Environmental Factors
    'rainfall_mm': np.random.exponential(5, n_samples),  # Rainfall in mm (exponential distribution)
    'temperature_range_c': np.random.uniform(5, 30, n_samples),  # Temperature variation
    'groundwater_depth_m': np.random.uniform(1, 50, n_samples),  # Groundwater depth
    'freeze_thaw_cycles': np.random.poisson(15, n_samples),  # Number of freeze-thaw cycles

    # Operational Parameters
    'blasting_distance_m': np.random.uniform(10, 500, n_samples),  # Distance from last blast
    'vibration_intensity': np.random.uniform(0, 10, n_samples),  # Vibration intensity
    'days_since_blast': np.random.uniform(1, 60, n_samples),  # Days since last blast
    'mining_depth_m': np.random.uniform(5, 150, n_samples),  # Current mining depth

    # Time-related features
    'days_since_rain': np.random.uniform(0, 30, n_samples),  # Days since last significant rain
    'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)  # Season
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert season to numerical
season_encoder = LabelEncoder()
df['season_encoded'] = season_encoder.fit_transform(df['season'])

print("Basic features created. Now generating derived features...")

# Create derived features based on geotechnical engineering principles
df['stability_index'] = (df['cohesion_kpa'] + df['unit_weight_kn_m3'] * df['slope_height_m'] *
                         np.tan(np.radians(df['friction_angle_deg']))) / \
                        (df['unit_weight_kn_m3'] * df['slope_height_m'] *
                         np.sin(np.radians(df['slope_angle_deg'])))

df['weather_risk_score'] = (df['rainfall_mm'] * df['temperature_range_c']) / (df['days_since_rain'] + 1)

df['operational_stress'] = df['vibration_intensity'] / (df['blasting_distance_m'] + 1) * \
                           (60 - df['days_since_blast']) / 60

df['geological_weakness'] = (100 - df['rqd_percent']) / df['joint_spacing_m']

df['slope_steepness_factor'] = np.tan(np.radians(df['slope_angle_deg'])) * df['slope_height_m']

print("Derived features created. Now generating target variable...")

# Create target variable based on realistic risk factors
risk_score = (
    # Slope geometry impact (40% weight)
        0.4 * (df['slope_steepness_factor'] / df['slope_steepness_factor'].max()) +

        # Geological weakness (25% weight)
        0.25 * (df['geological_weakness'] / df['geological_weakness'].max()) +

        # Weather conditions (20% weight)
        0.2 * (df['weather_risk_score'] / df['weather_risk_score'].max()) +

        # Operational factors (15% weight)
        0.15 * (df['operational_stress'] / df['operational_stress'].max())
)

# Add some noise to make it more realistic
risk_score += np.random.normal(0, 0.1, n_samples)

# Create risk categories based on thresholds
df['risk_score'] = risk_score
df['risk_level'] = pd.cut(risk_score,
                          bins=[-np.inf, 0.3, 0.7, np.inf],
                          labels=['Low', 'Medium', 'High'])

# Create binary target (Rockfall/No Rockfall)
df['rockfall_binary'] = (risk_score > 0.6).astype(int)

# Display dataset info
print(f"\nDataset created successfully!")
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['risk_level'].value_counts())
print(f"\nBinary target distribution:")
print(df['rockfall_binary'].value_counts())

# Display first few rows
print(f"\nFirst 5 rows of the dataset:")
print(df.head())

# Save the dataset
df.to_csv('rockfall_synthetic_dataset.csv', index=False)
print(f"\nDataset saved as 'rockfall_synthetic_dataset.csv'")