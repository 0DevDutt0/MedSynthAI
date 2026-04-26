import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df = pd.read_csv('cleanset_v1.csv')

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:")
print(df['stable'].value_counts())

# Show some sample data
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Show feature importance
# Get feature columns that were used for training
feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in feature_columns if col != 'stable']

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Demonstrate prediction on a few samples from the dataset
print("\nMaking predictions on first 5 samples from the dataset:")
sample_data = df.head(5).copy()

# Remove the target column for prediction
sample_features = sample_data.drop('stable', axis=1)

# Fill missing values
sample_features = sample_features.fillna(sample_features.median())

# Scale features
sample_features_scaled = scaler.transform(sample_features)

# Make predictions
predictions = model.predict(sample_features_scaled)
probabilities = model.predict_proba(sample_features_scaled)

# Display results
for i in range(len(sample_data)):
    print(f"Sample {i+1}:")
    print(f"  Actual stability: {sample_data.iloc[i]['stable']}")
    print(f"  Predicted stability: {predictions[i]}")
    print(f"  Probability stable: {probabilities[i][0]:.4f}")
    print(f"  Probability unstable: {probabilities[i][1]:.4f}")
    print()