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

# Show feature importance (only numeric features)
feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in feature_columns if col != 'stable']

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

print("\nModel training completed successfully!")
print(f"Model accuracy: {model.score(scaler.transform(df[feature_columns]), df['stable']):.4f}")