import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('cleanset_v1.csv')

# Display basic info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum().sort_values(ascending=False))

# Define features and target
# Using all numeric columns as features
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
target_column = 'stable'  # Target variable

# Remove target from features
feature_columns = [col for col in numeric_columns if col != target_column]

print(f"\nFeature columns: {feature_columns}")
print(f"Target column: {target_column}")

# Prepare the data
X = df[feature_columns]
y = df[target_column]

# Handle missing values in features by filling with median
X = X.fillna(X.median())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved successfully!")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
