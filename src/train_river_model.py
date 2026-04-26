import pandas as pd
import numpy as np
from river import linear_model, preprocessing, metrics
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Tuple, List
from embedding_client import EmbeddingClient

class RiverModelTrainer:
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features from the dataframe"""
        # Select numerical features
        numerical_features = [
            'heart_rate', 'respiratory_rate', 'oxygen_saturation', 'temperature',
            'systolic_bp', 'diastolic_bp', 'glucose', 'lactate', 'creatinine',
            'bun', 'wbc', 'platelets', 'hemoglobin', 'sodium', 'potassium',
            'chloride', 'bicarbonate', 'bun_creatinine_ratio', 'urine_output'
        ]

        # Filter for available features
        available_features = [f for f in numerical_features if f in df.columns]

        # Extract features and target
        X = df[available_features].values
        y = df['sepsis_risk'].values

        return X, y

    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> None:
        """Train the River model"""
        print("Preparing features...")
        X, y = self.prepare_features(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model
        self.model = linear_model.LogisticRegression()

        # Train model incrementally (online learning)
        print("Training model...")
        for i in range(len(X_scaled)):
            # Convert to numpy array if needed
            x_sample = X_scaled[i].reshape(1, -1)
            y_sample = y[i]

            # Fit the model incrementally
            self.model.learn_one(x_sample[0], y_sample)

        self.is_fitted = True
        print("Model training completed!")

    def evaluate_model(self, df: pd.DataFrame) -> dict:
        """Evaluate the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        X, y = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = []
        for i in range(len(X_scaled)):
            x_sample = X_scaled[i].reshape(1, -1)
            pred = self.model.predict_one(x_sample[0])
            predictions.append(pred)

        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == y)

        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'actual': y.tolist()
        }

    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        print(f"Model loaded from {filepath}")

# Main training function
def main():
    # Load data
    print("Loading data...")
    data_path = "data/raw_synthetic.csv.gz"
    df = pd.read_csv(data_path)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Initialize embedding client
    embedding_client = EmbeddingClient()

    # Initialize trainer
    trainer = RiverModelTrainer(embedding_client)

    # Train model
    trainer.train_model(df)

    # Evaluate model
    print("Evaluating model...")
    eval_results = trainer.evaluate_model(df)
    print(f"Model accuracy: {eval_results['accuracy']:.4f}")

    # Save model
    model_path = "models/river_model.pkl"
    os.makedirs("models", exist_ok=True)
    trainer.save_model(model_path)

if __name__ == "__main__":
    main()