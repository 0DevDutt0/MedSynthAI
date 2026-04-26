import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from litserve import LitServer, LitAPI
from litserve.utils import wrap_predictor
import torch
from torch import Tensor
from embedding_client import EmbeddingClient
from train_river_model import RiverModelTrainer

class SepsisRiskAPI(LitAPI):
    def __init__(self):
        super().__init__()
        self.model = None
        self.embedding_client = None
        self.scaler = None
        self.is_fitted = False

    def setup(self, device):
        """Initialize the model and components"""
        print("Setting up Sepsis Risk API...")

        # Initialize embedding client
        self.embedding_client = EmbeddingClient()

        # Load the trained model
        model_path = "models/river_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_fitted = model_data['is_fitted']
            print("Model loaded successfully")
        else:
            print("No trained model found. Please train the model first.")
            raise FileNotFoundError("Trained model not found")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Decode the incoming request"""
        return request

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for the input data"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        try:
            # Extract features from input
            features = inputs.get('features', {})

            # Prepare numerical features
            numerical_features = [
                'heart_rate', 'respiratory_rate', 'oxygen_saturation', 'temperature',
                'systolic_bp', 'diastolic_bp', 'glucose', 'lactate', 'creatinine',
                'bun', 'wbc', 'platelets', 'hemoglobin', 'sodium', 'potassium',
                'chloride', 'bicarbonate', 'bun_creatinine_ratio', 'urine_output'
            ]

            # Convert to numpy array
            feature_values = []
            for feature in numerical_features:
                if feature in features:
                    feature_values.append(float(features[feature]))
                else:
                    feature_values.append(0.0)  # Default value for missing features

            # Scale features
            feature_array = np.array(feature_values).reshape(1, -1)
            feature_scaled = self.scaler.transform(feature_array)

            # Make prediction
            prediction = self.model.predict_one(feature_scaled[0])

            # Get prediction probability
            try:
                # For River models, we can get the probability
                prob = self.model.predict_proba_one(feature_scaled[0])
                probability = prob.get(1, 0.0) if isinstance(prob, dict) else 0.0
            except:
                probability = 0.5  # Default probability if not available

            return {
                'sepsis_risk': int(prediction),
                'risk_probability': float(probability),
                'status': 'success'
            }

        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the response"""
        return output

def main():
    """Main function to start the LitServe server"""
    print("Starting LitServe server for Sepsis Risk Prediction...")

    # Create server
    server = LitServer(SepsisRiskAPI(), accelerator="cpu", devices=1)

    # Start server
    server.run(port=8000, host="0.0.0.0")

if __name__ == "__main__":
    main()