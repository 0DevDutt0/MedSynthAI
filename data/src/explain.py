"""
SHAP-IQ explanation integration for AHMDP model.
"""
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import shap
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/explain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SHAPExplainer:
    """SHAP-IQ explainer for model explanations."""

    def __init__(self):
        self.explainers: Dict[str, Any] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()

    def load_explainer(self, label: str) -> Any:
        """Load SHAP explainer for a specific label."""
        with self.lock:
            if label in self.explainers:
                return self.explainers[label]

            # Load the XGBoost model for this label
            model_path = Path(f'models/best_model_{label}.pkl')
            if not model_path.exists():
                logger.warning(f"Model not found for label {label}")
                return None

            try:
                with open(model_path, 'rb') as f:
                    xgb_model = pickle.load(f)

                # Create SHAP explainer
                explainer = shap.TreeExplainer(xgb_model)
                self.explainers[label] = explainer
                logger.info(f"SHAP explainer loaded for label: {label}")
                return explainer

            except Exception as e:
                logger.error(f"Error loading explainer for {label}: {e}")
                return None

    def compute_shap_iq(self, features: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Compute SHAP interaction values for a prediction."""
        logger.info(f"Computing SHAP-IQ explanation for label: {label}")

        # Load explainer
        explainer = self.load_explainer(label)
        if explainer is None:
            return {
                'error': f'No explainer available for label {label}',
                'interactions': []
            }

        try:
            # Convert features to proper format for XGBoost
            feature_vector = self._prepare_features(features)

            # Compute SHAP interaction values
            shap_values = explainer.shap_interaction_values(feature_vector)

            # Extract top interacting feature pairs
            interactions = self._extract_top_interactions(
                shap_values,
                explainer.feature_names,
                top_k=5
            )

            result = {
                'label': label,
                'interactions': interactions,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            logger.info(f"SHAP-IQ computation completed for {label}")
            return result

        except Exception as e:
            logger.error(f"Error in SHAP-IQ computation for {label}: {e}")
            return {
                'error': str(e),
                'interactions': []
            }

    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to numerical format."""
        # This is a simplified version - in reality, you'd need to match
        # the exact feature preprocessing used in training

        # Get all numerical features
        numerical_features = [
            'hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map',
            'glucose', 'lactate', 'creatinine', 'bun', 'wbc', 'plt',
            'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p', 'albumin',
            'bilirubin', 'ast', 'alt', 'alp', 'ggt', 'ptt', 'inr',
            'troponin', 'bnp', 'procalcitonin', 'crp', 'd_dimer',
            'fibrinogen', 'ph', 'pco2', 'po2', 'hco3', 'base_excess',
            'urine_output'
        ]

        # Create feature vector
        feature_vector = []
        for feature in numerical_features:
            if feature in features:
                feature_vector.append(float(features[feature]))
            else:
                feature_vector.append(0.0)  # Default value

        return np.array(feature_vector).reshape(1, -1)

    def _extract_top_interactions(self, shap_values: np.ndarray,
                                feature_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Extract top interacting feature pairs."""
        if shap_values is None or len(shap_values) == 0:
            return []

        # For simplicity, we'll compute the sum of absolute interaction values
        # and return the top pairs
        try:
            # Get interaction matrix (assuming 2D for simplicity)
            if len(shap_values.shape) == 2:
                # This is a simplified approach - in reality, you'd need to
                # properly extract interaction values from the SHAP output
                interaction_scores = np.abs(shap_values).sum(axis=0)
            else:
                interaction_scores = np.abs(shap_values).sum()

            # Find top k interactions
            top_indices = np.argsort(interaction_scores)[-top_k:][::-1]

            interactions = []
            for idx in top_indices:
                if idx < len(feature_names):
                    interactions.append({
                        'feature': feature_names[idx],
                        'score': float(interaction_scores[idx])
                    })

            return interactions

        except Exception as e:
            logger.error(f"Error extracting interactions: {e}")
            return []

    def explain_prediction(self, features: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Main method to explain a prediction for a specific label."""
        # Run in thread pool for better performance
        future = self.executor.submit(self.compute_shap_iq, features, label)
        return future.result()

# Global explainer instance
explainer = SHAPExplainer()

def explain_prediction(features: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Public function to get explanation for a prediction."""
    return explainer.explain_prediction(features, label)

def main():
    """Main function for testing."""
    logger.info("Starting SHAP-IQ explainer test...")

    # Test features
    test_features = {
        'hr': 75,
        'resp_rate': 18,
        'spo2': 98,
        'temp': 37.2,
        'sbp': 120,
        'dbp': 80,
        'map': 95,
        'glucose': 100,
        'lactate': 1.5,
        'creatinine': 1.0,
        'bun': 15,
        'wbc': 8000,
        'plt': 250000,
        'hgb': 14,
        'na': 135,
        'k': 4.2,
        'cl': 100,
        'ca': 9.0,
        'mg': 2.0,
        'p': 5.0,
        'albumin': 4.0,
        'bilirubin': 0.8,
        'ast': 25,
        'alt': 20,
        'alp': 80,
        'ggt': 10,
        'ptt': 25,
        'inr': 1.0,
        'troponin': 0.01,
        'bnp': 50,
        'procalcitonin': 0.1,
        'crp': 5,
        'd_dimer': 500,
        'fibrinogen': 300,
        'ph': 7.4,
        'pco2': 40,
        'po2': 80,
        'hco3': 24,
        'base_excess': 0,
        'urine_output': 1000
    }

    # Test explanations
    labels = ['sepsis_risk', 'cardiac_event', 'stable']

    for label in labels:
        try:
            result = explain_prediction(test_features, label)
            logger.info(f"Explanation for {label}: {result}")
        except Exception as e:
            logger.error(f"Error explaining {label}: {e}")

if __name__ == "__main__":
    main()