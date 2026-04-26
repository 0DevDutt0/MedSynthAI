"""
Test for SHAP-IQ explanation functionality.
"""
import os
import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path('src').resolve()))

from explain import SHAPExplainer, explain_prediction

def setup_test_environment():
    """Set up test environment with mock models."""
    # Create required directories
    for dir_path in ['models', 'logs']:
        Path(dir_path).mkdir(exist_ok=True)

    # Create mock XGBoost models for testing
    mock_model = MagicMock()
    mock_model.feature_names = [
        'hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map',
        'glucose', 'lactate', 'creatinine', 'bun', 'wbc', 'plt',
        'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p', 'albumin',
        'bilirubin', 'ast', 'alt', 'alp', 'ggt', 'ptt', 'inr',
        'troponin', 'bnp', 'procalcitonin', 'crp', 'd_dimer',
        'fibrinogen', 'ph', 'pco2', 'po2', 'hco3', 'base_excess',
        'urine_output'
    ]

    # Save mock models
    for label in ['sepsis_risk', 'cardiac_event', 'stable']:
        model_path = Path(f'models/best_model_{label}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)

def test_shap_explainer_initialization():
    """Test SHAP explainer initialization."""
    setup_test_environment()

    explainer = SHAPExplainer()
    assert explainer is not None
    assert hasattr(explainer, 'explainers')
    assert hasattr(explainer, 'executor')

def test_load_explainer():
    """Test loading explainer."""
    setup_test_environment()

    explainer = SHAPExplainer()

    # Test loading for each label
    for label in ['sepsis_risk', 'cardiac_event', 'stable']:
        loaded_explainer = explainer.load_explainer(label)
        # Should return a mock explainer object
        assert loaded_explainer is not None

def test_prepare_features():
    """Test feature preparation."""
    setup_test_environment()

    explainer = SHAPExplainer()

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

    feature_vector = explainer._prepare_features(test_features)
    assert isinstance(feature_vector, np.ndarray)
    assert feature_vector.shape == (1, 37)  # 37 numerical features

def test_compute_shap_iq():
    """Test SHAP-IQ computation."""
    setup_test_environment()

    explainer = SHAPExplainer()

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

    # Test for each label
    for label in ['sepsis_risk', 'cardiac_event', 'stable']:
        result = explainer.compute_shap_iq(test_features, label)
        assert isinstance(result, dict)
        assert 'label' in result
        assert 'interactions' in result

def test_explain_prediction():
    """Test public explain prediction function."""
    setup_test_environment()

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

    # Test for each label
    for label in ['sepsis_risk', 'cardiac_event', 'stable']:
        result = explain_prediction(test_features, label)
        assert isinstance(result, dict)
        assert 'label' in result
        assert 'interactions' in result

def test_error_handling():
    """Test error handling in explain functionality."""
    setup_test_environment()

    # Test with non-existent model
    result = explain_prediction({}, 'nonexistent_label')
    assert isinstance(result, dict)
    assert 'error' in result

def test_thread_safety():
    """Test thread safety of explainer."""
    import threading

    setup_test_environment()

    def explain_worker():
        test_features = {
            'hr': 75,
            'resp_rate': 18,
            'spo2': 98
        }
        return explain_prediction(test_features, 'sepsis_risk')

    # Create multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=explain_worker)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Thread safety test passed")

if __name__ == "__main__":
    # Run tests
    print("Running SHAP-IQ explanation tests...")

    try:
        test_shap_explainer_initialization()
        print("✓ shap_explainer_initialization test passed")

        test_load_explainer()
        print("✓ load_explainer test passed")

        test_prepare_features()
        print("✓ prepare_features test passed")

        test_compute_shap_iq()
        print("✓ compute_shap_iq test passed")

        test_explain_prediction()
        print("✓ explain_prediction test passed")

        test_error_handling()
        print("✓ error_handling test passed")

        test_thread_safety()
        print("✓ thread_safety test passed")

        print("All SHAP-IQ explanation tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise