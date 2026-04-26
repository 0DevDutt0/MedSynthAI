"""
Test for drift monitoring.
"""
import json
import pytest
import pandas as pd
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path('src').resolve()))

from monitor_drift import detect_drift, create_reference_data

def test_create_reference_data():
    """Test reference data creation."""
    # Create reference data
    reference_df = create_reference_data()

    # Check that dataset was created
    assert not reference_df.empty
    assert len(reference_df) >= 100  # Should have at least 100 rows

    # Check that it has numerical features
    numerical_features = ['hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map']
    for feature in numerical_features:
        assert feature in reference_df.columns

    print("Reference data creation test passed!")

def test_detect_drift():
    """Test drift detection with mock data."""
    # Create reference data
    reference_df = create_reference_data()

    # Create mock current data with some drift
    current_data = reference_df.copy()

    # Add drift to a few features
    current_data['hr'] = current_data['hr'] + 10  # Add drift to heart rate
    current_data['temp'] = current_data['temp'] + 2  # Add drift to temperature

    # Detect drift
    drift_results = detect_drift(reference_df, current_data)

    # Check that drift detection ran
    assert 'timestamp' in drift_results
    assert 'drift_detected' in drift_results
    assert 'drifted_features' in drift_results
    assert 'dataset_drift_score' in drift_results

    # Should detect some drift
    assert drift_results['drift_detected'] is True

    print("Drift detection test passed!")

def test_no_drift():
    """Test drift detection with no drift."""
    # Create reference data
    reference_df = create_reference_data()

    # Create current data that's identical to reference
    current_data = reference_df.copy()

    # Detect drift
    drift_results = detect_drift(reference_df, current_data)

    # Check that drift detection ran
    assert 'timestamp' in drift_results
    assert 'drift_detected' in drift_results
    assert 'drifted_features' in drift_results
    assert 'dataset_drift_score' in drift_results

    # Should not detect drift
    assert drift_results['drift_detected'] is False

    print("No drift test passed!")

if __name__ == "__main__":
    # Run all tests
    try:
        print("Running drift tests...")

        test_create_reference_data()
        test_detect_drift()
        test_no_drift()

        print("All drift tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise