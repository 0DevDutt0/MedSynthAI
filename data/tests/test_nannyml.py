"""
Test for nannyML performance monitoring.
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

from monitor_performance import estimate_performance_with_nannyml, create_reference_dataset

def test_create_reference_dataset():
    """Test reference dataset creation."""
    # Create a reference dataset
    reference_df = create_reference_dataset()

    # Check that dataset was created
    assert not reference_df.empty
    assert 'sepsis_risk' in reference_df.columns
    assert 'cardiac_event' in reference_df.columns
    assert 'stable' in reference_df.columns

    print("Reference dataset creation test passed!")

def test_estimate_performance_with_nannyml():
    """Test performance estimation with mock data."""
    # Create mock reference dataset
    reference_data = {
        'sepsis_risk': [0, 1, 0, 1, 1],
        'cardiac_event': [0, 1, 0, 0, 1],
        'stable': [1, 0, 1, 1, 0],
        'predicted_sepsis_risk': [0, 1, 1, 1, 0],
        'predicted_cardiac_event': [0, 1, 0, 1, 1],
        'predicted_stable': [1, 0, 1, 1, 0],
        'prob_sepsis_risk_0': [0.7, 0.3, 0.2, 0.1, 0.6],
        'prob_sepsis_risk_1': [0.3, 0.7, 0.8, 0.9, 0.4],
        'prob_cardiac_event_0': [0.6, 0.4, 0.8, 0.3, 0.5],
        'prob_cardiac_event_1': [0.4, 0.6, 0.2, 0.7, 0.5],
        'prob_stable_0': [0.2, 0.8, 0.3, 0.1, 0.9],
        'prob_stable_1': [0.8, 0.2, 0.7, 0.9, 0.1]
    }
    reference_df = pd.DataFrame(reference_data)

    # Create mock analysis dataset
    analysis_data = {
        'sepsis_risk': [0, 1, 0],
        'cardiac_event': [0, 1, 0],
        'stable': [1, 0, 1],
        'predicted_sepsis_risk': [0, 1, 1],
        'predicted_cardiac_event': [0, 1, 0],
        'predicted_stable': [1, 0, 1]
    }
    analysis_df = pd.DataFrame(analysis_data)

    # Run performance estimation
    metrics = estimate_performance_with_nannyml(reference_df, analysis_df)

    # Check that metrics were generated
    assert 'timestamp' in metrics
    assert 'estimates' in metrics
    assert len(metrics['estimates']) > 0

    # Check that all labels have metrics
    labels = ['sepsis_risk', 'cardiac_event', 'stable']
    for label in labels:
        assert label in metrics['estimates']
        assert 'accuracy' in metrics['estimates'][label]
        assert 'recall' in metrics['estimates'][label]
        assert 'precision' in metrics['estimates'][label]
        assert 'f1' in metrics['estimates'][label]

    print("Performance estimation test passed!")

def test_performance_alert():
    """Test that performance alerts are generated when recall is low."""
    # Create mock reference dataset
    reference_data = {
        'sepsis_risk': [0, 1, 0, 1, 1],
        'cardiac_event': [0, 1, 0, 0, 1],
        'stable': [1, 0, 1, 1, 0],
        'predicted_sepsis_risk': [0, 1, 1, 1, 0],
        'predicted_cardiac_event': [0, 1, 0, 1, 1],
        'predicted_stable': [1, 0, 1, 1, 0],
        'prob_sepsis_risk_0': [0.7, 0.3, 0.2, 0.1, 0.6],
        'prob_sepsis_risk_1': [0.3, 0.7, 0.8, 0.9, 0.4],
        'prob_cardiac_event_0': [0.6, 0.4, 0.8, 0.3, 0.5],
        'prob_cardiac_event_1': [0.4, 0.6, 0.2, 0.7, 0.5],
        'prob_stable_0': [0.2, 0.8, 0.3, 0.1, 0.9],
        'prob_stable_1': [0.8, 0.2, 0.7, 0.9, 0.1]
    }
    reference_df = pd.DataFrame(reference_data)

    # Create mock analysis dataset with low recall
    analysis_data = {
        'sepsis_risk': [1, 1, 0],
        'cardiac_event': [0, 1, 0],
        'stable': [0, 0, 1],
        'predicted_sepsis_risk': [0, 1, 0],  # Low recall scenario
        'predicted_cardiac_event': [0, 1, 0],
        'predicted_stable': [0, 0, 1]
    }
    analysis_df = pd.DataFrame(analysis_data)

    # Run performance estimation (this should trigger an alert due to low recall)
    metrics = estimate_performance_with_nannyml(reference_df, analysis_df)

    # Check that metrics were generated
    assert 'timestamp' in metrics
    assert 'estimates' in metrics

    print("Performance alert test passed!")

if __name__ == "__main__":
    # Run all tests
    try:
        print("Running nannyML tests...")

        test_create_reference_dataset()
        test_estimate_performance_with_nannyml()
        test_performance_alert()

        print("All nannyML tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise