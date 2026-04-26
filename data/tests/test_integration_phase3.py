"""
Integration test for Phase 3 monitoring system.
"""
import time
import json
import sqlite3
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path('src').resolve()))

from stream_simulator import simulate_stream, create_predictions_db
from monitor_performance import estimate_performance_with_nannyml, create_reference_dataset
from monitor_drift import detect_drift, create_reference_data

def test_stream_simulator():
    """Test the stream simulator with a small dataset."""
    # Create test database
    db_path = Path('logs/test_predictions_log.db')
    db_path.parent.mkdir(exist_ok=True)

    # Clean up any existing test database
    if db_path.exists():
        db_path.unlink()

    # Create a small test dataset
    test_df = pd.DataFrame({
        'hr': [70, 80, 90],
        'resp_rate': [16, 18, 20],
        'spo2': [98, 97, 96],
        'sepsis_risk': [0, 1, 0],
        'cardiac_event': [0, 1, 0],
        'stable': [1, 0, 1],
        'nursing_notes': ['note1', 'note2', 'note3'],
        'radiology_impression': ['impression1', 'impression2', 'impression3'],
        'patient_history': ['history1', 'history2', 'history3']
    })

    # Save test dataset
    test_df.to_csv('data/test_cleanset_v1.csv', index=False)

    # Mock the requests.post to avoid actual HTTP calls
    with patch('requests.post') as mock_post:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': [
                {
                    'probabilities': {
                        'sepsis_risk_0': 0.3,
                        'sepsis_risk_1': 0.7,
                        'cardiac_event_0': 0.6,
                        'cardiac_event_1': 0.4,
                        'stable_0': 0.2,
                        'stable_1': 0.8
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Run simulator with 3 rows
        simulate_stream(3)

        # Check that database was created and populated
        conn = sqlite3.connect('logs/test_predictions_log.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()

        # Should have 3 entries
        assert count == 3, f"Expected 3 entries, got {count}"

        print("Stream simulator test passed!")

def test_performance_monitoring():
    """Test performance monitoring with mock data."""
    # Create reference dataset
    reference_df = create_reference_dataset()

    # Create mock analysis data
    analysis_data = {
        'sepsis_risk': [0, 1, 0, 1, 1],
        'cardiac_event': [0, 1, 0, 0, 1],
        'stable': [1, 0, 1, 1, 0],
        'predicted_sepsis_risk': [0, 1, 1, 1, 0],
        'predicted_cardiac_event': [0, 1, 0, 1, 1],
        'predicted_stable': [1, 0, 1, 1, 0]
    }
    analysis_df = pd.DataFrame(analysis_data)

    # Run performance estimation
    metrics = estimate_performance_with_nannyml(reference_df, analysis_df)

    # Check that metrics were generated
    assert 'estimates' in metrics
    assert len(metrics['estimates']) > 0

    print("Performance monitoring test passed!")

def test_drift_monitoring():
    """Test drift monitoring with mock data."""
    # Create reference data
    reference_df = create_reference_data()

    # Create mock current data with some drift
    current_data = reference_df.copy()
    # Add some drift to a few features
    current_data['hr'] = current_data['hr'] + 10  # Add drift to heart rate

    # Detect drift
    drift_results = detect_drift(reference_df, current_data)

    # Check that drift detection ran
    assert 'timestamp' in drift_results
    assert 'drift_detected' in drift_results

    print("Drift monitoring test passed!")

def test_integration():
    """Test the integration of all components."""
    # Test that all modules can be imported
    try:
        import stream_simulator
        import monitor_performance
        import monitor_drift
        import alerts
        print("All modules imported successfully!")
    except ImportError as e:
        print(f"Import error: {e}")
        raise

    print("Integration test passed!")

if __name__ == "__main__":
    # Run all tests
    try:
        print("Running Phase 3 integration tests...")

        test_stream_simulator()
        test_performance_monitoring()
        test_drift_monitoring()
        test_integration()

        print("All Phase 3 tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise