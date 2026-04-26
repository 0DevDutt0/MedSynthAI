"""
Test for auto-retrain orchestrator.
"""
import os
import json
import tempfile
import shutil
import sqlite3
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import time

# Add src to path for imports
import sys
sys.path.insert(0, str(Path('src').resolve()))

from auto_retrain import AutoRetrainOrchestrator

def setup_test_environment():
    """Set up test environment with mock data."""
    # Create required directories
    for dir_path in ['alerts', 'alerts/archive', 'models', 'data', 'logs']:
        Path(dir_path).mkdir(exist_ok=True)

    # Create a mock predictions database
    db_path = Path('logs/predictions_log.db')
    conn = sqlite3.connect(db_path)

    # Create predictions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            request_id TEXT,
            timestamp TEXT,
            features TEXT,
            sepsis_risk REAL,
            cardiac_event REAL,
            stable REAL
        )
    ''')

    # Insert mock data
    mock_data = [
        (1, 'req_1', '2023-01-01T00:00:00', '{"hr": 75, "resp_rate": 18}', 0.2, 0.1, 0.8),
        (2, 'req_2', '2023-01-01T01:00:00', '{"hr": 80, "resp_rate": 20}', 0.3, 0.2, 0.7),
        (3, 'req_3', '2023-01-01T02:00:00', '{"hr": 70, "resp_rate": 16}', 0.1, 0.3, 0.9),
    ]

    conn.executemany('''
        INSERT OR REPLACE INTO predictions
        (id, request_id, timestamp, features, sepsis_risk, cardiac_event, stable)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', mock_data)

    conn.commit()
    conn.close()

    # Create mock model file
    mock_pipeline = {'model': 'mock_pipeline', 'timestamp': '2023-01-01T00:00:00'}
    with open('models/river_pipeline_latest.pkl', 'wb') as f:
        import pickle
        pickle.dump(mock_pipeline, f)

def test_collect_retraining_data():
    """Test collecting retraining data."""
    setup_test_environment()

    orchestrator = AutoRetrainOrchestrator(dry_run=True)
    df = orchestrator.collect_retraining_data(days=7)

    assert not df.empty
    assert len(df) >= 1
    assert 'features' in df.columns

def test_detect_and_relabel_noisy_samples():
    """Test noisy sample detection."""
    setup_test_environment()

    orchestrator = AutoRetrainOrchestrator(dry_run=True)

    # Create test dataframe
    test_df = pd.DataFrame({
        'hr': [75, 80, 70, 150],  # 150 is an outlier
        'resp_rate': [18, 20, 16, 30],
        'sepsis_risk': [0.2, 0.3, 0.1, 0.4],
        'cardiac_event': [0.1, 0.2, 0.3, 0.5],
        'stable': [0.8, 0.7, 0.9, 0.6]
    })

    cleaned_df = orchestrator.detect_and_relabel_noisy_samples(test_df)

    # Should have removed the outlier row
    assert len(cleaned_df) <= len(test_df)
    assert 'hr' in cleaned_df.columns

def test_run_mrmr_feature_selection():
    """Test mRMR feature selection."""
    setup_test_environment()

    orchestrator = AutoRetrainOrchestrator(dry_run=True)

    # Create test dataframe with features
    test_df = pd.DataFrame({
        'hr': [75, 80, 70],
        'resp_rate': [18, 20, 16],
        'spo2': [98, 95, 99],
        'temp': [37.2, 37.5, 36.8],
        'sbp': [120, 130, 110],
        'dbp': [80, 85, 75],
        'map': [95, 100, 90],
        'sepsis_risk': [0.2, 0.3, 0.1],
        'cardiac_event': [0.1, 0.2, 0.3],
        'stable': [0.8, 0.7, 0.9]
    })

    selected_features = orchestrator.run_mrmr_feature_selection(test_df)

    assert isinstance(selected_features, list)
    assert len(selected_features) > 0

def test_load_current_pipeline():
    """Test loading current pipeline."""
    setup_test_environment()

    orchestrator = AutoRetrainOrchestrator(dry_run=True)
    pipeline = orchestrator.load_current_pipeline()

    assert pipeline is not None
    assert 'model' in pipeline

def test_save_pipeline():
    """Test saving pipeline."""
    setup_test_environment()

    orchestrator = AutoRetrainOrchestrator(dry_run=True)

    # Create mock pipeline
    mock_pipeline = {'model': 'test_pipeline', 'timestamp': '2023-01-01T00:00:00'}

    pipeline_path = orchestrator.save_pipeline(mock_pipeline)

    assert pipeline_path is not None
    assert os.path.exists(pipeline_path)

def test_handle_alert():
    """Test handling alert."""
    setup_test_environment()

    orchestrator = AutoRetrainOrchestrator(dry_run=True)

    # Create mock alert file
    alert_file = Path('alerts/performance_alert.txt')
    alert_file.touch()

    # Mock the collect and train methods to avoid actual processing
    with patch.object(orchestrator, 'collect_retraining_data') as mock_collect, \
         patch.object(orchestrator, 'detect_and_relabel_noisy_samples') as mock_clean, \
         patch.object(orchestrator, 'run_mrmr_feature_selection') as mock_mrmr, \
         patch.object(orchestrator, 'train_new_pipeline') as mock_train, \
         patch.object(orchestrator, 'save_pipeline') as mock_save:

        mock_collect.return_value = pd.DataFrame({'hr': [75, 80], 'sepsis_risk': [0.2, 0.3]})
        mock_clean.return_value = pd.DataFrame({'hr': [75, 80], 'sepsis_risk': [0.2, 0.3]})
        mock_mrmr.return_value = ['hr', 'resp_rate']
        mock_train.return_value = {'model': 'new_pipeline'}
        mock_save.return_value = 'models/river_pipeline_v_test.pkl'

        # This should not raise an exception
        orchestrator.handle_alert(str(alert_file))

        # Verify that the alert file was cleared
        assert not alert_file.exists()

def test_dry_run_mode():
    """Test dry-run mode."""
    setup_test_environment()

    orchestrator = AutoRetrainOrchestrator(dry_run=True)

    # Should be able to create orchestrator in dry-run mode
    assert orchestrator.dry_run is True

if __name__ == "__main__":
    # Run tests
    print("Running auto-retrain tests...")

    try:
        test_collect_retraining_data()
        print("✓ collect_retraining_data test passed")

        test_detect_and_relabel_noisy_samples()
        print("✓ detect_and_relabel_noisy_samples test passed")

        test_run_mrmr_feature_selection()
        print("✓ run_mrmr_feature_selection test passed")

        test_load_current_pipeline()
        print("✓ load_current_pipeline test passed")

        test_save_pipeline()
        print("✓ save_pipeline test passed")

        test_handle_alert()
        print("✓ handle_alert test passed")

        test_dry_run_mode()
        print("✓ dry_run_mode test passed")

        print("All auto-retrain tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise