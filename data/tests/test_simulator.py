"""
Test for stream simulator.
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

def test_create_predictions_db():
    """Test database creation."""
    # Clean up any existing test database
    db_path = Path('logs/test_predictions_log.db')
    if db_path.exists():
        db_path.unlink()

    # Create database
    create_predictions_db()

    # Check if database was created
    assert db_path.exists()

    # Check if table was created
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    table_exists = cursor.fetchone()
    conn.close()

    assert table_exists is not None
    print("Database creation test passed!")

def test_simulate_stream():
    """Test stream simulation with small dataset."""
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
    test_df.to_csv('data/cleanset_v1.csv', index=False)

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

        # Check that true labels were updated
        conn = sqlite3.connect('logs/test_predictions_log.db')
        cursor = conn.cursor()
        cursor.execute("SELECT sepsis_risk, cardiac_event, stable FROM predictions WHERE sepsis_risk IS NOT NULL")
        labels_updated = cursor.fetchall()
        conn.close()

        # At least some labels should be updated (due to the short delay simulation)
        assert len(labels_updated) >= 1, "Some labels should have been updated"

        print("Stream simulator test passed!")

if __name__ == "__main__":
    # Run all tests
    try:
        print("Running stream simulator tests...")

        test_create_predictions_db()
        test_simulate_stream()

        print("All stream simulator tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise