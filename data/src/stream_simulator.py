"""
Stream simulator that replays data from cleanset_v1.csv as a live stream
and sends requests to the LitServe server.
"""
import time
import json
import random
import sqlite3
import logging
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stream_simulator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set deterministic seed for reproducibility
random.seed(42)
pd.np.random.seed(42)

def create_predictions_db():
    """Create the predictions database if it doesn't exist."""
    db_path = Path('logs/predictions_log.db')
    db_path.parent.mkdir(exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE,
            timestamp TEXT,
            probabilities TEXT,
            features_hash TEXT,
            features TEXT,
            sepsis_risk INTEGER,
            cardiac_event INTEGER,
            stable INTEGER
        )
    ''')

    conn.commit()
    conn.close()

def get_request_payload(row: pd.Series) -> Dict[str, Any]:
    """Construct request payload from a row."""
    # Extract numerical features
    numerical_features = ['hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map',
                         'glucose', 'lactate', 'creatinine', 'bun', 'wbc', 'plt',
                         'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p', 'albumin',
                         'bilirubin', 'ast', 'alt', 'alp', 'ggt', 'ptt', 'inr',
                         'troponin', 'bnp', 'procalcitonin', 'crp', 'd_dimer',
                         'fibrinogen', 'ph', 'pco2', 'po2', 'hco3', 'base_excess',
                         'urine_output']

    # Extract text features
    text_features = ['nursing_notes', 'radiology_impression', 'patient_history']

    # Build payload
    payload = {}

    # Add numerical features
    for feature in numerical_features:
        if feature in row and not pd.isna(row[feature]):
            payload[feature] = float(row[feature])
        else:
            payload[feature] = None

    # Add text features
    for feature in text_features:
        if feature in row and not pd.isna(row[feature]):
            payload[feature] = str(row[feature])
        else:
            payload[feature] = None

    return payload

def send_request(payload: Dict[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
    """Send request to LitServe server with retry logic."""
    url = "http://localhost:8000/predict"

    for attempt in range(3):
        try:
            response = requests.post(url, json={"data": [payload]}, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for request {request_id}: {e}")
            if attempt < 2:  # Don't sleep on the last attempt
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to send request {request_id} after 3 attempts")
                return None

    return None

def update_prediction_labels(request_id: str, labels: Dict[str, int]):
    """Update the prediction row with true labels."""
    db_path = Path('logs/predictions_log.db')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE predictions
        SET sepsis_risk = ?, cardiac_event = ?, stable = ?
        WHERE request_id = ?
    ''', (labels['sepsis_risk'], labels['cardiac_event'], labels['stable'], request_id))

    conn.commit()
    conn.close()

def simulate_stream(max_rows: int = None):
    """Simulate streaming data and send requests to the server."""
    logger.info("Starting stream simulation...")

    # Create database
    create_predictions_db()

    # Load dataset
    df = pd.read_csv('data/cleanset_v1.csv')

    if max_rows:
        df = df.head(max_rows)

    logger.info(f"Processing {len(df)} rows...")

    processed_count = 0
    success_count = 0

    for index, row in df.iterrows():
        # Generate unique request ID
        request_id = f"req_{index}_{int(time.time())}"

        # Create payload
        payload = get_request_payload(row)

        # Send request to server
        response = send_request(payload, request_id)

        if response:
            # Extract probabilities from response
            probabilities = response.get('data', [{}])[0].get('probabilities', {})

            # Calculate features hash for deduplication
            features_str = json.dumps({k: v for k, v in payload.items() if k != 'nursing_notes' and k != 'radiology_impression' and k != 'patient_history'}, sort_keys=True)
            features_hash = hashlib.md5(features_str.encode()).hexdigest()

            # Store prediction in database
            db_path = Path('logs/predictions_log.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            try:
                cursor.execute('''
                    INSERT INTO predictions
                    (request_id, timestamp, probabilities, features_hash, features, sepsis_risk, cardiac_event, stable)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    request_id,
                    datetime.now().isoformat(),
                    json.dumps(probabilities),
                    features_hash,
                    json.dumps({k: v for k, v in payload.items() if not isinstance(v, str)}),  # Store numerical features only
                    row.get('sepsis_risk', None),
                    row.get('cardiac_event', None),
                    row.get('stable', None)
                ))
                conn.commit()
                success_count += 1
            except sqlite3.IntegrityError:
                logger.warning(f"Duplicate request_id {request_id} encountered")
                conn.rollback()
            except Exception as e:
                logger.error(f"Failed to store prediction for request {request_id}: {e}")
                conn.rollback()
            finally:
                conn.close()

            # Simulate delayed label arrival using exponential distribution
            # Mean = 5 seconds, but we'll use a shorter time for simulation
            delay = random.expovariate(1/5) * 0.1  # Scale down for faster simulation
            time.sleep(delay)

            # Update with true labels (this simulates delayed ground truth)
            true_labels = {
                'sepsis_risk': row.get('sepsis_risk', 0),
                'cardiac_event': row.get('cardiac_event', 0),
                'stable': row.get('stable', 0)
            }
            update_prediction_labels(request_id, true_labels)

            processed_count += 1

            # Log progress
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{len(df)} rows")
        else:
            logger.error(f"Failed to get response for row {index}")

    logger.info(f"Stream simulation completed. Processed: {processed_count}, Success: {success_count}")

def main():
    """Main function to run the stream simulator."""
    parser = argparse.ArgumentParser(description='Stream simulator for AHMDP')
    parser.add_argument('--max-rows', type=int, default=None, help='Maximum number of rows to process')

    args = parser.parse_args()

    simulate_stream(args.max_rows)

if __name__ == "__main__":
    main()