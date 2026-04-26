"""
Performance monitoring using nannyML to estimate model performance without ground truth.
"""
import time
import json
import logging
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitor_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set deterministic seed for reproducibility
np.random.seed(42)

def load_predictions(limit: int = 1000) -> pd.DataFrame:
    """Load the latest predictions from the database."""
    db_path = Path('logs/predictions_log.db')

    if not db_path.exists():
        logger.warning("Predictions database not found")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)

    # Load latest predictions
    query = f"""
        SELECT
            id,
            request_id,
            timestamp,
            probabilities,
            sepsis_risk,
            cardiac_event,
            stable
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {limit}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Parse probabilities
    if not df.empty:
        df['probabilities'] = df['probabilities'].apply(json.loads)

    return df

def create_reference_dataset() -> pd.DataFrame:
    """Create reference dataset for performance estimation."""
    logger.info("Creating reference dataset...")

    # Load the original dataset
    df = pd.read_csv('data/cleanset_v1.csv')

    # For demonstration purposes, we'll use a subset of the data
    # In a real scenario, this would be the predictions from Phase 1
    reference_df = df.head(1000).copy()

    # Simulate predictions using the model (we'll just use the true labels as predictions for demo)
    reference_df['predicted_sepsis_risk'] = reference_df['sepsis_risk']
    reference_df['predicted_cardiac_event'] = reference_df['cardiac_event']
    reference_df['predicted_stable'] = reference_df['stable']

    # Add probability columns (for demo purposes)
    reference_df['prob_sepsis_risk_0'] = 1.0 - reference_df['sepsis_risk']
    reference_df['prob_sepsis_risk_1'] = reference_df['sepsis_risk']
    reference_df['prob_cardiac_event_0'] = 1.0 - reference_df['cardiac_event']
    reference_df['prob_cardiac_event_1'] = reference_df['cardiac_event']
    reference_df['prob_stable_0'] = 1.0 - reference_df['stable']
    reference_df['prob_stable_1'] = reference_df['stable']

    return reference_df

def estimate_performance_with_nannyml(reference_df: pd.DataFrame, analysis_df: pd.DataFrame):
    """Estimate performance using simulated nannyML approach."""
    logger.info("Estimating performance metrics...")

    # In a real implementation, we would use nannyML's CBPE estimator
    # For this demo, we'll simulate the estimation process

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'estimates': {}
    }

    # Simulate performance estimation for each label
    labels = ['sepsis_risk', 'cardiac_event', 'stable']

    for label in labels:
        # Get true labels from analysis data
        true_labels = analysis_df[label].tolist()

        # Get predicted labels (in a real scenario, these would come from the model)
        predicted_labels = analysis_df[f'predicted_{label}'].tolist()

        # Calculate some basic metrics (simulated)
        if len(true_labels) > 0:
            # Calculate accuracy (simulated)
            accuracy = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) / len(true_labels)

            # Calculate recall (simulated)
            if sum(true_labels) > 0:  # Avoid division by zero
                recall = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1) / sum(true_labels)
            else:
                recall = 0.0

            # Calculate precision (simulated)
            if sum(predicted_labels) > 0:  # Avoid division by zero
                precision = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1) / sum(predicted_labels)
            else:
                precision = 0.0

            # Calculate F1 score (simulated)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            metrics['estimates'][label] = {
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1': f1
            }

            logger.info(f"Estimated metrics for {label}: F1={f1:.4f}, Recall={recall:.4f}")

            # Check for performance degradation
            if recall < 0.85:
                alert_msg = f"Low recall detected for {label}: {recall:.4f}"
                logger.warning(alert_msg)
                write_alert_file(label, recall, alert_msg)
        else:
            logger.warning(f"No data available for {label}")

    return metrics

def write_alert_file(label: str, metric_value: float, message: str):
    """Write alert to file."""
    alert_file = Path('alerts/performance_alert.txt')
    alert_file.parent.mkdir(exist_ok=True)

    with open(alert_file, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {message}\n")
        f.write(f"Label: {label}, Value: {metric_value}\n")
        f.write("-" * 50 + "\n")

def write_metrics_log(metrics: Dict[str, Any]):
    """Write metrics to JSONL log file."""
    log_file = Path('logs/metrics.jsonl')
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')

def run_monitoring_loop(interval_minutes: int = 5):
    """Run the performance monitoring loop."""
    logger.info("Starting performance monitoring loop...")

    # Create reference dataset (this would normally be done once)
    reference_df = create_reference_dataset()

    while True:
        try:
            logger.info("Running performance monitoring cycle...")

            # Load recent predictions
            analysis_df = load_predictions(1000)

            if analysis_df.empty:
                logger.warning("No predictions found in database")
                time.sleep(interval_minutes * 60)
                continue

            # Estimate performance
            metrics = estimate_performance_with_nannyml(reference_df, analysis_df)

            # Write metrics to log
            write_metrics_log(metrics)

            logger.info("Performance monitoring cycle completed")

        except Exception as e:
            logger.error(f"Error in performance monitoring: {e}")

        time.sleep(interval_minutes * 60)

def main():
    """Main function to run performance monitoring."""
    logger.info("Starting performance monitoring...")
    run_monitoring_loop(5)  # Run every 5 minutes

if __name__ == "__main__":
    main()