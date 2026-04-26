"""
Drift monitoring using Evidently to detect data drift.
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
        logging.FileHandler('logs/monitor_drift.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set deterministic seed for reproducibility
np.random.seed(42)

def create_reference_data() -> pd.DataFrame:
    """Create reference dataset for drift detection."""
    logger.info("Creating reference dataset...")

    # Load the original dataset
    df = pd.read_csv('data/cleanset_v1.csv')

    # Use first 5,000 rows as reference data (numerical features only)
    reference_df = df.head(5000).copy()

    # Select only numerical features for drift detection
    numerical_features = ['hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map',
                         'glucose', 'lactate', 'creatinine', 'bun', 'wbc', 'plt',
                         'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p', 'albumin',
                         'bilirubin', 'ast', 'alt', 'alp', 'ggt', 'ptt', 'inr',
                         'troponin', 'bnp', 'procalcitonin', 'crp', 'd_dimer',
                         'fibrinogen', 'ph', 'pco2', 'po2', 'hco3', 'base_excess',
                         'urine_output', 'sepsis_risk', 'cardiac_event', 'stable']

    reference_df = reference_df[numerical_features].copy()

    # Handle missing values
    reference_df = reference_df.fillna(reference_df.median())

    return reference_df

def load_current_data(limit: int = 1000) -> pd.DataFrame:
    """Load current data from the predictions log."""
    db_path = Path('logs/predictions_log.db')

    if not db_path.exists():
        logger.warning("Predictions database not found")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)

    # Load latest predictions with features
    query = f"""
        SELECT
            id,
            request_id,
            timestamp,
            features
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {limit}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Parse features
    if not df.empty:
        df['features'] = df['features'].apply(json.loads)

    # Convert to DataFrame with numerical features
    if not df.empty:
        features_list = df['features'].tolist()
        # Create a DataFrame from the list of feature dictionaries
        current_df = pd.DataFrame(features_list)

        # Ensure all numerical features are present
        numerical_features = ['hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map',
                             'glucose', 'lactate', 'creatinine', 'bun', 'wbc', 'plt',
                             'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p', 'albumin',
                             'bilirubin', 'ast', 'alt', 'alp', 'ggt', 'ptt', 'inr',
                             'troponin', 'bnp', 'procalcitonin', 'crp', 'd_dimer',
                             'fibrinogen', 'ph', 'pco2', 'po2', 'hco3', 'base_excess',
                             'urine_output']

        # Filter to only numerical features
        current_df = current_df[numerical_features].copy()

        # Handle missing values
        current_df = current_df.fillna(current_df.median())

        return current_df
    else:
        return pd.DataFrame()

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect data drift between reference and current data."""
    logger.info("Detecting data drift...")

    # In a real implementation, we would use Evidently's DataDriftTable
    # For this demo, we'll simulate drift detection using statistical methods

    drift_results = {
        'timestamp': datetime.now().isoformat(),
        'drift_detected': False,
        'drifted_features': [],
        'dataset_drift_score': 0.0,
        'feature_drifts': {}
    }

    if reference_df.empty or current_df.empty:
        logger.warning("Reference or current data is empty")
        return drift_results

    # Calculate statistical differences for each feature
    drifted_features = []
    total_drift_score = 0.0

    for feature in reference_df.columns:
        if feature in current_df.columns:
            ref_series = reference_df[feature].dropna()
            curr_series = current_df[feature].dropna()

            if len(ref_series) > 0 and len(curr_series) > 0:
                # Calculate Kolmogorov-Smirnov test statistic (simplified)
                ref_mean = ref_series.mean()
                curr_mean = curr_series.mean()
                ref_std = ref_series.std()
                curr_std = curr_series.std()

                # Simple drift score based on difference in means relative to standard deviations
                if ref_std > 0:
                    drift_score = abs(ref_mean - curr_mean) / ref_std
                else:
                    drift_score = abs(ref_mean - curr_mean)

                total_drift_score += drift_score

                # If drift score exceeds threshold, consider it drifted
                if drift_score > 1.0:  # Threshold for drift detection
                    drifted_features.append(feature)
                    drift_results['feature_drifts'][feature] = {
                        'score': drift_score,
                        'ref_mean': ref_mean,
                        'curr_mean': curr_mean
                    }

    # Calculate average drift score
    if len(reference_df.columns) > 0:
        drift_results['dataset_drift_score'] = total_drift_score / len(reference_df.columns)

    # Mark if drift is detected
    drift_results['drift_detected'] = len(drifted_features) > 5  # Threshold for dataset drift

    # Store drifted features
    drift_results['drifted_features'] = drifted_features

    logger.info(f"Drift detection completed. {len(drifted_features)} features drifted.")

    return drift_results

def write_drift_report(drift_results: Dict[str, Any]):
    """Write drift report to JSON file."""
    report_file = Path(f'reports/drift_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump(drift_results, f, indent=2)

def write_alert_file(drift_results: Dict[str, Any]):
    """Write drift alert to file."""
    if drift_results['drift_detected']:
        alert_file = Path('alerts/drift_alert.txt')
        alert_file.parent.mkdir(exist_ok=True)

        with open(alert_file, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] Data drift detected\n")
            f.write(f"Number of drifted features: {len(drift_results['drifted_features'])}\n")
            f.write(f"Dataset drift score: {drift_results['dataset_drift_score']:.4f}\n")
            f.write("Drifted features:\n")
            for feature in drift_results['drifted_features']:
                f.write(f"  - {feature}\n")
            f.write("-" * 50 + "\n")

def run_monitoring_loop(interval_hours: int = 1):
    """Run the drift monitoring loop."""
    logger.info("Starting drift monitoring loop...")

    # Create reference data (this would normally be done once)
    reference_df = create_reference_data()

    while True:
        try:
            logger.info("Running drift monitoring cycle...")

            # Load current data
            current_df = load_current_data(1000)

            if current_df.empty:
                logger.warning("No current data found in database")
                time.sleep(interval_hours * 3600)
                continue

            # Detect drift
            drift_results = detect_drift(reference_df, current_df)

            # Write report
            write_drift_report(drift_results)

            # Write alert if needed
            write_alert_file(drift_results)

            logger.info("Drift monitoring cycle completed")

        except Exception as e:
            logger.error(f"Error in drift monitoring: {e}")

        time.sleep(interval_hours * 3600)

def main():
    """Main function to run drift monitoring."""
    logger.info("Starting drift monitoring...")
    run_monitoring_loop(1)  # Run every hour

if __name__ == "__main__":
    main()