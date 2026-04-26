"""
Alert handling module for monitoring system.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alerts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def write_performance_alert(label: str, metric_value: float, message: str):
    """Write performance alert to file."""
    alert_file = Path('alerts/performance_alert.txt')
    alert_file.parent.mkdir(exist_ok=True)

    with open(alert_file, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {message}\n")
        f.write(f"Label: {label}, Value: {metric_value}\n")
        f.write("-" * 50 + "\n")

    logger.info(f"Performance alert written: {message}")

def write_drift_alert(drift_results: Dict[str, Any]):
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

        logger.info("Drift alert written")

def write_alert_file(alert_file_path: str, message: str):
    """Generic function to write alerts to a file."""
    file_path = Path(alert_file_path)
    file_path.parent.mkdir(exist_ok=True)

    with open(file_path, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {message}\n")
        f.write("-" * 50 + "\n")

    logger.info(f"Alert written to {alert_file_path}")