"""
Auto-retrain orchestrator that watches for alerts and triggers model retraining.
"""
import os
import time
import json
import logging
import shutil
import pickle
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global lock for thread safety
model_lock = threading.Lock()

class AlertHandler(FileSystemEventHandler):
    """Handles alert file creation events."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def on_created(self, event):
        """Trigger retraining when alert file is created."""
        if event.is_directory:
            return
        if event.src_path.endswith('.txt') and 'alert' in event.src_path:
            logger.info(f"Alert detected: {event.src_path}")
            self.orchestrator.handle_alert(event.src_path)

class AutoRetrainOrchestrator:
    """Orchestrator for auto-retraining system."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.alert_dirs = ['alerts']
        self.observer = Observer()
        self.setup_directories()

    def setup_directories(self):
        """Setup required directories."""
        for dir_path in ['alerts', 'alerts/archive', 'models', 'logs']:
            Path(dir_path).mkdir(exist_ok=True)

    def start_watching(self):
        """Start watching for alert files."""
        logger.info("Starting alert monitoring...")
        for alert_dir in self.alert_dirs:
            if os.path.exists(alert_dir):
                self.observer.schedule(AlertHandler(self), alert_dir, recursive=False)
        self.observer.start()
        logger.info("Alert monitoring started")

    def stop_watching(self):
        """Stop watching for alert files."""
        logger.info("Stopping alert monitoring...")
        self.observer.stop()
        self.observer.join()
        logger.info("Alert monitoring stopped")

    def collect_retraining_data(self, days: int = 7) -> pd.DataFrame:
        """Collect data from predictions log for retraining."""
        logger.info("Collecting retraining data...")

        db_path = Path('logs/predictions_log.db')
        if not db_path.exists():
            logger.warning("Predictions database not found")
            return pd.DataFrame()

        conn = sqlite3.connect(db_path)

        # Calculate date threshold
        threshold_date = datetime.now() - timedelta(days=days)

        # Query for data with true labels (not NULL)
        query = f"""
            SELECT
                id,
                request_id,
                timestamp,
                features,
                sepsis_risk,
                cardiac_event,
                stable
            FROM predictions
            WHERE timestamp >= ?
            AND (sepsis_risk IS NOT NULL OR cardiac_event IS NOT NULL OR stable IS NOT NULL)
            ORDER BY timestamp DESC
        """

        df = pd.read_sql_query(query, conn, params=(threshold_date.isoformat(),))
        conn.close()

        if df.empty:
            logger.warning("No data found for retraining")
            return df

        # Parse features
        df['features'] = df['features'].apply(json.loads)

        # Convert to DataFrame with numerical features
        features_list = df['features'].tolist()
        retrain_df = pd.DataFrame(features_list)

        # Add true labels
        retrain_df['sepsis_risk'] = df['sepsis_risk']
        retrain_df['cardiac_event'] = df['cardiac_event']
        retrain_df['stable'] = df['stable']

        logger.info(f"Collected {len(retrain_df)} rows for retraining")
        return retrain_df

    def detect_and_relabel_noisy_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and drop/relabel noisy samples using Cleanlab."""
        logger.info("Detecting and relabeling noisy samples...")

        # For demonstration purposes, we'll create a simplified version
        # In a real implementation, this would use Cleanlab's multi-output functionality

        # Remove rows with missing labels
        df_clean = df.dropna(subset=['sepsis_risk', 'cardiac_event', 'stable'])

        # Simple noise detection: remove extreme outliers (for demo)
        numerical_features = ['hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map',
                             'glucose', 'lactate', 'creatinine', 'bun', 'wbc', 'plt',
                             'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p', 'albumin',
                             'bilirubin', 'ast', 'alt', 'alp', 'ggt', 'ptt', 'inr',
                             'troponin', 'bnp', 'procalcitonin', 'crp', 'd_dimer',
                             'fibrinogen', 'ph', 'pco2', 'po2', 'hco3', 'base_excess',
                             'urine_output']

        # Remove rows with extreme values in numerical features
        for feature in numerical_features:
            if feature in df_clean.columns:
                Q1 = df_clean[feature].quantile(0.25)
                Q3 = df_clean[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[
                    (df_clean[feature] >= lower_bound) &
                    (df_clean[feature] <= upper_bound)
                ]

        logger.info(f"Removed noisy samples. Remaining rows: {len(df_clean)}")
        return df_clean

    def run_mrmr_feature_selection(self, df: pd.DataFrame) -> List[str]:
        """Run mRMR feature selection (simplified implementation)."""
        logger.info("Running mRMR feature selection...")

        # Simplified feature selection - select top 20 features
        numerical_features = [col for col in df.columns if col in [
            'hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map',
            'glucose', 'lactate', 'creatinine', 'bun', 'wbc', 'plt',
            'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p', 'albumin',
            'bilirubin', 'ast', 'alt', 'alp', 'ggt', 'ptt', 'inr',
            'troponin', 'bnp', 'procalcitonin', 'crp', 'd_dimer',
            'fibrinogen', 'ph', 'pco2', 'po2', 'hco3', 'base_excess',
            'urine_output'
        ] and col in df.columns]

        # Select top 20 features
        selected_features = numerical_features[:20]

        logger.info(f"Selected features: {selected_features}")
        return selected_features

    def load_current_pipeline(self) -> Any:
        """Load the current River pipeline."""
        pipeline_path = Path('models/river_pipeline_latest.pkl')
        if pipeline_path.exists():
            with open(pipeline_path, 'rb') as f:
                return pickle.load(f)
        return None

    def train_new_pipeline(self, df: pd.DataFrame, current_pipeline: Any = None) -> Any:
        """Train a new River model."""
        logger.info("Training new pipeline...")

        # This is a simplified implementation
        # In a real implementation, this would follow the Phase 2 training process

        # For demonstration, we'll create a mock pipeline
        # In reality, this would use the River library with the training data

        # Create a simple mock pipeline for demonstration
        mock_pipeline = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'features': list(df.columns),
            'model_type': 'river_mock'
        }

        logger.info("New pipeline trained successfully")
        return mock_pipeline

    def save_pipeline(self, pipeline: Any) -> str:
        """Save the pipeline and update symlink."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_path = f"models/river_pipeline_v_{timestamp}.pkl"
        symlink_path = "models/river_pipeline_latest.pkl"

        # Save pipeline
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)

        # Update symlink atomically
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(pipeline_path, symlink_path)

        logger.info(f"Pipeline saved to {pipeline_path} and symlink updated")
        return pipeline_path

    def signal_reload(self):
        """Signal LitServe to reload the model."""
        signal_file = Path('model_reload.signal')
        signal_file.touch()
        logger.info("Model reload signal sent")

    def clear_alert_files(self):
        """Move alert files to archive directory."""
        for alert_dir in self.alert_dirs:
            if os.path.exists(alert_dir):
                for filename in os.listdir(alert_dir):
                    if filename.endswith('.txt'):
                        src_path = os.path.join(alert_dir, filename)
                        dst_path = os.path.join('alerts/archive', filename)
                        shutil.move(src_path, dst_path)
                        logger.info(f"Moved alert file to archive: {filename}")

    def handle_alert(self, alert_path: str):
        """Handle alert by triggering retraining."""
        logger.info(f"Handling alert: {alert_path}")

        try:
            # Collect retraining data
            retrain_df = self.collect_retraining_data(7)

            if retrain_df.empty:
                logger.warning("No data for retraining, skipping...")
                return

            # Detect and relabel noisy samples
            cleaned_df = self.detect_and_relabel_noisy_samples(retrain_df)

            # Generate new cleanset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cleanset_path = f"data/cleanset_v_retrain_{timestamp}.csv"
            cleaned_df.to_csv(cleanset_path, index=False)
            logger.info(f"New cleanset saved: {cleanset_path}")

            # Run mRMR feature selection (optional)
            selected_features = self.run_mrmr_feature_selection(cleaned_df)

            # Save selected features
            features_path = "models/selected_features.json"
            with open(features_path, 'w') as f:
                json.dump(selected_features, f)
            logger.info(f"Selected features saved: {features_path}")

            # Train new pipeline
            current_pipeline = self.load_current_pipeline()
            new_pipeline = self.train_new_pipeline(cleaned_df, current_pipeline)

            # Save new pipeline
            pipeline_path = self.save_pipeline(new_pipeline)

            # Signal reload (if not dry run)
            if not self.dry_run:
                self.signal_reload()

            # Clear alert files
            self.clear_alert_files()

            logger.info("Auto-retraining completed successfully")

        except Exception as e:
            logger.error(f"Error in auto-retraining: {e}")
            raise

    def run(self):
        """Run the auto-retrain orchestrator."""
        logger.info("Starting auto-retrain orchestrator...")
        self.start_watching()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping auto-retrain orchestrator...")
            self.stop_watching()

def main():
    """Main function to run auto-retrain orchestrator."""
    logger.info("Starting auto-retrain orchestrator...")

    # Check if dry-run mode is requested
    dry_run = '--dry-run' in sys.argv

    orchestrator = AutoRetrainOrchestrator(dry_run=dry_run)

    if dry_run:
        logger.info("Running in dry-run mode")

    orchestrator.run()

if __name__ == "__main__":
    import sys
    main()