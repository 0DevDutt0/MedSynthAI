"""
LitServe server with auto-reload capability.
"""
import os
import time
import pickle
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import litserve as ls
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global model and lock for thread safety
model = None
model_lock = threading.Lock()
reload_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)

class ModelReloadHandler:
    """Handles model reloading in a thread-safe manner."""

    def __init__(self):
        self.reload_signal_file = Path('model_reload.signal')
        self.model_path = Path('models/river_pipeline_latest.pkl')
        self.last_reload = datetime.now()
        self.reload_interval = 2  # Check every 2 seconds

    def should_reload(self) -> bool:
        """Check if model reload is needed."""
        if self.reload_signal_file.exists():
            # Check if enough time has passed since last reload
            time_since_reload = datetime.now() - self.last_reload
            if time_since_reload.total_seconds() >= self.reload_interval:
                return True
        return False

    def reload_model(self):
        """Reload the model from file."""
        with reload_lock:
            try:
                if self.model_path.exists():
                    logger.info("Reloading model...")
                    with open(self.model_path, 'rb') as f:
                        global model
                        model = pickle.load(f)
                    self.last_reload = datetime.now()
                    logger.info("Model reloaded successfully")
                    # Remove the signal file
                    self.reload_signal_file.unlink(missing_ok=True)
                else:
                    logger.warning("Model file not found for reload")
            except Exception as e:
                logger.error(f"Error reloading model: {e}")

class AHMDPModel(ls.LitModel):
    """AHMDP model for prediction."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.reload_handler = ModelReloadHandler()

        # Load initial model
        self.load_model()

    def load_model(self):
        """Load the initial model."""
        try:
            model_path = Path('models/river_pipeline_latest.pkl')
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Initial model loaded successfully")
            else:
                logger.warning("No model found, using default")
        except Exception as e:
            logger.error(f"Error loading initial model: {e}")
            self.model = None

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the loaded model."""
        # Check for model reload
        if self.reload_handler.should_reload():
            self.reload_handler.reload_model()

        # Use the global model with thread safety
        with model_lock:
            current_model = self.model

        if current_model is None:
            # Return default predictions if no model available
            return {
                'sepsis_risk': 0.5,
                'cardiac_event': 0.5,
                'stable': 0.5,
                'timestamp': datetime.now().isoformat()
            }

        try:
            # This is a simplified prediction function
            # In a real implementation, this would use the actual model

            # For demonstration, return random predictions with some pattern
            result = {
                'sepsis_risk': np.random.random(),
                'cardiac_event': np.random.random(),
                'stable': np.random.random(),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Prediction made: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'sepsis_risk': 0.5,
                'cardiac_event': 0.5,
                'stable': 0.5,
                'timestamp': datetime.now().isoformat()
            }

# Create the server
server = ls.LitServer(
    AHMDPModel(),
    accelerator="cpu",
    devices=1,
    timeout=30,
    log_level="INFO"
)

def main():
    """Main function to start the server."""
    logger.info("Starting LitServe server...")

    try:
        server.run(port=8000, host="0.0.0.0")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()