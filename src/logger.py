import logging
import os
from datetime import datetime

# Create the artifacts directory if it doesn't exist
log_dir = "artifacts"
os.makedirs(log_dir, exist_ok=True)

# Define log file name with timestamp
log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)