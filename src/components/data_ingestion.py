import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join("artifacts", 'train.csv')
    test_data_path = os.path.join("artifacts", 'test.csv')
    raw_data_path = os.path.join("artifacts", 'raw.csv')

# Create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")

        try:
            # Read the dataset
            df = pd.read_csv(r"C:\Credit Card Fraud Detection\notebook\data\credit_card_fraud.csv")
            logging.info("Dataset read as pandas DataFrame")

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            logging.info("Train-test split started")

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error occurred in Data Ingestion: {str(e)}")  # Log the error message
            raise CustomException("Error occurred during data ingestion.", sys)  # Raise the custom exception
        

        