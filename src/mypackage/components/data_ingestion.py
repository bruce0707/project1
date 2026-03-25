import os
import sys
from mypackage.logger import logging
from mypackage.exception import CustomException
from dataclasses import dataclass
import pandas as pd
from mypackage.components.data_transformation import DataTransformation
from mypackage.components.data_transformation import DataTransformationConfig
from mypackage.components.model_trainer import ModelTrainer
from mypackage.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig: 
    raw_data_path: str = os.path.join(os.getcwd(),'artifacts','raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started.")
        try:
            df = pd.read_csv('../notebook/data/stud.csv')
            logging.info("Dataset read successfully.")

            # Create the artifacts directory 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("train test split initiated.")

            # Split the data into train and test sets
            from sklearn.model_selection import train_test_split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into train and test sets.")

            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data Ingestion completed successfully.")

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error("Error occurred during data ingestion.")
            raise CustomException(e, sys)   


if __name__ == "__main__":

    from mypackage.components.data_transformation import DataTransformation
    from mypackage.components.data_ingestion import DataIngestion

    #  Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    #  Data Transformation
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    #  Model Training
    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(train_arr, test_arr)

    print("Final Model R2 Score:", r2)