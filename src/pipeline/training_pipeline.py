import sys

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self) -> None:
        self.data_ingestion = DataIngestion()

        self.data_transformation = DataTransformation()

        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Enter Training Pipeline")
            
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            (
                train_arr,
                test_arr,
                preprocessor_file_path,
            ) = self.data_transformation.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )

            r2_square = self.model_trainer.initiate_model_trainer(train_arr,test_arr)
            print("training completed. Trained model score : ", r2_square)
            logging.info("Training Pipeline Completed")

        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    obj=TrainPipeline()
    obj.run_pipeline()