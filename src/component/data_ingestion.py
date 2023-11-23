from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import sys

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join("artifacts","raw.csv")
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            logging.info("Entered in the Data Ingestion Class")

            df=pd.read_csv('notebook/data/cementCleaned.csv')


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train and Test Split")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info("Data Ingestion Completed")

            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e,sys)


