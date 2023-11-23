from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import sys

@dataclass
class DataTransformationConfig:
    scaler_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transform=DataTransformationConfig()

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Enter in Data Transformation Class")
            train_set=pd.read_csv(train_path)
            test_set=pd.read_csv(train_path)

            X_train,y_train=train_set.iloc[:,:-1],train_set.iloc[:,-1]
            X_test,y_test=test_set.iloc[:,:-1],test_set.iloc[:,-1]

            logging.info("Fit and transform Dataset by Scaling")
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)
            
            logging.info("Converting train and test set into array")
            train_arr=np.column_stack((X_train,y_train))
            test_arr=np.column_stack((X_test,y_test))

            logging.info("Save preprocessor as a pickle file ")
            save_objects(self.data_transform.scaler_path,scaler)

            logging.info("Data Transformation Completed")

            return train_arr,test_arr,scaler
        except Exception as e:
            raise CustomException(e,sys)
