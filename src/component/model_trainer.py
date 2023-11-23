from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
import os
import sys



@dataclass
class ModelTrainerConfig:

    model_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Enter in the Model Trainer Class")
                
            X_train,y_train,X_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            model=RandomForestRegressor(criterion='absolute_error', min_samples_split=2,
                      n_estimators=200, oob_score=True)
            
            model.fit(X_train,y_train)
            logging.info("Training Model With RandomForestRegressor Completed")

            y_pred=model.predict(X_test)
            logging.info("model Prediction is completed")

            score=r2_score(y_test,y_pred)
            logging.info("Model Evaluation ins Completed")

            save_objects(self.model_config.model_path,model)

            logging.info("Model Training is  completed")

            return score ,model
        except Exception as e:
            raise CustomException(e,sys)
        
        


