import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomException
import pickle

def save_objects(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)        

def evaluate_models(X_train,X_test,y_train,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            test_model_score=r2_score(y_pred,y_test)
            report[list(models.keys())[i]]=test_model_score
        return report

    except Exception as e:
        raise CustomException(e,sys)    


        