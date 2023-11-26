import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomException
import pickle
import json
from pymongo.mongo_client import MongoClient


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


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)      
    


def import_data_from_mongodb(collection_name,database_name): 
    try:
        uri = "mongodb+srv://shivam805556:shivam@cluster0.nqainwm.mongodb.net/?retryWrites=true&w=majority"

        # Create a new client and connect to the server
        client = MongoClient(uri) 

        collection=client[database_name][collection_name]

        df=pd.DataFrame(list(collection.find()))

        if "_id" in df.columns.to_list():
            df=df.drop('_id',axis=1,inplace=True) 

        return df
    except Exception as e:
        raise CustomException(e,sys)
        