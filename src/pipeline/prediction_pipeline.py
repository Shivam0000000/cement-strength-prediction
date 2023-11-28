import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(np.array(features))
            preds=model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self,c1:float,c2:float,
                 c3:float,c4:float,c5:float,
                 c6:float,c7:float,c8:int):
        
        self.c1=c1
        self.c2=c2
        self.c3=c3
        self.c4=c4
        self.c5=c5
        self.c6=c6
        self.c7=c7
        self.c8=c8

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={'c1':[self.c1],
                                    'c2':[self.c2],
                                    'c3':[self.c3],
                                    'c4':[self.c4],
                                    'c5':[self.c5],
                                    'c6':[self.c6],
                                    'c7':[self.c7],
                                    'c8':[self.c8]}
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e,sys)



