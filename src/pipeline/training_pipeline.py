from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer




if __name__=='__main__':
    obj=DataIngestion()
    train,test=obj.initiate_data_ingestion()

    obj2=DataTransformation()

    train_arr,test_arr,scaler=obj2.initiate_data_transformation(train,test)

    obj3=ModelTrainer()

    score,model=obj3.initiate_model_trainer(train_arr,test_arr)