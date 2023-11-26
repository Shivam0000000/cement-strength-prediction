
from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri = "mongodb+srv://shivam805556:shivam@cluster0.nqainwm.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# create database name and collection name
DATABASE_NAME='MyDataBase'
COLLECTION_NAME='CementStrength'

# read the data as a dataframe
df=pd.read_csv(r"D:\ML_Projects\cement-strength-prediction\notebook\data\cementCleaned.csv")

json_record=list(json.loads(df.T.to_json()).values())

# Dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)