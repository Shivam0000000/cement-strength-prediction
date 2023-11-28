from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    
    else:
        data=CustomData(c1=request.form.get('Cement (component 1)(kg in a m^3 mixture)'),
                        c2=request.form.get('Blast Furnace Slag (component 2)(kg in a m^3 mixture)'),
                        c3=request.form.get('Fly Ash (component 3)(kg in a m^3 mixture)'),
                        c4=request.form.get('Water  (component 4)(kg in a m^3 mixture)'),
                        c5=request.form.get('Superplasticizer (component 5)(kg in a m^3 mixture)'),
                        c6=request.form.get('Coarse Aggregate  (component 6)(kg in a m^3 mixture)'),
                        c7=request.form.get('Fine Aggregate (component 7)(kg in a m^3 mixture)'),
                        c8=request.form.get('Age (day)')
                        )
        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        return render_template('home.html',results=results[0])


if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)     


