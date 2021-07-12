from flask import Flask , render_template , request , redirect
from flask_cors import CORS,cross_origin
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__,template_folder='C:\\Users\\User\\Desktop\\car\\template')
model=pickle.load(open('C:\\Users\\User\\Desktop\\car\\LinearRegressionModel.pkl','rb'))
car=pd.read_csv('C:\\Users\\User\\Desktop\\car\\final_car_details.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies = sorted(car['company name'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_types= car['fuel'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html' , companies=companies, car_models=car_models, years=year,fuel_types=fuel_types)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    driven=int(request.form.get('kilo_driven'))
    

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(abs(prediction[0]),2))

if __name__=='__main__':
    app.run()
