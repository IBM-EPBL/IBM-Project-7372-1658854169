import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "<your key>"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)
model = pickle.load(open('linear_regression_model_sc.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET','post'])
def predict():
	
	GRE_Score = int(request.form['GRE Score'])
	TOEFL_Score = int(request.form['TOEFL Score'])
	University_Rating = int(request.form['University Rating'])
	SOP = float(request.form['SOP'])
	LOR = float(request.form['LOR'])
	CGPA = float(request.form['CGPA'])
	Research = int(request.form['Research'])
	
	# NOTE: manually define and pass the array(s) of values to be scored in the next line
	payload_scoring = {"input_data": [{"field": ['GRE Score', 'TOEFL Score', 'University Rating','SOP','LOR','CGPA','Research'], "values": [GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]}]}
	response_scoring = requests.post('cloudlocation', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
	print("Scoring response")
	print(response_scoring.json())
	final_features = pd.DataFrame([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]])
	
	predict = model.predict(final_features)
	
	output = predict[0]
	
	return render_template('predict.html', Admission_Prediction='Admission chances are {}'.format(output))
	
if __name__ == "__main__":
	app.run(debug=True)
