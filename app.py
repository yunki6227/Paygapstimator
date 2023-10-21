from flask import Flask, render_template, request
import pickle
import numpy as np 
import pandas as pd
from salaryEstimator import full_pipeline

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=['POST'])
def predict():
    age = float(request.form['Age'])
    education = request.form['Education']
    job_title = request.form['JobTitle']
    years = float(request.form['YearsOfExperience'])
    # user_input_data_male = {
    #     'Age':[age],
    #     'Gender':["Male"],
    #     'Education Level':[education],
    #     'Job Title':[job_title],
    #     'Years of Experience':[years]
    # }
    # user_input_data_female = {
    #     'Age':[age],
    #     'Gender':["Female"],
    #     'Education Level':[education],
    #     'Job Title':[job_title],
    #     'Years of Experience':[years]
    # }
    user_input_data_male = pd.DataFrame({
        'Age': [age],
        'Gender': ["Male"],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [years]
    })
    user_input_data_female = pd.DataFrame({
        'Age': [age],
        'Gender': ["Female"],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [years]
    })

    user_input_prepared_male = full_pipeline.transform(user_input_data_male)  # No need to reshape
    user_input_prepared_female = full_pipeline.transform(user_input_data_female)


    # user_input_prepared_male = full_pipeline.transform(user_input_data_male)
    # user_input_prepared_female = full_pipeline.transform(user_input_data_female)

    predicted_salary_male = model.predict(user_input_prepared_male)
    predicted_salary_female = model.predict(user_input_prepared_female)



    #prediction = model.predict([[age, education, job_title, years]]) 
    #model output [women, men's salary]
    # women = 1000
    # men = 1000
    # return render_template('index.html', prediction_text="The expected salaries for women is ${predicted_salary_female}, and for men: ${predicted_salary_male}")
    return render_template('index.html', prediction_text=f"The expected salaries for women is ${predicted_salary_female[0]:.2f}, and for men: ${predicted_salary_male[0]:.2f}")

if __name__ == "__main__":
    app.run()