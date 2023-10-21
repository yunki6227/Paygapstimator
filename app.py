from flask import Flask, render_template, request
import pickle
from salaryEstimator import full_pipeline

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    age = int(request.form['Age'])
    education = str(request.form['Education'])
    job_title = str(request.form['JobTitle'])
    years = int(request.form['YearsOfExperience'])
    user_input_data_male = {
        'Age':[age],
        'Gender':["Male"],
        'Education Level':[education],
        'Job Title':[job_title],
        'Years of Experience':[years]
    }
    user_input_data_female = {
        'Age':[age],
        'Gender':["Female"],
        'Education Level':[education],
        'Job Title':[job_title],
        'Years of Experience':[years]
    }

    user_input_prepared_male = full_pipeline.transform(user_input_data_male)
    user_input_prepared_female = full_pipeline.transform(user_input_data_female)

    predicted_salary_male = model.predict(user_input_prepared_male)
    predicted_salary_female = model.predict(user_input_prepared_female)



    #prediction = model.predict([[age, education, job_title, years]]) 
    #model output [women, men's salary]
    # women = 1000
    # men = 1000
    return render_template('index.html', prediction_text="The expected salaries for women is ${predicted_salary_female}, and for men: ${predicted_salary_male}")

if __name__ == " __main__":
    app.run()