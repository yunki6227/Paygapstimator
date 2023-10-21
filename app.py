from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    age = int(request.form['Age'])
    education = int(request.form['education'])
    job_title = int(request.form['Job title'])
    years = int(request.form['Year of Experience'])
    prediction = model.predict([[age, education, job_title, years]]) 
    model output [women, men's salary]
    women = 1000
    men = 1000
    return render_template('index.html', prediction_text="The expected salaries for women is ${women}, and for men: ${men}")

if __name__ == " __main__":
    app.run()