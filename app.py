from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your model
model = pickle.load(open('gwp.pkl', 'rb'))

# Route for home page
@app.route("/")
def home():
    return render_template('home.html')

# Route for about page
@app.route("/about")
def about():
    return render_template('about.html')

# Route for predict form
@app.route("/predict")
def predict_form():
    return render_template('predict.html')

# Route for submit result
@app.route("/submit")
def submit_page():
    return render_template('submit.html', prediction_text="Submit page accessed without prediction.")

# Prediction logic
@app.route("/pred", methods=['POST'])
def predict():
    try:
        print("Form submitted!")

        # Collect data
        quarter = int(request.form['quarter'])
        department = int(request.form['department'])
        day = int(request.form['day'])
        team = int(request.form['team'])
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        over_time = int(request.form['over_time'])
        incentive = int(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = int(request.form['idle_men'])
        no_of_style_change = int(request.form['no_of_style_change'])
        no_of_workers = float(request.form['no_of_workers'])
        month = int(request.form['month'])

        features = [[
            quarter, department, day, team,
            targeted_productivity, smv, over_time, incentive,
            idle_time, idle_men, no_of_style_change, no_of_workers, month
        ]]

        print("Input features:", features)

        prediction = model.predict(features)
        pred_value = float(prediction[0])

        print("Prediction value:", pred_value)

        if pred_value <= 0.3:
            result_text = 'The employee is averagely productive.'
        elif pred_value <= 0.8:
            result_text = 'The employee is medium productive.'
        else:
            result_text = 'The employee is highly productive.'

        print("Result text:", result_text)

        return render_template('submit.html', prediction_text=result_text)

    except Exception as e:
        print("Prediction Error:", str(e))
        return render_template('submit.html', prediction_text="An error occurred during prediction.")
    

if __name__ == "__main__":
    app.run(debug=True)
