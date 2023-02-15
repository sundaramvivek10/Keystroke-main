from flask import Flask, request, render_template
import pickle
import numpy as np

json_string = """
{
    "Model": "RF",
    "HT": {
            "Mean": 27.1,
            "STD": 34.34
            },
    "PPT": {
            "Mean": 100.43,
            "STD": 17.41
            },
    "RRT": {
            "Mean": 184.43,
            "STD": 45.34
            },
    "RPT": {
            "Mean": 162.56,
            "STD": 47.12
            }
}
"""
app = Flask(__name__)

# Load the SVM, RF, and XGB models
svm_model = pickle.load(open("svm.pkl", "rb"))
rf_model = pickle.load(open("rf.pkl", "rb"))
xgb_model = pickle.load(open("xgb.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    alldata = [x for x in request.form.values()]
    model_type = alldata[0]
    init_features = alldata[1:]
    features = np.array(init_features, dtype=object)
    match model_type:
        case "SVM":
            prediction = svm_model.predict([features])[0]
        case "RF":
            prediction = rf_model.predict([features])[0]
        case "XGB":
            # +1 is added here because XGB starts from 0
            prediction = xgb_model.predict([features])[0] + 1
            # -1 was added when fitting the data
        case other:
            return render_template('index.html', prediction_text="Error: Invalid model type")

    return render_template('index.html', prediction_text='The user you are trying to predict is {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
