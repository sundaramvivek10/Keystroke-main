from flask import Flask, request
import json
import pickle

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


@app.route("/", methods=["GET", "POST"])
def predict():
    # Get the input data from the request
    #data = json.loads(json_string)
    #data = request.get_json()
    #data = json.loads(request.data)
    with open("request.json", "r") as f:
        data = json.load(f)

    model_type = data["Model"]
    features = [data["HT"]["Mean"], data["HT"]["STD"],
                data["PPT"]["Mean"], data["PPT"]["STD"],
                data["RRT"]["Mean"], data["RRT"]["STD"],
                data["RPT"]["Mean"], data["RPT"]["STD"]]
    match model_type:
        case "SVM":
            prediction = svm_model.predict([features])[0]
        case "RF":
            prediction = rf_model.predict([features])[0]
        case "XGB":
            prediction = xgb_model.predict([features])[0] + 1 #+1 is added here because XGB starts from 0
            #-1 was added when fitting the data
        case other:
            return "Error: Invalid model type"
    
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)