from flask import Flask, request
import json
import pickle

json_string = """
{
    "Model": "SVM",
    "HT": {
            "Mean": 27,
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

data = json.loads(json_string)
print (data)

app = Flask(__name__)

# Load the SVM, RF, and XGB models
svm_model = pickle.load(open("svm.pkl", "rb"))
rf_model = pickle.load(open("rf.pkl", "rb"))
#xgb_model = pickle.load(open("xgb.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def predict():
    # Get the input data from the request
    data = json.loads(json_string)
    model_type = data["Model"]
    features = [data["HT"]["Mean"], data["HT"]["STD"],
                data["PPT"]["Mean"], data["PPT"]["STD"],
                data["RRT"]["Mean"], data["RRT"]["STD"],
                data["RPT"]["Mean"], data["RPT"]["STD"]]
    
   # Predict the UserID using the selected model
    if model_type == "SVM":
        prediction = svm_model.predict([features])[0]
    elif model_type == "RF":
        prediction = rf_model.predict([features])[0]
    # elif model_type == "XGB":
    #     prediction = xgb_model.predict([features])[0]
    else:
        return "Error: Invalid model type"
    
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)