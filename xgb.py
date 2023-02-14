from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import pickle

# Load the data into a pandas dataframe
df = pd.read_csv("AvgStdData.csv")

# Split the data into features (X) and target (y)
X = df.drop(["User ID", "Number"], axis=1)
y = df["User ID"] - 1

xgbc = XGBClassifier()

xgbc.fit(X.values, y.values)

pickle.dump(xgbc, open("xgb.pkl", "wb"))