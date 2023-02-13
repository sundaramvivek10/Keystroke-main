import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your data into a pandas dataframe
df = pd.read_csv("AvgStdData.csv")

# Split your data into features (X) and target (y)
X = df.drop(["User ID", "Number"], axis=1)
y = df["User ID"]

# Initialize and fit the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)

# Save the model to a pickle file
with open("rf.pkl", "wb") as f:
    pickle.dump(clf, f)
