import pickle
import pandas as pd
from sklearn import svm

# Load the data into a pandas dataframe
df = pd.read_csv("AvgStdData.csv")

# Split the data into features (X) and target (y)
X = df.drop(["User ID", "Number"], axis=1)
y = df["User ID"]

#Train the SVM model on the training data
clf = svm.SVC(kernel="linear", C=1, probability=True)
clf.fit(X, y)

print (X.shape)

# Save the model as a pickle file
filename = 'svm.pkl'
pickle.dump(clf, open(filename, 'wb'))
