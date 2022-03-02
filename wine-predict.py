
import pandas as pd  # import library that performs dataset manipulation in pycharm
dataset = pd.read_csv(r"C:\Users\jemmy\Documents\Skills Bootcamp\Edgehill University\Excel Files\wine.csv")
dataset.to_csv("wine.csv")
print(pd.read_csv("wine.csv"))
pd.set_option('display.width', 1200)
pd.set_option('display.max_columns', 100)
print(dataset)

import sklearn.datasets as dt  # import the dataset folder from library
datasetSKL = dt.load_wine()
print(datasetSKL)
print(datasetSKL.feature_names)

""""The data from pandas is the actual dataset from the csv file whereas the data from sklearn is a sparse matrix 
    representation of the data in the csv file. This allows more for more computation on datasets and reduced 
    computational time"""

# TASK 2
print(dataset.describe())
label = dataset.iloc[:,[-1]]
features = dataset.drop(label.columns, axis=1)
print(label)
print(features)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, label, train_size=0.6)
print(X_train, X_test)
print(Y_train, Y_test)

from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier()
d_tree.fit(X_train, Y_train)
Y_pred = d_tree.predict(X_test)
print(Y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test, Y_pred)
print("Accuracy =", acc*100)
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(Y_test, Y_pred)
plt.xlabel("Expected")
plt.ylabel("Prediction")
plt.show()