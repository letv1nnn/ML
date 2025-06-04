import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


train_dataset_file = "datasets/spaceship-titanic/train.csv"
test_dataset_file = "datasets/spaceship-titanic/test.csv"
train_dataset = pd.read_csv(train_dataset_file).dropna()
test_dataset = pd.read_csv(test_dataset_file)

x_train, y_train = train_dataset.iloc[:, :13], train_dataset["Transported"]
x_test = test_dataset.iloc[:, :13]

# Exploring dataset
#print(x_train.info())
#print(x_train)
#print(x_train.columns.tolist())
#print(y_train.head())


# Feature engineering
# we need to convert all non-numerical values to int or float
# and get rid of all useless data like passengerId, because it
# does not influence on the prediction

def feature_engineering(x):
    x = x.copy()
    x["VIP"] = x["VIP"].apply(lambda item: 1.0 if item is True else 0.0)
    x["CryoSleep"] = x["CryoSleep"].apply(lambda item: 1.0 if item is True else 0.0)
    x["Cabin"] = x["Cabin"].apply(lambda item: 1.0 if isinstance(item, str) and item.split("/")[2] == "P" else 0.0)
    x["Destination"] = x["Destination"].apply(
        lambda item: 0.0 if item == 'TRAPPIST-1e' else 1.0 if item == 'PSO J318.5-22' else 2.0)
    x = x.drop(columns=["PassengerId", "HomePlanet", "Name"])
    return x

x_train = feature_engineering(x_train)
x_test = feature_engineering(x_test.fillna(method='ffill'))


# Feature scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Training model
model = LogisticRegression()
model.fit(x_train_scaled, y_train)


# Testing the model
y_pred = model.predict(x_train_scaled)
accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Visualization
def visualize(x: np.array, y: np.array, feature_argument: int):
    plt.title("Spaceship Titanic")
    plt.scatter(x[:, feature_argument], y)
    plt.ylabel("Passenger was lost")
    plt.show()

visualize(x_train_scaled, y_train, 5)

