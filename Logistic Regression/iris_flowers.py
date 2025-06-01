import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

# Feature Engineering
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
normalizer = Normalizer(norm='l2')
x_train, x_test = normalizer.fit_transform(x_train), normalizer.transform(x_test)

# Training
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# Testing
predicted = model.predict(x_test)
accuracy = accuracy_score(y_test, predicted)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualization
df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")
plt.show()

