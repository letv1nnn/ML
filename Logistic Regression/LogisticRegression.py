import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionOneVariable(object):

    def __init__(self, x: np.array, y: np.array, weights = None, bias: float = 0.0, alpha: float = 0.001, iterations: int = 1000):
        self.x = self.normalization(x)
        self.y = y.reshape(-1, 1)
        self.m, self.n = self.x.shape
        self.w = weights if weights is not None else np.zeros((self.n, 1))
        self.b = bias
        self.alpha = alpha
        self.iter = iterations

    def normalization(self, x: np.array):
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        return (x - self.x_mean) / self.x_std

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self):
        z = np.dot(self.x, self.w) + self.b
        h = self.sigmoid(z)
        epsilon = 1e-8
        cost = -np.mean(self.y * np.log(h + epsilon) + (1 - self.y) * np.log(1 - h + epsilon))
        return cost

    def gradient_computation(self):
        z = np.dot(self.x, self.w) + self.b
        h = self.sigmoid(z)
        error = h - self.y
        dw = np.dot(self.x.T, error) / self.m
        db = np.sum(error) / self.m
        return dw, db

    def gradient_descent(self):
        for i in range(self.iter):
            der_w, der_b = self.gradient_computation()
            self.w -= der_w * self.alpha
            self.b -= der_b * self.alpha

            if i % 100 == 0:
                cost = self.cost_function()
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict(self, x: np.array):
        x = (x - self.x_mean) / self.x_std
        z = np.dot(x, self.w) + self.b
        return (z >= 0.5).astype(int)

    def visualization(self):
        plt.scatter(self.x, self.y, c='r', marker='x', label="Training examples")
        x_line = np.linspace(min(self.x), max(self.x), 100).reshape(-1, 1)
        probs = self.sigmoid(np.dot(x_line, self.w) + self.b)
        plt.plot(x_line, probs, color='b', label='Sigmoid boundary')

        plt.xlabel("Feature")
        plt.ylabel("Probability")

        plt.legend()
        plt.grid(True)
        plt.show()


features = np.array([[2], [3], [4], [5], [6], [7], [8]])
labels = np.array([0, 0, 0, 1, 1, 1, 1])
model = LogisticRegressionOneVariable(features, labels, alpha=0.1, iterations=1000)
model.gradient_descent()
model.visualization()

test_input = np.array([[4.5]])
prediction = model.predict(test_input)
print(f"Predicted class for {test_input.flatten()[0]}: {prediction[0][0]}")
