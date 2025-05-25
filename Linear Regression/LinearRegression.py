import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionOneVariable(object):

    def __init__(self, x: np.array, y: np.array, weight: float = 0.01, bias: float = 0.0, alpha: float = 0.001, iterations: int = 1000):
        self.x = self.normalization(x)
        self.y = y
        self.w = weight
        self.b = bias
        self.alpha = alpha
        self.iterations = iterations
        self.m = self.x.shape[0]

    def normalization(self, x: np.array):
        self.x_mean = np.mean(x)
        self.x_std = np.std(x)
        return (x - self.x_mean) / self.x_std

    def cost_function(self):
        cost = sum([((self.w * self.x[i] + self.b) - self.y[i]) ** 2 for i in range(self.m)])
        return cost / (2 * self.m)

    def gradient_computation(self):
        der_w, der_b = 0.0, 0.0
        for i in range(self.m):
            func = (self.x[i] * self.w + self.b) - self.y[i]
            der_w += func * self.x[i]
            der_b += func
        return der_w / self.m, der_b / self.m

    def gradient_descent(self):
        min_cost = float("inf")
        for i in range(self.iterations):
            der_w, der_b = self.gradient_computation()
            self.w -= self.alpha * der_w
            self.b -= self.alpha * der_b

            if i % 100 == 0:
                cost = self.cost_function()
                if cost < min_cost:
                    min_cost = cost
                else:
                    print("Alpha is to large!")
                    break
                print(f"Cost = {cost}\tw = {self.w}\tb = {self.b}")

    def predict(self, x: float):
        x_normalized = (x - self.x_mean) / self.x_std
        return x_normalized * self.w + self.b

    def visualization(self):
        plt.scatter(self.x, self.y, c="r", marker='x', label="Training examples")
        func = self.w * self.x + self.b
        plt.plot(self.x, func, color="b", label="Model")

        plt.xlabel("Training features")
        plt.ylabel("Training targets")

        plt.legend()
        plt.show()



features = np.array([5, 10, 15, 25, 30, 40, 45])
targets = np.array([15, 20, 25, 35, 40, 50, 60])
lr = LinearRegressionOneVariable(features, targets, alpha=0.015)
lr.gradient_descent()
lr.visualization()

print(f"Predicted y is {lr.predict(17.5)}.")
