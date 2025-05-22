import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, x: np.array, y: np.array, w=0.0, b=0.0, alpha=0.01, n_iter=1000):
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.alpha = alpha
        self.iterations = n_iter
        self.m = x.shape[0]
        self.cost_history = []

    def cost_function(self):
        y_pred = self.w * self.x + self.b
        cost = np.sum((y_pred - self.y) ** 2) / (2 * self.m)
        return cost

    def compute_gradients(self):
        y_pred = self.w * self.x + self.b
        error = y_pred - self.y
        dj_dw = np.dot(error, self.x) / self.m
        dj_db = np.sum(error) / self.m
        return dj_dw, dj_db

    def gradient_descent(self):
        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradients()
            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db

            if i % 100 == 0:
                cost = self.cost_function()
                self.cost_history.append(cost)
                print(f"Iteration {i}: Cost = {cost:.2f}, w = {self.w:.4f}, b = {self.b:.4f}")

    def predict(self, x_new):
        return self.w * x_new + self.b

    def r_squared(self):
        y_pred = self.w * self.x + self.b
        ss_res = np.sum((self.y - y_pred) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def visualize(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(self.x, self.y, c='red', marker='x', label='Actual Data')
        plt.plot(self.x, self.predict(self.x), color='blue', linewidth=2, label='Regression Line')
        plt.xlabel('Walking Time (minutes)')
        plt.ylabel('Calories Burned')
        plt.title('Calories vs Walking Time')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(0, self.iterations, 100), self.cost_history, 'g-')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Reduction Over Time')
        plt.tight_layout()
        plt.show()


np.random.seed(42)
x = np.array([10, 20, 30, 40, 50, 60])
y = 5 * x + np.random.normal(0, 15, size=x.shape[0])  # True relationship: y = 5x + noise

model = LinearRegression(x, y, alpha=0.01, n_iter=1000)
model.gradient_descent()

print(f"\nFinal Parameters: w = {model.w:.4f}, b = {model.b:.4f}")
print(f"R-squared: {model.r_squared():.4f}")

walk_time = 45
print(f"Predicted calories for {walk_time} min walk: {model.predict(walk_time):.2f}")

model.visualize()
