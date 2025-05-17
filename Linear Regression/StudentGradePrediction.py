import numpy as np
import matplotlib.pyplot as plt


# Grade prediction class is going to take just a training data, at least primarily.
# Additional arguments are weight and bias

class GradePrediction(object):

    def __init__(self, x: np.array, y: np.array, weight: float = 0.001, bias: float = 0, alpha: float = 0.001, iterations: int = 1000):
        self.x = x
        self.y = y
        self.w = weight
        self.b = bias
        self.a = alpha
        self.iter = iterations

        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()

    def cost_calculation(self):
        # J(w, b) = 1/2m * Σ((w * x[i] + b) - y[i]) ^ 2
        m = self.x.shape[0]
        cost = sum([((self.w * self.x[i] + self.b) - self.y[i]) ** 2 for i in range(m)])
        return cost / (2 * m)


    def gradient_calculation(self):
        m = self.x.shape[0]
        dj_dw, dj_db = 0.0, 0.0
        for i in range(m):
            error = (self.w * self.x[i] + self.b) - self.y[i]
            dj_dw += error * self.x[i]
            dj_db += error

        return dj_dw / m, dj_db / m


    def gradient_descent(self):
        for i in range(self.iter):
            dj_dw, dj_db = self.gradient_calculation()
            self.w -= self.a * dj_dw
            self.b -= self.a * dj_db

            if i % 100 == 0:
                cost = self.cost_calculation()
                print(f"Iteration {i}: Cost {cost:.4f}, w = {self.w:.4f}, b = {self.b:.4f}")


    def predict_grade(self, hours: float):
        return self.w * hours + self.b


    def visualization(self):
        plt.scatter(self.x, self.y, c='r', marker='x', label='Data')
        plt.plot(self.x, self.w * self.x + self.b, color='blue', label='Model')
        plt.xlabel("Hours of studying")
        plt.ylabel("Student's grade")
        plt.legend()
        plt.title("Linear Regression Fit")
        plt.show()


x = np.array([10, 25, 30, 40, 50])
y = np.array([20, 47, 55, 88, 95])

x_scaled = (x - x.min()) / (x.max() - x.min())
y_scaled = (y - y.min()) / (y.max() - y.min())

model = GradePrediction(x_scaled, y_scaled, alpha=0.1, iterations=1000)
model.gradient_descent()
model.visualization()

hours = 50
scaled_hours = (hours - x.min()) / (x.max() - x.min())
pred_scaled = model.predict_grade(scaled_hours)
pred = pred_scaled * (y.max() - y.min()) + y.min()
print(f"If a student studied for {hours} h., he probably would attain {pred} grade")
