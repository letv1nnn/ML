import math
import numpy as np
import matplotlib.pyplot as plt

# Default data, like features and targets
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


def compute_cost(x: np.array, y: np.array, w, b):
    m = x.shape[0]
    cost = sum([(((w * x[i] + b) - y[i]) ** 2) for i in range(m)])
    return cost / (2 * m)


def compute_gradient(x: np.array, y: np.array, w, b):
    m = x.shape[0]
    deriv_J_dw, deriv_J_db = 0, 0
    for i in range(m):
        func_wb = w * x[i] + b
        deriv_J_dw_i = (func_wb - y[i]) * x[i]
        deriv_J_db_i = func_wb - y[i]
        deriv_J_dw += deriv_J_dw_i
        deriv_J_db += deriv_J_db_i
    deriv_J_dw /= m
    deriv_J_db /= m
    return deriv_J_dw, deriv_J_db


def gradient_descent(x: np.array, y: np.array, weight, bias, alpha: float, n_iter, cost_function, gradient_function):
    J_history = []
    p_history = [] # history of parameters w and b
    w, b = weight, bias
    for i in range(n_iter):
        deriv_J_dw, deriv_J_db = gradient_function(x, y, w, b)
        b -= alpha * deriv_J_db
        w -= alpha* deriv_J_dw

        if i < 1000000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        if i % math.ceil(n_iter / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {deriv_J_dw: 0.3e}, dj_db: {deriv_J_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history

def visualization(x: np.array, y: np.array, w, b):
    plt.figure(figsize=(8, 5))

    plt.scatter(x, y, c='r', marker='x', label="Actual prices")

    x_vals = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
    y_vals = w * x_vals + b
    plt.plot(x_vals, y_vals, label="Model prediction", color='blue')

    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


w_init, b_init = 0, 0
iterations = 1000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)

visualization(x_train, y_train, w_final, b_final)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")
