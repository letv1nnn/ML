import math
import numpy as np
import matplotlib.pyplot as plt
import copy

# this implementation will predict the houses prices
# with the given vector of features

# Features: size, number of bedrooms, number of floors and age of home.
# Target: price

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def compute_cost(X: np.array, y: np.array, w, b):
    m = X.shape[0]
    cost = sum([((np.dot(X[i], w) + b) - y[i]) ** 2 for i in range(0, m)])
    return cost / (2 * m)


def compute_gradient(X: np.array, y: np.array, w, b):
    m = X.shape[0]
    dj_dw, dj_db = np.zeros((X.shape[1], )), 0.
    for i in range(0, m):
        error = ((np.dot(X[i], w) + b) - y[i])
        for j in range(X.shape[1]):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    return dj_db / m, dj_dw / m

'''
testing gradient computations

tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
'''

def gradient_descent(X: np.array, y: np.array, weight: np.ndarray, bias: float, cost, gradient, alpha: float, iterations: int):
    J_history, p_history = [], []
    w, b = copy.deepcopy(weight), bias

    for i in range(0, iterations):
        dj_db, dj_dw = gradient(X, y, w, b)

        b -= alpha * dj_db
        w -= alpha * dj_dw

        if i < 1000000:
            J_history.append(cost(X, y, w, b))
            p_history.append([w, b])

    return w, b, J_history


def visualize():
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");
    ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost');
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step');
    ax2.set_xlabel('iteration step')
    plt.show()


initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent
w_final, b_final, J_hist = gradient_descent(
    X_train, y_train, initial_w, initial_b,
    compute_cost, compute_gradient,
    alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}")


visualize()
