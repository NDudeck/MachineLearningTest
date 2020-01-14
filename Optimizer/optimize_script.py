import numpy as np
import math
import matplotlib.pyplot as plt
import Optimizer as Op

"""Setting up function and gradient"""

# Goal: Find parameters that minimize error function (experimental - actual)
# Step 1: Create function with noise
n = 9  # sampling rate
m = 3  # Degree of approximation
[a, b] = [0, 1]  # interval
t = np.zeros((n, 1))  # true function
x = np.zeros((n, 1))  # experimental function
x0 = np.linspace(a, b, n)  # sample space
for i in range(0, n):
    t[i] = math.cos(math.pi * 2 * x0[i] / n)
for i in range(0, n):
    x[i] = t[i] + np.random.normal() / 2

plt.plot(x0, t)
plt.plot(x0, x)


# Step 2: Find error function between the approximation and actual
def error_fun(w):
    y = np.zeros((n, 1))
    error = 0
    for i_ef in range(0, n):
        for j_ef in range(0, m):
            y[i_ef] += w[j_ef] * x0[i_ef] ** j_ef
    for i_ef in range(0, n):
        error += (t[i_ef] - y[i_ef]) ** 2
    return .5 * error


# Step 3: Find gradient
def error_grad(w):
    error_gr = np.zeros((m, 1))
    y = np.zeros((n, 1))
    for i_eg in range(0, n):
        for j_eg in range(0, m):
            y[i_eg] += w[j_eg] * x0[i_eg] ** j_eg
    for i_eg in range(0, m):
        for j_eg in range(0, n):
            error_gr[i_eg] += (y[j_eg] - t[j_eg]) * x0[j_eg] ** i_eg
    return error_gr


w0 = np.ones((m, 1))
res = Op.Optimize(error_fun, w0, error_grad)
