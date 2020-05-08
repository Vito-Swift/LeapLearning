import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {"font.family": 'sans-serif',
          'font.serif': 'Times New Roman',
          'font.style': 'italic',
          'font.weight': 'normal',  # or 'blod'
          }
rcParams.update(params)

learning_rate = 0.01
max_iters = 1000


def compute_cost(X, y, w):
    C = X.dot(w) - y
    J2 = (C.T.dot(C)) / (2 * len(X))
    return J2


def gradient_descent(X, y, rate, iters):
    cost = np.zeros(iters)
    w = np.zeros((5, 1))
    for i in range(0, iters):
        w = w - (rate / len(X)) * (X.T.dot(X.dot(w) - y))
        cost[i] = compute_cost(X, y, w)
    return w, cost


def plot_cost(cost):
    fig = plt.figure(figsize=(10, 8))
    x = [i + 1 for i in range(max_iters)]
    plt.plot(x, cost, linewidth=1, marker='^', color='b')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid()
    plt.show()


def regression_result(w, x):
    result = 0
    for i in range(len(w)):
        result += w[i][0] * x[i]
    print("Regression Y =", result)


def main():
    data = np.genfromtxt(fname="NODensity.csv", delimiter=",")

    # normalization
    X, y = np.split(data, [-1], axis=1)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    w, cost = gradient_descent(X, y, learning_rate, max_iters)
    print("Weight:", w)
    print("Final Cost:", cost[-1])

    test_data = np.array([1436.0, 28.0, 68.0, 2.00])
    test_data = (test_data - X_mean) / X_std
    test_one = np.array([1])
    test_data = np.concatenate((test_one, test_data), axis=0)
    print(test_data)
    regression_result(w, test_data)
    plot_cost(cost)


if __name__ == '__main__':
    main()
