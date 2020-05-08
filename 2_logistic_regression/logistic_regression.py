import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {"font.family": 'sans-serif',
          'font.serif': 'Times New Roman',
          'font.style': 'italic',
          'font.weight': 'normal',  # or 'blod'
          }
rcParams.update(params)

learning_rate = 0.05
max_iters = 1000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_ascent(X, y, rate, iters):
    w = np.ones((4, 1))
    costs = np.zeros(iters)
    for i in range(iters):
        h = sigmoid(X.dot(w))
        cost = y - h
        w = w + rate * X.T.dot(cost)
        costs[i] = np.average(np.abs(cost))
    return w, costs


def regression_result(w, x, i):
    result = 0
    for i in range(len(w)):
        result += w[i][0] * x[i]
    print("Regression Y =", sigmoid(result))


def plot_cost(cost):
    fig = plt.figure(figsize=(10, 8))
    x = [i + 1 for i in range(max_iters)]
    plt.plot(x, cost, linewidth=1, marker='^', color='b')
    plt.xlabel("Iteration")
    plt.ylabel("Average Cost")
    plt.grid()
    plt.show()


def main():
    data = np.genfromtxt(fname="SmartphoneUser.csv", delimiter=",")

    # normalization
    X, y = np.split(data, [-1], axis=1)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    w, cost = gradient_ascent(X, y, learning_rate, max_iters)
    print("Weight:", w)
    print("Final Cost:", cost[-1])
    plot_cost(cost)

    test_data_1 = np.array([90, 60, 8])
    test_data_2 = np.array([80, 20, 10])
    test_data_1 = (test_data_1 - X_mean) / X_std
    test_data_2 = (test_data_2 - X_mean) / X_std
    test_one = np.array([1])
    test_data_1 = np.concatenate((test_one, test_data_1), axis=0)
    test_data_2 = np.concatenate((test_one, test_data_2), axis=0)
    regression_result(w, test_data_1, 1)
    regression_result(w, test_data_2, 2)


if __name__ == '__main__':
    main()
