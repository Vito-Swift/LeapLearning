import numpy as np

learning_rate = 0.01
max_iters = 10000


def compute_cost(X, y, w):
    _ = np.power(((X @ w.T) - y), 2)
    return np.sum(_) / (2 * len(X))


def gradient_descent(X, y, rate, iters):
    cost = np.zeros(iters)
    w = np.zeros((1, 5))
    for i in range(iters):
        w = w - (rate / len(X)) * np.sum(X * (X @ w.T - y), axis=0)
        cost[i] = compute_cost(X, y, w)
    return w, cost


def main():
    data = np.genfromtxt(fname="NODensity.csv", delimiter=",")

    # normalization
    data = (data - data.mean()) / data.std()

    X, y = np.split(data, [-1], axis=1)
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)

    w, cost = gradient_descent(X, y, learning_rate, max_iters)
    print(cost)


if __name__ == '__main__':
    main()
