import numpy as np

learning_rate = 0.05
max_iters = 10000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_ascent(X, y, rate, iters):
    w = np.ones((1, 4))
    cost = 0
    for i in range(iters):
        h = sigmoid(X * w)
        cost = y - h
        w = w + rate * X * cost
    return w, cost


def main():
    data = np.genfromtxt(fname="SmartphoneUser.csv", delimiter=",")

    # normalization
    X, y = np.split(data, [-1], axis=1)
    X = (X - X.mean()) / X.std()
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    w, cost = gradient_ascent(X, y, learning_rate, max_iters)
    print(cost)


if __name__ == '__main__':
    main()
