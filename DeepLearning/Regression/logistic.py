import numpy as np
import matplotlib.pyplot as plt

iris = np.genfromtxt('../datasets/iris_binary.csv', delimiter=',', skip_header=1)
X = iris[:, :-1]
y = iris[:, -1]
N = iris.shape[0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Binary cross entropy
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def predict(X, w):
    z = np.dot(X, w)
    h = sigmoid(z)
    return h.round()


def trainModel(X, y, w_init, eta, epoches):
    cost_his = []
    acc_his = []
    w = w_init
    for i in range(epoches):
        # feed forward
        z = np.dot(X, w)
        h = sigmoid(z)
        # loss function
        cost = loss(h, y)
        # gradient for each weight
        dw = np.dot(X.T, (h - y)) / N
        w = w - eta * dw
        # save cost after 10 epoches
        if (i % 10 == 0):
            cost_his.append(cost)
            # acuuracy
            preds = predict(X, w)
            acc = (preds == y).mean()
            acc_his.append(acc)
    return cost_his, w, acc_his


Xbar = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
w_init = np.random.randn(Xbar.shape[1])
eta = 0.01
epoches = 1500

cost_his, w, acc_his = trainModel(Xbar, y, w_init, eta, epoches)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(cost_his)), cost_his)

plt.subplot(1, 2, 2)
plt.plot(range(len(acc_his)), acc_his)
plt.show()
