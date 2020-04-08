import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data = genfromtxt('advertising.csv', delimiter=',', skip_header=1)

N = data.shape[0]
X = data[:,0:-1]
y = data[:, -1]

''' 
The features in datasets have cùng đơn vị (money) so we can chuẩn hóa all of us at the same time,
mà không nhất thiết phải chuẩn hóa theo từng feature
'''
maxi = np.max(X)
mini = np.min(X)
meani = np.mean(X)
# Chuẩn hóa X
X = (X - meani) / (maxi - mini)


def SGD(X, y, w_init, eta, epoches):
    cost_his = []
    w = w_init
    for epoch in range(epoches):
        for i in range(N):
            # lấy ngẫu nhiên một sample
            rd = np.random.randint(N)
            xi = X[rd]
            yi = y[rd]
            predict = np.dot(xi, w)
            cost = ((predict - yi) ** 2) / 2
            dw = np.dot(xi.T, predict - yi)
            w = w - eta * dw
            cost_his.append(cost)
    
    return cost_his, w


def MiniBGD(X, y, w_init, eta, epoches, miniBatchSize = 20):
    cost_his = []
    w = w_init
    nbatches = int(N / miniBatchSize)
    for epoch in range(epoches):
        mix_data = np.random.permutation(N) # trộn ngẫu nhiên chỉ số của data
        for i in range(0, N, nbatches):
            # lấy n sample trong N samples đã được trộn ngẫu nhiên
            batch_ids = mix_data[i:i + nbatches]
            xi = X[batch_ids]
            yi = y[batch_ids]
            predict = np.dot(xi, w)
            cost = np.sum((predict - yi) ** 2) / (2 * miniBatchSize)
            dw = np.dot(xi.T, predict - yi) / miniBatchSize
            w = w - eta * dw
            cost_his.append(cost)
    
    return cost_his, w


Xbar = np.hstack((np.ones((N, 1)), X))
w_init = np.random.randn(4)
eta = 0.01
epoches = 50

cost_his, w = SGD(Xbar, y, w_init, eta, epoches)
print(w)
# display 500 cost history đẩu tiên
plt.subplot(2, 2, 1)
plt.plot(range(0, 200), cost_his[:200])

cost_his, w = MiniBGD(Xbar, y, w_init, eta, epoches)
print(w)
plt.subplot(2, 2, 2)
plt.plot(range(0, 200), cost_his[:200])
plt.show()
