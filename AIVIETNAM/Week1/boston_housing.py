import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Boston_Housing.csv')


def normal(x: list):
    maxi = max(x)
    mini = min(x)
    avg = np.mean(x)
    new = [(i-avg)/(maxi-mini) for i in x]
    return new


df = data.copy()
df = df.apply(normal, axis=0)
Xd = df.drop(columns=['medv'])
Xd.insert(0, 'X0', 1)  # bias

# numpy array format
y = df.medv.values
X = Xd.values

# sample size
m = len(df.index)
n = X.shape[1]


''' Train '''
def BGD(X, y, w_init, eta, epoches):
    cost_his = []
    w = w_init
    for i in range(epoches):
        predict = np.dot(X, w)
        cost = np.sum((predict - y) ** 2) / (2 * m)
        dw = np.dot(X.T, predict - y) / m
        w = w - eta * dw
        cost_his.append(cost)
    return cost_his, w

w_init = np.ones(n)
eta = 0.01  # learning rate
cost_his, w = BGD(X, y, w_init, eta, epoches = 1000)

print(w)
plt.plot(range(1000), cost_his)
plt.show()
