import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('BostonHousing.csv')


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

w = np.ones(n)
eta = 0.01  # learning rate
cost_his = []

for i in range(1000):
    y_pred = np.dot(X, w)
    cost = np.sum((y_pred - y) ** 2) / (2 * m)
    dw = np.dot(X.T, y_pred - y) / m
    w = w - eta * dw
    cost_his.append(cost)

plt.plot(range(1000), cost_his)
plt.xlabel('epoch')
plt.ylabel('Giá trị loss')
print(w)
plt.show()
