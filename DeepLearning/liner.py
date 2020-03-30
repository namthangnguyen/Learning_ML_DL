import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('Advertising.csv')
X = dataframe.values[:, 2]
y = dataframe.values[:, 4]


# plt.scatter(X, y, marker='o')
# plt.show()


def predict(newX, weight, bias):
    return weight*newX + bias


def cost_func(X, y, weight, bias):
    n = len(X)
    sum_err = 0
    for i in range(n):
        sum_err += (y[i] - (weight*X[i] + bias))**2

    return sum_err/n


def update_weight(X, y, weight, bias, learningRate):
    n = len(X)
    weightTemp = 0.0
    biasTemp = 0.0
    for i in range(n):
        weightTemp += -2*X[i] * (y[i] - (X[i] * weight + bias))
        biasTemp += -2*(y[i] - (X[i] * weight + bias))
    weight -= (weightTemp/n)*learningRate
    bias -= (biasTemp/n)*learningRate

    return weight, bias


def train(X, y, weight, bias, learningRate, iter):
    cosHis = []
    for i in range(iter):
        weight, bias = update_weight(X, y, weight, bias, learningRate)
        cost = cost_func(X, y, weight, bias)
        cosHis.append(cost)

    return weight, bias, cosHis


weight, bias, cosHis = train(X, y, 0.03, 0.0014, 0.001, 30)

print('Ket qua: ')
print(weight)
print(bias)

print('Ket qua du doan: ')
print(predict(19, weight, bias))

iter = [i for i in range(30)]
plt.plot(iter, cosHis)
plt.show()
