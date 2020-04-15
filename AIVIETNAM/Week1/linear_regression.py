import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data = genfromtxt('Basic_Housing.csv', delimiter=',', skip_header=1)
N = data.shape[0]
X = data[:, 0]
y = data[:, 1]
Xbar = np.hstack((np.ones((N, 1)), X.reshape(-1, 1)))


# use batch gradient descent
def linear_model(X, y, w_init, eta, epoch):
    cost_his = []
    w = w_init

    for i in range(epoch):
        # dự đoán output bằng w hiện tại (feed forward)
        # matrix (N, 2) dot vector (2,) => (N,)
        predict = X.dot(w)

        # tính giá trị lỗi trung bình cho N mẫu dữ liệu
        mean_cost = np.sum((predict - y) ** 2) / (2 * N)

        # tính đạo hàm cho các tham số, (2, N) dot (N,) = (2,)
        dw = X.T.dot(predict - y) / N
        w -= eta * dw

        cost_his.append(mean_cost)

    return cost_his, w


w_init = np.array([0.1, -0.1])
eta = 0.01
epoch = 100

cost_his, w = linear_model(Xbar, y, w_init, eta, epoch)

# cost history
plt.plot(range(epoch), cost_his)
plt.xlabel('epoch')
plt.ylabel('Giá trị loss')
plt.show()

# predict line
plt.scatter(X, y)
plt.xlabel('Diện tích nhà (x 100$m^2$)')
plt.ylabel('Giá nhà (x $10^6$\$)')
plt.plot(X, w[0] + w[1]*X, 'r', label='predict line')
plt.legend()
plt.show()
