import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

np.random.seed(2)

iris = np.genfromtxt('iris_full.csv', delimiter=',', skip_header=1)
X = iris[:,:-1] # (N, 4)
y = iris[:, -1] # (N,)

# Pre-processing data by Sklearn
X = preprocessing.scale(X)
y = y.astype('uint8')


def softmax(Z):
    # Compute softmax values for each sets of scores in Z, 
    # each column of Z is a set of scores.
    # Z: a np array of shape (N, C)
    # C: number of classess
    # return a np array of shape (N, C)
    exp_Z = np.exp(Z)
    A = exp_Z / exp_Z.sum(axis = 1, keepdims = True)
    return A


def loss(X, y, W):
    # W: 2d np array of shape (d, C), each column correspoding to one ouput node
    # X: 2d np array of shape (N, d), each row is one data input
    # y: 1d np array of shape (N,) --- label of each row of X
    A = softmax(X.dot(W)) # predictd output, a np array of shape (N, C)
    id0 = range(X.shape[0]) # A[id0, y]), lấy những node trong A, có y thực bằng 
    loss = -np.mean(np.log(A[id0, y]))
    return loss


def grad(X, y, W):
    # W: 2d np array of shape (d, C), each column correspoding to one ouput node
    # X: 2d np array of shape (N, d), each row is one data input
    # y: 1d np array of shape (N,) --- label of each row of X
    A = softmax(X.dot(W)) # predictd output, a np array of shape (N, C)
    # you must convert y to 2d np array shape (N, C) --- chuyển y => Y ở dạng one-hot
    # to caculate the difference between the predicted output and the actual ouput
    # hoặc tính bằng CT đã được rút gọn
    id0 = range(X.shape[0])
    A[id0, y] -= 1 # A - Y, shape of (N, C)
    return np.dot(X.T, A) / X.shape[0]


def softmax_fit_BGD(X, y, W, eta, epoches):
    loss_hist = [loss(X, y, W)]
    for i in range(epoches):
        dW = grad(X, y, W)
        W = W - eta * dW
        loss_hist.append(loss(X, y, W))
    return loss_hist, W


def pred(X, W):
    # predict output of each columns of X . Class of each x_i is determined by
    # location of max probability. Note that classes are indexed from 0.
    return np.argmax(X.dot(W), axis =1)


C = 3 # number of clasess
Xbar = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
W_init = np.random.randn(Xbar.shape[1], C)

lost_hist, W = softmax_fit_BGD(Xbar, y, W_init, eta = 0.05, epoches = 1000)

plt.plot(range(len(lost_hist)), lost_hist)
plt.show()

# evaluate training set accuracy
predict_class = pred(Xbar, W)
print(predict_class)
print('Training accuracy: %.2f' %(np.mean(predict_class == y)))