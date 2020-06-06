import numpy as np
np.random.seed(2)


def sigmoid(s):
    return 1/(1 + np.exp(-s))


def logistic_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]
    N, d = Xbar.shape[1], Xbar.shape[0]
    check_w_after = 20
    for count in range(max_count):
        # min data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-1]) < tol: # sao lại check w nhỉ, phải check loss chứ
                    return w
            w.append(w_new)
    return w

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
w_init = np.random.randn(Xbar.shape[0], 1)

w = logistic_regression(Xbar, y, w_init, eta = .05)

print(w[-1])
# đầu ra y được dự đoán theo công thức: y = sigmoid(-4.1 + 1.55*x)
