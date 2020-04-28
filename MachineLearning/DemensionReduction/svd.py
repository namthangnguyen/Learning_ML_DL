import numpy as np
from numpy import linalg

m, n = 3, 4
A = np.random.rand(m, n)
U, S, V = linalg.svd(A)

a = np.zeros((m, n))
np.fill_diagonal(a, S)

print('U = \n', U)
print('S = \n', S)
print('V = \n', V)
print('A = \n', A)
print('U*S*V = \n', np.dot(np.dot(U, a), V))
# checking if U, V are orthogonal and S is a diagonal matrix with
# nonnegative decreasing elements
print('Fobenius norm of (UU^T - I) = ', linalg.norm(U.dot(U.T) - np.eye((m))))
print('Fobenius norm of (VV^T - I) = ', linalg.norm(V.dot(V.T) - np.eye((n))))