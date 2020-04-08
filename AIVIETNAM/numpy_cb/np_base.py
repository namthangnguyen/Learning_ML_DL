import numpy as np

# cho 2 ma trận x, y và hai vector v, w
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

# NOTE: vector trong numpy là vector cột => vector v có n phần tử trong np sẽ là (n, )
print('vector trong np', v.shape)

# Phép cộng
print('Cộng\n', x + y)
print(np.add(x, y))

# Phép trừ
print('Trừ\n', x - y)
print(np.subtract(x, y))

''' Element-wise (hadamard product): chia or nhân từng phần tử của ma trận cho nhau '''
print('Nhân hadamard\n', x * y)
print(np.multiply(x, y))

print('Chia hadamard\n', x / y)
print(np.divide(x, y))

# Nhân vô hướng 2 vector
print('Nhân vô hướng\n', v.dot(w))
print(np.dot(v, w))

# Nhân ma trận
print('Nhân ma trận\n', x.dot(y))
print(np.dot(x, y))

# Tổng phần tử trong vector
print(np.sum(v))
# Tổng tất cả phần tử trong ma trận
print(np.sum(x))
# Tổng phần tử mỗi hàng, prints [4 6]
print(np.sum(x, axis=0))
# Tổng phần tử mỗi cột, prints [3 7]
print(np.sum(x, axis=1))

