import numpy as np

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Phép cộng
print(x + y)
print(np.add(x, y))

# Phép trừ
print("Phep tru", x - y)
print(np.subtract(x, y))

# Phép nhân element-wise
print(x * y)
print(np.multiply(x, y))

# Phép chia element-wise
print(x / y)
print(np.divide(x, y))

# Nhân và chia element-wise: tức là chia or nhân từng phần tử của ma trận cho nhau
# Nếu muốn nhân và chia ma trân bình thường ta dùng

v = np.array([9,10])
w = np.array([11, 12])

print("heh", v.shape[0])

# Nhân vô hướng 2 vector
print(v.dot(w))
print(np.dot(v, w))

# Nhân ma trận
print(x.dot(y))
print(np.dot(x, y))

# Tổng phần tử trong vector
print(np.sum(v))
# Tổng tất cả phần tử trong ma trận
print(np.sum(x))
# Tổng phần tử mỗi hàng, prints [4 6]
print(np.sum(x, axis=0))
# Tổng phần tử mỗi cột, prints [3 7]
print(np.sum(x, axis=1))
