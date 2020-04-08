import numpy as np 

a = np.array([[5, 1, 2],[0, 2, 4],[0, 3, 6]])
b = np.where(a >= 4)
print(b)

c = np.arange(5, 14)
d = np.where(c % 2 == 1)
print(d)

y = np.arange(69, 78).reshape(3, 3)
print('matrix y\n', y)
# Lấy từ ma trận a các phần tử tương ứng thỏa mãn điều kiện, các phần tử còn lại lấy từ ma trận y
# (a, y và ma trận trả về cùng shape)
f = np.where(a < 4, a, y)
print('matrix f = np.where(a < 4, a, y)\n', f)