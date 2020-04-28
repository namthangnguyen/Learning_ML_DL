"""
Broadcasting là một kĩ thuật cho phép numpy làm việc với các array có shape khác nhau khi thực hiện các phép toán.
"""
import numpy as np

# Cộng vector v với mỗi hàng của ma trận x, kết quả lưu ở ma trận v.
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])

''' Cách thông tường '''
# Tạo một aray có chiều giống x
y = np.empty_like(x)
# Dùng loop để cộng vector v với mỗi hàng của ma trận
for i in range(4):
  y[i, :] = x[i, :] + v
# Kết quả của y
print(y)

''' Numpy broadcasting cho phép chúng ta thực thi tính toán này mà không cần phải làm thêm các bước thêm nào. '''
z = x + v
print(z)