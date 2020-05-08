# Trong trường hợp chưa chỉnh các trục toạ độ
import numpy as np
import matplotlib.pyplot as plt

'''
Axis spine: là đường ranh giới khu vực dữ liệu (khung của plot)
'''

x = np.arange(-10., 30., 0.2)
y = np.sin(x)

# plt.plot(x, y)
# plt.show()

# Chỉnh lại trục toạ độ
fig, ax = plt.subplots(figsize=(10, 6))

# Ẩn đường biên bên trên và bên phải
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# Di chuyển đường bên dưới vào giữa  & ở vị trí y = 0
ax.xaxis.set_ticks_position('bottom') # thấy dòng này ko có tác dụng lắm ?
ax.spines['bottom'].set_position(('data', 0))

# Di chuyển đường bên trái vào giữa & ở vị trí x = 0
ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))

# Hoặc thay ('data', 0) thành 'center',
# NOTE: ('data', 0) là cho khớp vào trục tung, còn ('center') là cho vào giữa Figure
ax.spines['left'].set_position('center')

ax.plot(x, y)
plt.show()
