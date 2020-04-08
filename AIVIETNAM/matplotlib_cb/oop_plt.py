import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.01)
y = x ** 3

# Khởi tạo figure trống
fig = plt.figure()
fig.suptitle('Hehe')

# Có thể gọi tắt subplot(2,2,1) thành subplot(221)
# fig.add_subplot(221)

'''
add_axes([x, y, axw, axh], facecolor)

Coi góc trái bên dưới là gốc của Figure và Axes 
    (-) x, y thuộc (-inf, inf): là tọa độ góc của Axes (coi weight, height của Figure là 1)
    (-) axw, axh thuộc [0; inf): là tỉ lệ weight và height của Axes với weight và height của Figure
    (-) facecolor (option): is background color
'''

ax = fig.add_axes([0.7, 0.65, 0.25, 0.25], facecolor='#BECAFB')
ax.plot(x, y, 'red')
ax.set_xlabel('Trục x')
ax.set_ylabel('Trục y')
ax.set_title('Hình thứ 1')

# Tạo thêm 1 axes trong cùng 1 figure
ax2 = fig.add_axes([0.1, 0.1, 0.5, 0.5])
x = np.linspace(0, 2, 100)
ax2.plot(x, x, label='linear')
ax2.plot(x, x**2, label='quadratic')
ax2.plot(x, x**3, label='cubic')

# Thêm tiêu đề cho Axes
ax2.set_xlabel('Trục x')
ax2.set_ylabel('Trục y')
ax2.set_title('Hình thứ 2')

ax2.legend()

# Hiển thị figure hiện tại
plt.show()
