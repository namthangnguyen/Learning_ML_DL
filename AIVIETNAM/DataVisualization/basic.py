import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 100)

# Đặt tên
plt.title('Hàm y = x, z = x^2, t = x^3')
plt.xlabel('Trục x')
plt.ylabel('Trục y')

# Hiển thị lưới
plt.grid()

# Vẽ plt.plot(x, y, color, marker, linestyle, linewidth, markersize, label,..)
plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')


# Hiển thị chú thích: plt.legend(loc='upper left', frameon=False)
# loc: ghim chú thích ở phía trên, bên trái (mặc định là cho vào chỗ nào trống)
# frameon: thêm khung cho ghi chú
plt.legend()

plt.show()

# plt.figure(): create a figure
# plt.savefig(): save figure
# plt.clf(): delete current figure (useful when have many figure in a program)

# Draw multiple plots on a figure
# plt.subplot(nrows, ncols, plot_number)
x = [1, 2, 3, 5]
y = [4, 5, 6, 7]

plt.subplot(1, 2, 1)
# plt.plot(x, y, linestyle='--', marker='o', color='b') hoặc viết gọn '--ob' 
plt.plot(x, y, '--ob')

x = np.linspace(-np.pi, np.pi, 30)
y = np.sin(x)
markers_on = [0, 19, 12, 5]

plt.subplot(1, 2, 2)
plt.plot(x, y, '-.rD', markevery=markers_on)

plt.show()
