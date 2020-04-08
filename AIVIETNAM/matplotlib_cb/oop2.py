import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.01)
y = x ** 3

# Khởi tạo nhanh Figure và Axes -> plt.subplots()
fig, ax = plt.subplots(figsize=(10, 8))

x = np.linspace(0, 2, 100)
ax.plot(x, x, label='linear')
ax.plot(x, x**2, label='quadratic')
ax.plot(x, x**3, label='cubic')

ax.set_xlabel('Trục x')
ax.set_ylabel('Trục y')
ax.set_title('Hình thứ 2')

ax.legend()

plt.show()
# Lưu figure cùng thư mục với code
# fig.savefig('my_figure.png')
