# Tìm minimum của Hàm số f(x) = x^2 + 5*sin(x)

import math
import numpy as np 
import matplotlib.pyplot as plt 


# Tính đạo hàm
def grad(x):
    return 2*x + 5*np.cos(x)


# Tính giá trị hàm số
def cost(x):
    return x**2 + 5*np.sin(x)


# Thực hiện thuật toán Gradient descent
def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3: 
            break
        x.append(x_new)
    return (x, it)


# Test với các điểm khởi tạo khác nhau
(x1, it1) = myGD1(.1, - 5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d interations' %(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d interations' %(x2[-1], cost(x2[-1]), it2))
    

        


