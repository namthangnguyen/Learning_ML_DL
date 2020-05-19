import numpy as np 

rand_arr = np.random.random((2, 3))
print('array\n', rand_arr)

# set số chữ số sau dấu phẩy trong np
np.set_printoptions(precision=2)
print('array .2f\n', rand_arr)

x, y = np.ogrid[:3, :4]

print(x)
print(y)