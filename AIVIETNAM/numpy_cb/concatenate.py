import numpy as np 

arr1 = np.arange(10).reshape(2, -1)
print('ndarray 1\n', arr1)

arr2 = np.arange(10, 0, -1).reshape(2, -1)
print('ndarray 2\n', arr2)


''' Xếp chồng 2 mảng theo chiều dọc '''

# Cách 1:
out1 = np.concatenate((arr1, arr2), axis=0)
print('out 1\n', out1)

# Cách 2:
out2 = np.r_[arr1, arr2]
print('out 2\n', out2)

# Cách 3:
out3 = np.vstack((arr1, arr2))
print('out 3\n', out3)


''' Xếp chồng 2 mảng theo chiều ngang '''

# Cách 1:
out1 = np.concatenate((arr1, arr2), axis=1)
print('out 1\n', out1)

# Cách 2:
out2 = np.c_[arr1, arr2]
print('out 2\n', out2)

# Cách 3:
out3 = np.hstack((arr1, arr2))
print('out 3\n', out3)