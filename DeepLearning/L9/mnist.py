import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# 2. Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000, :], y_train[50000:60000]
X_train, y_train = X_train[:50000, :], y_train[:50000]
print(X_train.shape)

# # 3. Reshape lại dữ liệu cho đúng kích thước mà keras yêu cầu
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# # 4. One hot encoding label (Y)
# Y_train = np_utils.to_categorical(y_train, 10)
# Y_val = np_utils.to_categorical(y_val, 10)
# Y_test = np_utils.to_categorical(y_test, 10)
# print('Dữ liệu y ban đầu ', y_train[0])
# print('Dữ liệu y sau one-hot encoding ', Y_train[0])
