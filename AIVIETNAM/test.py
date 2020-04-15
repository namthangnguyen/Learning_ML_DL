import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.random.seed(100)
# Tạo dang sách 9 phần tử ngẫu nhiên từ m_train có 60.000 phần tử
m_train = x_train.shape[0]
indices = list(np.random.randint(m_train, size=64))

fig = plt.figure(figsize=(8, 7))
columns = 8
rows = 8

for i in range(1, columns*rows + 1):
    img = x_train[indices[i-1]].reshape(28, 28)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

plt.show()
