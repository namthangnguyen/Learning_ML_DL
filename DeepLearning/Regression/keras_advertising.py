import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt

data = np.genfromtxt('../datasets/advertising.csv', delimiter=',', skip_header=1)

m = data.shape[0]
X = data[:, :3]
y = data[:, 3]

# normalize data
X = (X - np.mean(X)) / (np.max(X) - np.min(X))

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=(3,))
])

model.compile(
    optimizer=keras.optimizers.SGD(lr=0.05),
    loss=keras.losses.MeanSquaredError()
)

# model info
model.summary()

history = model.fit(X, y, epochs=500, verbose=0)
print(model.layers[0].weights)

plt.plot(history.history['loss'], color='r')
plt.show()
