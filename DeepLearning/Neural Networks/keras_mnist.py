from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load data 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0
N = x_train.shape[0]

# soft max model
sf_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(10, activation='softmax')
])

# compile and train
sf_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

sf_history = sf_model.fit(x_train, y_train, validation_split=0.2, epochs=20, verbose=0)

# MLP model construction
mlp_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

mlp_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

mlp_history = mlp_model.fit(x_train, y_train, validation_split=0.2, epochs=20, verbose=0)

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(sf_history.epoch, np.array(sf_history.history['accuracy']), label='Train accuracy')
plt.plot(sf_history.epoch, np.array(sf_history.history['val_accuracy']), label = 'Val accuracy')
plt.legend()

plt.subplot(122)
plt.plot(mlp_history.epoch, np.array(mlp_history.history['accuracy']), label='Train accuracy')
plt.plot(mlp_history.epoch, np.array(mlp_history.history['val_accuracy']), label = 'Val accuracy')
plt.legend()

plt.show()