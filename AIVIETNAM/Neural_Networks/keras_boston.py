import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

boston_housings = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housings.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# Calculating the mean and std
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

# Normalize data
train_data = (train_data - mean) / std
tes_data = (test_data - mean) / std

# build model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

# compline
loss = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.RMSprop()
metrics = keras.losses.MeanAbsoluteError()

model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

# model infor
model.summary()

# train model    
history = model.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=0)

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [1000$]')
plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss')
plt.plot(history.epoch, np.array(history.history['val_loss']), label = 'Val loss')
plt.legend()
plt.show()

# forecast
test_predictions = model.predict(test_data)
