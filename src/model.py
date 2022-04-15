import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def CNN_model(n_f, n_d):
	model = models.Sequential()
	model.add(layers.Conv2D(n_f, (8, 4), activation='tanh', padding='same', input_shape=(72, 24, 6)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(n_f, (8, 4), activation='tanh', padding='same'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(n_f, (8, 4), activation='tanh', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dense(n_d, activation='relu'))
	model.add(layers.Dense(23))

	return model