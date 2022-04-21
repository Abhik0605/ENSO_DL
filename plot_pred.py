from src.model import CNN_model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt


data_path = 'data'
# train_data = np.load(f'{data_path}/tr_data.npy')
# train_data_new = (train_data - train_data.min())/(train_data.max() - train_data.min())
# train_label = np.load(f'{data_path}/tr_label.npy')
# train_label_new = (train_label - train_label.min())/(train_label.max() - train_label.min())

val_data = np.load(f'{data_path}/val_data.npy')
val_data_new = (val_data - val_data.min())/(val_data.max() - val_data.min()) 
val_label = np.load(f'{data_path}/val_label.npy')
# val_label_new = (val_label - val_label.min())/(val_label.max() - val_label.min())
# # print((val_data.shape, val_label.shape, train_data.shape, train_label.shape))


save_path = 'out/model'

new_model = tf.keras.models.load_model(f'{save_path}/my_model.h5')

predictions = new_model.predict(np.expand_dims(val_data_new[-1], axis=0))
predictions = predictions*(val_label.max() - val_label.min()) + val_label.min()
# print(np.expand_dims(val_data[-1], axis=0).shape)
# print(predictions)

plt.plot(predictions[0], linewidth = '2.5', color='black')
plt.ylabel('Nino3.4')
plt.xlabel('Months after 2001')
plt.xticks(np.arange(0, 23, 1.0))
plt.savefig('out/pred.png')