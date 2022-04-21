import argparse
import json
from src.model import CNN_model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ENSO prediction with Deep Learning')
    parser.add_argument('--param', help = 'hparams', default=f'params.json')
    parser.add_argument('--npydata', help = 'numpy data from data builder', default=f'data')
    parser.add_argument('--load_model', help= 'defaul True, set False if training', default='True')

    args = parser.parse_args()

    data_path = args.npydata

    f = open(args.param, "r")
    hparams = json.loads(f.read())

    lr = hparams['exec']['learning_rate']
    epochs = hparams['exec']['num_epochs']
    batch_size = hparams['exec']['batch_size']
    n_f = hparams['exec']['n_f']
    n_d = hparams['exec']['n_d']

    train_data = np.load(f'{data_path}/tr_data.npy')
    train_data_new = (train_data - train_data.min())/(train_data.max() - train_data.min())
    train_label = np.load(f'{data_path}/tr_label.npy')
    train_label_new = (train_label - train_label.min())/(train_label.max() - train_label.min())

    val_data = np.load(f'{data_path}/val_data.npy')
    val_data_new = (val_data - val_data.min())/(val_data.max() - val_data.min()) 
    val_label = np.load(f'{data_path}/val_label.npy')
    val_label_new = (val_label - val_label.min())/(val_label.max() - val_label.min())
    # print((val_data.shape, val_label.shape, train_data.shape, train_label.shape))


    model = CNN_model(n_f, n_d)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    print(model.summary())
    
    model.compile(tf.keras.optimizers.Adam(learning_rate=lr),
              loss="mse")

    save_path = 'out/model'
    if args.load_model == 'True':

      new_model = tf.keras.models.load_model(f'{save_path}/my_model.h5')

      loss, acc, _ = new_model.evaluate(val_data_new, val_label_new, verbose=2)
      print('Restored model, accuracy: {:5.2f}%'.format(loss))

    else:
      history = model.fit(train_data_new, train_label_new, 
                          epochs=10, 
                          validation_data=(val_data_new, val_label_new), 
                          batch_size=batch_size)

      model.save(f'{save_path}/my_model.h5')

      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.xlabel('Epoch')
      plt.ylabel('loss')
      plt.savefig('out/loss.png')


