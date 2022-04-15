import argparse
import json
from src.model import CNN_model
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ENSO prediction with Deep Learning')
    parser.add_argument('--param', help = 'hparams', default=f'params.json')
    parser.add_argument('--npydata', help = 'numpy data from data builder', default=f'data')

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
    train_label = np.load(f'{data_path}/tr_label.npy')

    val_data = np.load(f'{data_path}/val_data.npy')
    val_label = np.load(f'{data_path}/val_label.npy')

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
              loss="mse", metrics=["mae", "acc"])

    history = model.fit(train_data, train_label, epochs=epochs, 
                    validation_data=(val_data, val_label), batch_size=batch_size)
