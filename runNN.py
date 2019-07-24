import NN
import numpy as np
import tensorflow as tf


# os.environ["CUDA_DEVICE_ORDER"]="1,0" # disables GPU to run on CPU
# Used to select graphics card on multiple GPU setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set to allow dynamic allocation of memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

################# set config
layer_type = 'LSTM'
size = 64
encode_layers = 4
# decode_layers = 4 # will be same as encode_layers
embedding = 16
activation='relu'
learning_rate = 3e-5
dropout = 0.2
l1_norm = 2e-5
batch_norm = [1, 1]
batch_size = 450
epochs = 150000
scale=False

postfix = ''
# load data
trainingforML = np.load('prosessedY.npy')
# initialize model
model = NN.NN(input_data_size=trainingforML.shape[-2], input_data_dim=trainingforML.shape[-1],
              num_encoding_layers=encode_layers, size_encoding_layers=size, num_embeddings=embedding,
              learning_rate=learning_rate, dropout=dropout, bidirectional=True, l1_norm=l1_norm, layer_type=layer_type,
              batch_norm=batch_norm,scale=scale, loss_type='mse', postfix=postfix)
# train model
model.train(trainingforML.reshape((-1, trainingforML.shape[-2], trainingforML.shape[-1])),
            data_val=None, folder=None,
            batch_size=batch_size, epochs=epochs, seed=10)
