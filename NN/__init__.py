import keras
import datetime
import os
import numpy as np
import tensorflow as tf
import warnings




class NN():
    def __init__(self, input_data_size=None, input_data_dim=1,
                 num_encoding_layers=4, size_encoding_layers=64, num_embeddings=16,activation='relu', num_decoding_layers=None,
                 size_decoding_layers=None, learning_rate=3e-5, dropout=0., bidirectional=True, l1_norm=1e-4, layer_type='LSTM',
                 batch_norm=[False, False],scale=True, loss_type='mse',  load_path=None, weights=None, postfix=None):
        """
        costructor model
        to create new model, use the following inputs

            input_data_size: int, mandatory unless loading a model
                size of th einput vector(s) to the recurrent neurons

            input_data_dim: int, defaults to 1
                number of input vectors of size input_data_size for each case

            num_encoding_layers: int, defaults to 4
                number of encoding layers

            size_encoding_layers: int, defaults to 64
                number of nodes in each encoding layer

            num_embeddings: int, defaults to 16
                number of embeddings

            num_decoding_layers: int, defaults to None
                number of encoding layers, if None is set to equal num_encoding_layers

            size_decoding_layers: int, defaults to None
                size of encoding layers, if None is set to equal size_encoding_layers

            learning_rate: float, defaults to 3e5
                learning rate for Adam

            dropout: float, optional
                dopout rate

            bidirectional: bool, optional
                sets the direction of the rucurrent nodes

            l1_norm: bool, optional
                sets the l1 normlaization to enforce sparcity

            layer_type: string or keras layer
                'LSTM' or 'GRU' are supported strings, otherwise a layer from keras.layers. can be added directly

            batch_norm: [bool, bool], or [int, int] optional
                adds batch normalization before and after embedding layer when set to True
            loss_type: string, optional
                the string is passed to self.model.compile(___, loss=loss_type), defaults to 'mse'

        to load a model, use the following inputs

            load_path: string, optional
                path to a stored model, saved as .h5

            weights: string, optional
                path to storew weights, saved as .hdf5

        postfix: string, optional
            A postfix that is added to the file path when the model is created

        """

        if load_path is None:  # build model
            self.loaded = False
            # define id
            id = [str(input_data_size), str(input_data_dim), str(num_encoding_layers), str(size_encoding_layers), str(num_embeddings),
                    str(activation), str(num_decoding_layers), str(learning_rate), str(dropout), str(bidirectional), str(l1_norm),
                    str(int(batch_norm[0])), str(int(batch_norm[1])), str(loss_type), str(scale)]
            self.id = '_'.join(id)
            # set vaules for decoder if not supplied
            if num_decoding_layers is None:
                num_decoding_layers = num_encoding_layers
            if size_decoding_layers is None:
                size_decoding_layers = size_encoding_layers
            # build model
            self.buildModel(num_encoding_layers, size_encoding_layers, num_embeddings,activation,
                            input_data_size, input_data_dim, num_decoding_layers, size_decoding_layers,
                            learning_rate, dropout, bidirectional, l1_norm, layer_type, batch_norm, scale, loss_type)
        else:  # load model
            self.loaded = True
            # load model
            self.model = keras.models.load_model(load_path)
            # attempt to get the model id
            if len(load_path.split('/')) > 2:
                # assumes the path is  load_path='models/self.id/timestring/somemodel.h5' by default
                self.id = load_path.split('/')[-3]
            else:
                self.id = 'RandomAssignedModelID_' + \
                    str(np.random.random()).replace('.0', '')
        if not weights is None:
            self.model.load_weights(weights)
        if not postfix is None:
            self.id = str(self.id)+'_'+postfix
        # timestamp
        self.model_make_time = datetime.datetime.now()
        self.model_make_time_string = '{0}_{1}_{2}_{3}h_{4}m'.format(
            self.model_make_time.month, self.model_make_time.day,
            self.model_make_time.year, self.model_make_time.hour,
            self.model_make_time.minute)
        return

    def buildModel(self, num_encoding_layers, size_encoding_layers, num_embeddings,activation,
                   input_data_size, input_data_dim, num_decoding_layers, size_decoding_layers,
                   learning_rate, dropout, bidirectional, l1_norm, layer_type, batch_norm, scale, loss_type):
        """
        Bulds model using parameters described in the constructor
        """
        if isinstance(layer_type, str):
            if layer_type == 'LSTM':
                layer = keras.layers.LSTM
            elif layer_type == 'GRU':
                layer = keras.layers.GRU
        else:
            layer = layer_type
        if bidirectional:
            wrapper = keras.layers.Bidirectional
        else:
            def wrapper(x): return x

        self.model = keras.models.Sequential()
        # make encoder
        self.model.add(wrapper(layer(size_encoding_layers, return_sequences=(num_encoding_layers > 1)),
                               input_shape=(input_data_size, input_data_dim)))
        if dropout > 0:
            self.model.add(keras.layers.Dropout(dropout))
        for i in range(1, num_encoding_layers):
            self.model.add(wrapper(
                layer(size_encoding_layers, return_sequences=(i < num_encoding_layers - 1))))
            if dropout > 0:
                self.model.add(keras.layers.Dropout(dropout))
        # adds batch normalization prior to embedding layer
        if batch_norm[0]==1:
            self.model.add(keras.layers.BatchNormalization(scale=scale))
        # builds the embedding layer
        if l1_norm == None:
            # embedding layer without l1 regularization
            self.model.add(keras.layers.Dense(
                num_embeddings, activation=activation, name='encoding'))
        else:
            # embedding layer with l1 regularization
            self.model.add(keras.layers.Dense(num_embeddings, activation=activation,
                                              name='encoding', activity_regularizer=keras.regularizers.l1(l1_norm)))#, kernel_constraint=keras.constraints.UnitNorm(axis=0)

        # self.model.add(keras.layers.Lambda(
        #    lambda x: keras.backend.dot(x, tensorflow.nn.top_k(x, 1)[0])))
        # adds batch normalization after embedding layer
        if batch_norm[1]==1:
            self.model.add(keras.layers.BatchNormalization())
        # builds the repeat vector
        self.model.add(keras.layers.RepeatVector(input_data_size))

        # builds the decoding layers
        for i in range(num_decoding_layers):
            self.model.add(
                wrapper(layer(size_decoding_layers, return_sequences=True)))
            if dropout > 0:
                self.model.add(keras.layers.Dropout(dropout))
        # if isinstance(lay1r, keras.layers.recurrent.RNN):
        self.model.add(keras.layers.TimeDistributed(
            keras.layers.Dense(input_data_dim, activation='linear')))
        self.model.compile(keras.optimizers.Adam(
            learning_rate), loss=loss_type)
        return self.model

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def make_encoder(self):
        """
        Makes an encoder section from the network self.model
        """
        # find out where the encoder ends and the decoder starts
        decoderStart = self.getDecoderStart()
        if type(self.model.layers[decoderStart-1]) is keras.layers.BatchNormalization:
            decoderStart = decoderStart-1
        # make first layer with input size input_dim
        input_dim = (self.model.layers[0].input_shape[1],
                     self.model.layers[0].input_shape[2])
        input = keras.models.Input(shape=input_dim)
        encoder_layer = self.model.layers[0](input)
        # make remaining layers
        for i in range(1, decoderStart, 1):
            encoder_layer = self.model.layers[i](encoder_layer)
        # compile to a model
        self.encoder = keras.models.Model(input, encoder_layer)
        return self.encoder

    def make_decoder(self):
        """
        Makes the decoder section from the network self.model
        """
        # find out where the encoder ends and the decoder starts
        decoderStart = self.getDecoderStart()
        if type(self.model.layers[decoderStart-1]) is keras.layers.BatchNormalization:
            decoderStart = decoderStart-1
        # make first layer with input size encoding_dim
        encoding_dim = self.model.layers[decoderStart].input_shape[1]
        encoded_input = keras.models.Input(shape=(encoding_dim,))
        decoder_layer = self.model.layers[decoderStart](encoded_input)
        # make remaining layers
        for i in range(decoderStart+1, len(self.model.layers), 1):
            decoder_layer = self.model.layers[i](decoder_layer)
        # compile to a model
        self.decoder = keras.models.Model(encoded_input, decoder_layer)
        return self.decoder

    def getDecoderStart(self):
        """
            Finds the layer in the model where the decoder start
            This is done my comparing the output shape of each layers
            The final layer with minimum size is assumed to be the encoding layer
            Returns the layer number for the layer after this

            This should be changed to find the layer named 'encoding', rather than
            the layer of minimum size. When I made this I did not know that you
            could name layers
        """
        if not hasattr(self, 'decoderStart'):
            minLayerSize = 1000000
            for i, layer in enumerate(self.model.layers):
                # print(layer.output_shape)
                if len(layer.output_shape) == 2:
                    if layer.output_shape[1] <= minLayerSize:
                        minLayerSize = layer.output_shape[1]
                        self.decoderStart = i+1
        return self.decoderStart

    def get_embeddings(self, data, **kwargs):
        """
        Makes the encoder part of the model, and canlulates the embeddings for the given input

        data: numpy array
            should have shape (num_cases, input_data_size, input_data_dim)
            will also accept shape (num_cases, shape_for_LSTM) in the case where num_input_vectors = 1

        returns: numpy array
            array of shape (num_cases, num_embeddings)
        """
        # check if the encoder has already been created, and if the model has not been trained since
        if (not hasattr(self, 'trainedAfterEncoderCreated')) or self.trainedAfterEncoderCreated:
            self.make_encoder()
            self.trainedAfterEncoderCreated = False
        # predict
        return self.encoder.predict(np.atleast_3d(data), **kwargs)

    def get_decoded(self, embeddings, **kwargs):
        """
        Makes the decoder part of the model, and canlulates the embeddings for the given input

        data: numpy array
            array of shape (num_cases, num_embeddings)

        returns: numpy array
            array of shape (num_cases, input_data_size, input_data_dim)
        """
        # check if the decoder has already been created, and if the model has not been trained since
        if (not hasattr(self, 'trainedAfterDecoderCreated')) or self.trainedAfterDecoderCreated:
            self.make_decoder()
            self.trainedAfterDecoderCreated = False
        # predict
        return self.decoder.predict(embeddings, **kwargs)

    def train(self, data, data_val=None, folder=None,
              batch_size=1800, epochs=25000, seed=42):
        """
        Function which trains the model

        data  : numpy, float
            training data, numpy array of shape (number_of_samples, input_data_size, input_data_dim)
        data_val : numpy, float, or none
            validation data
        folder : string, optional
            folder to save the results, will default to self.id
        batch_size : int, optional
            number of samples in the batch. This is limited by the GPU memory
        epochs : int, optional
            number of epochs to train for
        seed : int, optional
            sets a standard seed for reproducible training
        """
        # mark the model as recently trained,
        # so that the encoder and decoder has to be remade if they are to be used again
        self.trainedAfterEncoderCreated = True
        self.trainedAfterDecoderCreated = True

        # computes the current time to add to filename
        traintime = datetime.datetime.now()
        timestring = '{0}_{1}_{2}_{3}h_{4}m'.format(
            traintime.month, traintime.day, traintime.year, traintime.hour, traintime.minute)

        # sets the folder path
        if folder is None:
            trainid = '_'.join([str(batch_size), str(epochs), str(seed)])
            folder = 'models/'+self.id+'/'+trainid+'_'+timestring+'/'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # write a log file with the training parameters
        with open(folder+'model settings and history', 'w') as file:
            if self.loaded == False:
                file.write('model created at ' +
                           self.model_make_time_string+'\n')
            else:  # if self.loaded==True:
                file.write('model loaded at '+self.model_make_time_string+'\n')
            file.write('model id '+self.id+'\n')
            file.write('batch size '+str(batch_size)+'\n')
            file.write('epochs '+str(epochs)+'\n')
            file.write('seed '+str(seed)+'\n')
            if not hasattr(self, 'traintimes'):
                self.trainTimes = []
                self.completedEpochs = []
            else:
                for time in self.trainTimes:
                    file.write('trained at '+time+'\n')
        # append the time to a list, in case the model is to be trained multiple times
        self.trainTimes.append(timestring)
        self.completedEpochs.append(0)
        # fixes the seed for reproducible training
        np.random.seed(seed)
        # save initial model
        keras.models.save_model(
            self.model, folder + 'start_seed_{0:03d}.h5'.format(seed))
        # sets the file path for saving checkpoints
        if data_val is not None:
            filepath = folder + 'weights.{epoch:06d}-{val_loss:.4f}.hdf5'
            monitor = 'val_loss'
        else:
            filepath = folder + 'weights.{epoch:06d}-{loss:.4f}.hdf5'
            monitor = 'loss'
        # callback for saving checkpoints. Checkpoints are only saved when the model improves
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                                     verbose=0, save_best_only=True,
                                                     save_weights_only=True, mode='min', period=1)
        logger = keras.callbacks.CSVLogger(
            folder + 'log.csv', separator=',', append=True)

        if data_val is not None:
            history = self.model.fit(np.atleast_3d(data), np.atleast_3d(data),
                                     epochs=epochs, batch_size=batch_size,
                                     validation_data=(np.atleast_3d(
                                         data_val), np.atleast_3d(data_val)),
                                     callbacks=[checkpoint, logger])
        else:
            history = self.model.fit(np.atleast_3d(data), np.atleast_3d(data),
                                     epochs=epochs, batch_size=batch_size,
                                     callbacks=[checkpoint, logger])
         # mark how many epochs the model has trained for, in case the model is trained multiple times
        self.completedEpochs[-1] = epochs
