import tensorflow as tf
import numpy as np
from keras.engine import Layer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from keras.models import Sequential, Model
from keras.layers import LSTM, MaxPooling1D, CuDNNLSTM, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv2D, MaxPool2D
from keras.layers import Input, Embedding, Dense, concatenate, Reshape, Flatten, Concatenate, Dropout, Bidirectional
from keras import initializers, regularizers, constraints
from keras.callbacks import *


#from tensorflow.compat.v1.keras.layers import CuDNNLSTM

# Contains various models and different metrics
#
#
max_features = 50 # Number of character level embeddings (SMILEs string has about 45 unique characters)
embed_size = 128 # Embedding size


def rf_train(x_train, y_train):
    '''Build and return a random forest model

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns an sklearn random forest model
    '''
    rf_model = RandomForestClassifier(n_estimators = 10, verbose=2)

    return rf_model


def svm_train(x_train, y_train):
    '''Build and return an svm model

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns an sklearn svm model
    '''
    svm_model = svm.SVC()

    return svm_model


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)



def mlp_train(x_train, y_train):
    '''Build and return a multilayer perceptron model

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras neural network model
    '''

    number_of_features = x_train.shape[1]
    number_of_labels = max(set(y_train))
    y_train = np.reshape(y_train, (-1, 1))
    print('Shape of y_train', y_train.shape)


    model = Sequential()
    model.add(Dense(units=number_of_features, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=number_of_features, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=number_of_features, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=number_of_features//4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=number_of_features//32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=number_of_labels+1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def lstm_train(X_train, y_train):
    '''Build and return a single layer lstm model, optimised for GPU training

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras single layer lstm model
    '''

    number_of_labels = max(set(y_train))
    embedding_dim = 64

    model = Sequential()
    model.add(Embedding(input_dim=200, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    if tf.test.is_gpu_available():
        print("Found GPU - Training with CuDNNLSTM")
        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
    else:
        print("GPU not found : Training with LSTM")
        model.add(Bidirectional(LSTM(64,activation='tanh',dropout=0.2,recurrent_dropout=0.2)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def lstm_2layer_train(X_train, y_train):
    '''Build and return a 2 layer lstm model, optimised for GPU training

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras model
    '''

    number_of_labels = max(set(y_train))
    embedding_dim = 64


    model = Sequential()
    model.add(Embedding(input_dim=46, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    if tf.test.is_gpu_available():
        print("Found GPU - Training with CuDNNLSTM")
        model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
    else:
        print("GPU not found : Training with LSTM")
        model.add(Bidirectional(LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def cnn_lstm_train(X_train, y_train):
    '''Build and return a cnn lstm model, optimised for GPU training

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras model
    '''

    number_of_labels = max(set(y_train))
    embedding_dim = 64


    model = Sequential()
    model.add(Embedding(input_dim=46, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='tanh'))
    model.add(MaxPooling1D(pool_size=8))
    if tf.test.is_gpu_available():
        print("Found GPU - Training with CuDNNLSTM")
        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
    else:
        print("GPU not found : Training with LSTM")
        model.add(Bidirectional(LSTM(64,activation='tanh',dropout=0.2,recurrent_dropout=0.2)))
    model.add(Dense(number_of_labels + 1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def cnn_2layer_lstm(X_train, y_train):
    '''Build and return a model with cnn and 2 lstm layer, optimised for GPU training

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras model
    '''

    number_of_labels = max(set(y_train))
    embedding_dim = 64

    model = Sequential()
    model.add(Embedding(input_dim=46, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='tanh'))
    model.add(MaxPooling1D(pool_size=8))
    if tf.test.is_gpu_available():
        print("Found GPU - Training with CuDNNLSTM")
        model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
    else:
        print("GPU not found : Training with LSTM")
        model.add(Bidirectional(LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def model_cnn(X_train, y_train):
    '''Build and return a CNN text model (using multiple filter sizes), credits: https://www.kaggle.com/yekenot/2dcnn-textclassifier

        Args :
            x_train (numpy.ndarray): Features for training
            y_train (numpy.ndarray): Classification labels for training

        Returns :
            model (object): Returns a Keras model
    '''

    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(X_train.shape[1],))
    x = Embedding(max_features, embed_size)(inp)
    x = Reshape((X_train.shape[1], embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='relu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(X_train.shape[1] - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(max(y_train) + 1, activation="softmax")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model



def model_lstm_du(X_train, y_train):
    '''Build and return a lstm model followed by global max pooling and global average pooling, optimised for GPU
            Args :
                x_train (numpy.ndarray): Features for training
                y_train (numpy.ndarray): Classification labels for training

            Returns :
                model (object): Returns a Keras model
    '''

    inp = Input(shape=(X_train.shape[1],))
    x = Embedding(max_features, embed_size)(inp)

    if tf.test.is_gpu_available():
        print("Found GPU - Training with CuDNNLSTM")
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    else:
        print("GPU not found : Training with LSTM")
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(max(y_train) + 1, activation="softmax")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def lstm_2layer_du(X_train, y_train):
    '''Build and return a 2 layer lstm model followed by global max pooling and global average pooling, optimised for GPU
                Args :
                    x_train (numpy.ndarray): Features for training
                    y_train (numpy.ndarray): Classification labels for training

                Returns :
                    model (object): Returns a Keras model
    '''

    inp = Input(shape=(X_train.shape[1],))
    x = Embedding(max_features, embed_size)(inp)

    if tf.test.is_gpu_available():
        print("Found GPU - Training with CuDNNLSTM")
        x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
    else:
        print("GPU not found : Training with LSTM")
        x = Bidirectional(LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(max(y_train) + 1, activation="softmax")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



class AttentionWithContext(Layer):
    '''
    Credit - #https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras
    #https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/

    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    '''

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def model_lstm_atten(X_train, y_train):
    '''Build and return a 2 layer lstm model followed by attention, optimised for GPU
                Args :
                    x_train (numpy.ndarray): Features for training
                    y_train (numpy.ndarray): Classification labels for training

                Returns :
                    model (object): Returns a Keras model
    '''

    inp = Input(shape=(X_train.shape[1],))
    x = Embedding(max_features, embed_size)(inp)
    x = Dropout(0.2)(x)
    if tf.test.is_gpu_available():
        print("Found GPU - Training with CuDNNLSTM")
        x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
    else:
        print("GPU not found : Training with LSTM")
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(max(y_train) + 1, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def cnn_lstm_atten(X_train, y_train):
    '''Build and return a 2 layer lstm model preceded by a CNN with Maxpooling and followed by attention, optimised for GPU
                Args :
                    x_train (numpy.ndarray): Features for training
                    y_train (numpy.ndarray): Classification labels for training

                Returns :
                    model (object): Returns a Keras model
    '''
    inp = Input(shape=(X_train.shape[1],))
    x = Embedding(max_features, embed_size)(inp)
    x = Dropout(0.2)(x)
    x = Conv1D(64, 5, activation='tanh')(x)
    x = MaxPooling1D(pool_size=8)(x)

    if tf.test.is_gpu_available():
        print("Found GPU - Training with CuDNNLSTM")
        x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
    else:
        print("GPU not found : Training with LSTM")
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(max(y_train) + 1, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def mlp_mol2vec_train(x_train, y_train):
    '''Build and train a multilayer perceptron model for mol2vec feature vectors

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras neural network fit on the input data
    '''

    x_train = np.array(x_train).astype('float')
    print('Data type of train data : ', x_train.dtype)
    y_train = np.array(y_train)#.astype(float)
    number_of_features = x_train.shape[1]
    number_of_labels = max(set(y_train))
    print('Number of features : ', number_of_features)
    print('Number of classification labels : ', len(set(y_train)))
    y_train = np.reshape(y_train, (-1, 1))
    print('Shape of x_train : ', x_train.shape)
    print('Shape of y_train', y_train.shape)
    #print(y_train[:10])

    mlp_model = tf.keras.Sequential([
        tf.keras.layers.Dense(number_of_features, activation = tf.nn.relu),
        tf.keras.layers.Dense(number_of_features*2, activation = tf.nn.relu),
        tf.keras.layers.Dense(number_of_features*2, activation = tf.nn.relu),
        tf.keras.layers.Dense(number_of_features//2, activation = tf.nn.relu),
        tf.keras.layers.Dense(number_of_labels + 1, activation = tf.nn.softmax)
    ])

    opt = tf.keras.optimizers.SGD(lr = 0.1, momentum = 0.9)
    mlp_model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

    history = mlp_model.fit(x_train,
                        y_train,
                        batch_size = 64,
                        epochs = 5,
                        validation_split = 0.1)

    mlp_model.summary()

    return mlp_model


