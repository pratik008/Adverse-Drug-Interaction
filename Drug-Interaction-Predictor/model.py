from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Bidirectional, CuDNNLSTM, Conv1D, MaxPooling1D
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers
import keras.backend as K
from keras.callbacks import *




#from tensorflow.compat.v1.keras.layers import CuDNNLSTM

# Contains various models and different metrics
#
#
maxlen = 512

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


#https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras
#https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
class AttentionWithContext(Layer):
    """
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
    """

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

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
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


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def rf_train(x_train, y_train):
    '''Build and train a random forest

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns an sklearn random forest model trained on the input data
    '''
    rf_model = RandomForestClassifier(n_estimators = 100, verbose=1)

    rf_model.fit(x_train, y_train)

    return rf_model


def svm_train(x_train, y_train):
    '''Build and train an svm

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns an sklearn svm model fit on the input data
    '''
    svm_model = svm.SVC()

    svm_model.fit(x_train, y_train)

    return svm_model

class my_callback(tf.keras.callbacks.Callback):
    '''Callback class for Keras model'''
    print("Inside my callback")
    pass
    #def on_epoch_end(self, epoch, logs = {}):
        #pass
        #if logs.get('acc') > 0.99:
            #print("\nReached 100% accuracy. Stopping training...")
            #self.model.stop_training = True

def mlp_train(x_train, y_train):
    '''Build and train a multilayer perceptron model

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras neural network fit on the input data
    '''
    #callbacks = my_callback()

    #x_train = np.array(x_train).astype('float')
    print('Data type of train data : ', x_train.dtype)
    #y_train = np.array(y_train)#.astype(float)
    number_of_features = x_train.shape[1]
    number_of_labels = max(set(y_train))
    print('Number of features : ', number_of_features)
    print('Number of classification labels : ', len(set(y_train)))
    print('Shape of y_train', y_train.shape)
    y_train = np.reshape(y_train, (-1, 1))
    print('Shape of x_train : ', x_train.shape)
    print('Shape of y_train', y_train.shape)
    #print(y_train[:10])

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

    history = model.fit(x_train,
                        y_train,
                        batch_size = 64,
                        epochs = 8,
                        validation_split = 0.2
                        ,verbose = 2)
                        #,callbacks = [callbacks])

    print(model.summary())

    return model

def lstm_train(X_train, y_train):
    '''Build and train a multilayer perceptron model

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras neural network fit on the input data
    '''
    #callbacks = my_callback()

    print('Data type of train data : ', X_train.dtype)
    number_of_features = X_train.shape[1]
    number_of_labels = max(set(y_train))
    print('Number of features : ', number_of_features)
    print('Number of classification labels : ', len(set(y_train)))
    #y_train = np.reshape(y_train, (-1, 1))
    print('Shape of x_train : ', X_train.shape)
    print('Shape of y_train', y_train.shape)
    embedding_dim = 64
    #print(y_train[:10])


    print('X_train[1].shape : ',X_train.shape[1])

    model = Sequential()
    model.add(Embedding(input_dim=46, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    if tf.test.is_gpu_available():
        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
    else:
        model.add(Bidirectional(LSTM(64,activation='tanh',dropout=0.2,recurrent_dropout=0.2)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=2)

    return model


def lstm_train_more(X_train, y_train):
    '''Build and train a multilayer perceptron model

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras neural network fit on the input data
    '''
    #callbacks = my_callback()

    print('Data type of train data : ', X_train.dtype)
    number_of_features = X_train.shape[1]
    number_of_labels = max(set(y_train))
    print('Number of features : ', number_of_features)
    print('Number of classification labels : ', len(set(y_train)))
    #y_train = np.reshape(y_train, (-1, 1))
    print('Shape of x_train : ', X_train.shape)
    print('Shape of y_train', y_train.shape)
    embedding_dim = 64
    #print(y_train[:10])


    print('X_train[1].shape : ', X_train.shape[1])

    model = Sequential()
    model.add(Embedding(input_dim=46, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    if tf.test.is_gpu_available():
        model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
    else:
        model.add(Bidirectional(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=2)

    return model


def cnn_lstm_train(X_train, y_train):
    '''Build and train a multilayer perceptron model

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras neural network fit on the input data
    '''
    # callbacks = my_callback()

    print('Data type of train data : ', X_train.dtype)
    number_of_features = X_train.shape[1]
    number_of_labels = max(set(y_train))
    print('Number of features : ', number_of_features)
    print('Number of classification labels : ', len(set(y_train)))
    # y_train = np.reshape(y_train, (-1, 1))
    print('Shape of x_train : ', X_train.shape)
    print('Shape of y_train', y_train.shape)
    embedding_dim = 64
    # print(y_train[:10])

    print('X_train[1].shape : ', X_train.shape[1])

    model = Sequential()
    model.add(Embedding(input_dim=46, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='tanh'))
    model.add(MaxPooling1D(pool_size=8))
    if tf.test.is_gpu_available():
        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
    else:
        model.add(Bidirectional(LSTM(64,activation='tanh',dropout=0.2,recurrent_dropout=0.2)))
    model.add(Dense(number_of_labels + 1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=2)

    return model


# https://www.kaggle.com/yekenot/2dcnn-textclassifier

def model_cnn(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='relu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# BiDirectional LSTM
def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    '''
    Here 64 is the size(dim) of the hidden state vector as well as the output vector. Keeping return_sequence we want the output for the entire sequence. So what is the dimension of output for this layer?
        64*70(maxlen)*2(bidirection concat)
    CuDNNLSTM is fast implementation of LSTM layer in Keras which only runs on GPU
    '''
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model





def mlp_mol2vec_train(x_train, y_train):
    '''Build and train a multilayer perceptron model for mol2vec feature vectors

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns a Keras neural network fit on the input data
    '''
    callbacks = my_callback()

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
        #tf.keras.layers.Dense(number_of_features, activation = tf.nn.relu),
        #tf.keras.layers.Dense(number_of_features, activation = tf.nn.relu),
        #tf.keras.layers.Dense(number_of_features, activation = tf.nn.relu),
        #tf.keras.layers.Dense(number_of_features, activation = tf.nn.relu),
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
                        validation_split = 0.1,
                        callbacks = [callbacks])

    mlp_model.summary()

    return mlp_model


def generate_model_report(model, x_test, y_test):
    '''Get various metrics for testing input model

    Args :
        model (object): Model to use for prediction
        x_test (numpy.ndarray): Data for testing
        y_test (numpy.ndarray): Target classification labels

    Returns :
        accuracy (int): Accuracy score
        precision (int): Precision score
        recall (int): Recall score
        f1 (int): F1 score
    '''
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred, decimals = 0).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    return accuracy, precision, recall, f1


def convert_to_2_class(y_true, y_pred, cls):
    '''Convert multi-class labels to binary labels with respect to a single class

    Args :
        y_true (numpy.ndarray): True classification labels
        y_pred (numpy.ndarray): Predicted classification labels
        cls (int): Class to use as reference for binary classification

    Returns :
        new_y_true (numpy.ndarray): True binary classification labels
        new_y_pred (numpy.ndarray): Predicted binary classification labels
    '''
    new_ytrue = []
    new_ypred = []
    for i in range(len(y_true)):
        if y_true[i] == cls and y_pred[i] == cls:
            new_ytrue.append(1)
            new_ypred.append(1)
        elif y_true[i] == cls and y_pred[i] != cls:
            new_ytrue.append(1)
            new_ypred.append(0)
        elif y_true[i] != cls and y_pred[i] ==cls:
            new_ytrue.append(0)
            new_ypred.append(1)
        elif y_true[i] != cls and y_pred[1] !=cls:
            new_ytrue.append(0)
            new_ypred.append(0)

    return new_ytrue, new_ypred


def generate_model_report_per_class(y_test, y_pred, classes):
    '''Get various metrics calculated classwise for testing model

    Args :
        y_true (numpy.ndarray): True classification labels
        y_pred (numpy.ndarray): Predicted classification labels
        cls (int): Class to use as reference for binary classification

    Returns :
        accuracy_per_class (list): List of classwise accuracy scores
        recall_per_class (list): List of classwise accuracy scores
        precision_per_class (list): List of classwise accuracy scores
        f1score_per_class (list): List of classwise accuracy scores
    '''
    accuracy_per_class = {}
    precision_per_class = {}
    recall_per_class = {}
    f1score_per_class = {}
    for cls in classes:
        new_ytrue, new_ypred = convert_to_2_class(y_test, y_pred, cls)
        accuracy_per_class[cls] = accuracy_score(new_ytrue, new_ypred)
        precision_per_class[cls] = precision_score(new_ytrue, new_ypred)
        recall_per_class [cls] = recall_score(new_ytrue, new_ypred)
        f1score_per_class[cls] = f1_score(new_ytrue, new_ypred)
    return accuracy_per_class, precision_per_class, recall_per_class, f1score_per_class
