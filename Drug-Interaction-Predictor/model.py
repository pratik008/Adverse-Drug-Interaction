from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Bidirectional, CuDNNLSTM
from keras.layers import Dropout

#from tensorflow.compat.v1.keras.layers import CuDNNLSTM

# Contains various models and different metrics
#
#

def rf_train(x_train, y_train):
    '''Build and train a random forest

    Args :
        x_train (numpy.ndarray): Features for training
        y_train (numpy.ndarray): Classification labels for training

    Returns :
        model (object): Returns an sklearn random forest model trained on the input data
    '''
    rf_model = RandomForestClassifier(n_estimators = 100, verbose=2)

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
    model.add(Embedding(input_dim=45, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64,activation='tanh',dropout=0.2,recurrent_dropout=0.2)))
    #model.add(Bidirectional(CuDNNLSTM(128,return_sequences=True)))
    #model.add(Dropout(0.2))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=2)

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
