import streamlit as st
import pprint
from model import *
from keras.models import load_model
from feature_generation import *
from keras.utils import CustomObjectScope, plot_model
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
import pandas as pd

def process_and_tokenize(smiles):
    # handles both preprocessing and tokenizing for any input type
    if smiles:
        # if string, single input
        if type(smiles) == str:
            # outputs must be packaged into a list
            smiles = [preprocess(smiles)]
            smiles_tokens = [tokenize(smiles[0])]
        else:
            # else, inputs will already be a list
            smiles = [preprocess(i) for i in smiles]
            smiles_tokens = [tokenize(i) for i in smiles]
    else:
        # if input is none or an empty string, assign None
        # handles optional target input
        smiles = None
        smiles_tokens = None

    return (smiles, smiles_tokens)


def canonicalize_smiles(smiles, remove_stereo=True):
    # Converts input SMILES to their canonical form
    # Has the option to remove stereochemistry from SMILES if desired
    # Currently predicting with stereochemistry is not supported, but may be in the future
    mol = Chem.MolFromSmiles(smiles)

    if remove_stereo and '@' in smiles:
        Chem.rdmolops.RemoveStereochemistry(mol)

    if mol is None:
        message = f'''Error: Input string {smiles} failed to convert to a molecular structure.
                         Ensure your SMILES strings are compatible with RDKit.'''
        st.error(message)
    assert mol is not None

    return Chem.MolToSmiles(mol, isomericSmiles=True)


def preprocess(smiles):
    # Function to preprocess a single SMILES text string

    # Loading from a file may add a '\n' or ' ' to the end of a SMILES string
    smiles = smiles.strip('\n')
    smiles = smiles.strip(' ')

    # if spaces are present, they are joined
    # this makes the processing pipeline robust to inputs that are already tokenized,
    # partially tokenized, tokenized with a different method or using space delimiting
    # to denote reactants from reagents
    if ' ' in smiles:
        smiles = ''.join(smiles.split(' '))

    # sometimes a '>' character is used to denote reactants from reagents
    # this convention is not supported by RDKit, and must be converted to the traditional
    # '.' delimiter
    if '>' in smiles:
        smiles = smiles.replace('>', '.')

    smiles = canonicalize_smiles(smiles, remove_stereo=True)

    return smiles


def tokenize(smiles):
    # tokenizes SMILES string by character
    return ' '.join([i for i in smiles])



def pad_tokenize_smiles(tokenize_smiles):
    return tokenize_smiles


def train_and_evaluate(x_train, y_train, x_test, y_test, model_name, epochs=5):

    model = None
    save_model_name = "./models/"+model_name.__name__ + '.h5'
    save_model_img = "./logs/"+model_name.__name__+'.png'
    save_metrics_name = "./logs/"+model_name.__name__+'_metrics.csv'


    #### Training a Model
    print('\nTraining model {0} with {1:2d} training samples!'.format(model_name, x_train.shape[0]))

    if model_name == rf_train:
        model = model_name(x_train, y_train)
        model.fit(x_train,y_train)

    else:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto')
        mcp_save = ModelCheckpoint(save_model_name, save_best_only=True, monitor='val_loss', mode='auto')

        try:
            ### Load an existing model if exists
            print('Name of the model to load: ', save_model_name)
            with CustomObjectScope({'AttentionWithContext': AttentionWithContext}):
                model = load_model(save_model_name)
            print("Loading pretrained model :", model)
        except:
            print('No saved model found.')
            model = model_name(x_train, y_train)

        plot_model(model, to_file=save_model_img)
        history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2, verbose=2, callbacks=[earlyStopping, mcp_save])
        print(model.summary())

    #### Evaluate the model
    print('\nPrediction / evaluation of Model {}: '.format(model_name))
    y_pred = model.predict(x_test)
    print('Shape of y_pred', y_pred.shape)

    if model_name != rf_train:
        y_pred = np.argmax(y_pred, axis=1).reshape((y_pred.shape[0], 1))

    classes = sorted(list(set(y_test)))

    accuracy_per_class, precision_per_class, recall_per_class, f1score_per_class, mcc_score_per_class = \
        generate_model_report_per_class(y_test, y_pred, classes)

    # Print F1 score per class
    print('F1 Score per class')
    pprint.pprint(f1score_per_class)
    print('MCC Score per class')
    pprint.pprint(mcc_score_per_class)


    print("Average F1 score per class: ",  sum(f1score_per_class.values()) / len(f1score_per_class.values()))
    print("Average accuracy per class: ",  sum(accuracy_per_class.values()) / len(accuracy_per_class.values()))
    print("Average precision per class: ",  sum(precision_per_class.values()) / len(precision_per_class.values()))
    print("Average recall per class: ",  sum(recall_per_class.values()) / len(recall_per_class.values()))
    print("Average mcc score per class: ",  sum(mcc_score_per_class.values()) / len(mcc_score_per_class.values()))


    # Plot and save training & validation accuracy values
    metrics = pd.DataFrame()
    metrics['accuracy'] = history.history['accuracy']
    metrics['val_accuracy'] = history.history['val_accuracy']
    metrics['loss'] = history.history['loss']
    metrics['val_loss'] = history.history['val_loss']

    print(metrics)
    metrics.to_csv(save_metrics_name)

    '''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    '''

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
    mcc_score_per_class = {}
    for cls in classes:
        new_ytrue, new_ypred = convert_to_2_class(y_test, y_pred, cls)
        accuracy_per_class[cls] = accuracy_score(new_ytrue, new_ypred)
        precision_per_class[cls] = precision_score(new_ytrue, new_ypred)
        recall_per_class [cls] = recall_score(new_ytrue, new_ypred)
        f1score_per_class[cls] = f1_score(new_ytrue, new_ypred)
        mcc_score_per_class[cls] = matthews_corrcoef(new_ytrue, new_ypred)
    return accuracy_per_class, precision_per_class, recall_per_class, f1score_per_class, mcc_score_per_class
