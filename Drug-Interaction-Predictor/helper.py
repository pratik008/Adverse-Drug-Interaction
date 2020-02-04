from rdkit import Chem
import streamlit as st
import numpy as np
import pprint
from model import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from feature_generation import *


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

    #### Training a Model
    print('\nTraining model {0} with {1:2d} training samples!'.format(model_name, x_train.shape[0]))
    model = model_name(x_train, y_train)
    #print(model.summary())

    if model_name == rf_train:
        model.fit(x_train,y_train)

    elif model_name == mlp_train :
        model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, verbose=2)
        print(model.summary())

    else:
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, verbose=2)
        print(model.summary())

    #### Evaluate the model
    print('\nPrediction / evaluation of Model {}: '.format(model_name))
    y_pred = model.predict(x_test)
    print('Shape of y_pred', y_pred.shape)

    if model_name != rf_train:
        y_pred = np.argmax(y_pred, axis=1).reshape((y_pred.shape[0], 1))

    classes = sorted(list(set(y_test)))

    accuracy_per_class, precision_per_class, recall_per_class, f1score_per_class = \
        generate_model_report_per_class(y_test, y_pred, classes)

    # Print F1 score per class
    pprint.pprint(f1score_per_class)

    totalF1 = 0
    for item in f1score_per_class:
        totalF1 = totalF1 + f1score_per_class[item]

    print("Average F1 score per class: ", totalF1 / max(f1score_per_class))
    print("MCC Score: ", metrics.matthews_corrcoef(y_test, y_pred))


