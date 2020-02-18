from scripts.helper import *
import collections
import timeit
import streamlit as st
import pprint
from scripts.inference import *
from scripts.interaction_labelling import *
from scripts.feature_generation import *
from scripts.read_data import *
from sklearn.model_selection import train_test_split
from scripts.model import *
from keras.utils import CustomObjectScope, plot_model
from keras.models import load_model
from sklearn.metrics import *
import matplotlib.pyplot as plt


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

def plot_horizonatal_bar_chart(top_prob, top_labels):
    # set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    # set the style of the axes and the text color
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'
    plt.rcParams['text.color'] = '#333F4B'

    # create some fake data
    percentages = pd.Series(top_prob,
                            index=top_labels)
    df = pd.DataFrame({'percentage': percentages})
    df = df.sort_values(by='percentage')

    # we first need a numeric placeholder for the y axis
    my_range = list(range(1, len(df.index) + 1))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # create for each expense type an horizontal line that starts at x = 0 with the length
    # represented by the specific expense percentage value.
    plt.hlines(y=my_range, xmin=0, xmax=df['percentage'], color='#007ACC', alpha=0.2, linewidth=5)

    # create for each expense type a dot at the level of the expense percentage value
    plt.plot(df['percentage'], my_range, "o", markersize=5, color='#007ACC', alpha=0.6)

    # set labels
    ax.set_xlabel('Percentage', fontsize=15, fontweight='black', color='#333F4B')
    # ax.set_ylabel('Interaction Type')

    # set axis
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.yticks(my_range, df.index)

    # add an horizonal label for the y axis
    fig.text(-0.23, 0.96, 'Interaction Type', fontsize=15, fontweight='black', color='#333F4B')

    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    # set the spines position
    ax.spines['bottom'].set_position(('axes', -0.04))
    ax.spines['left'].set_position(('axes', 0.015))

    fig.set_size_inches(8, 4)

    # st.pyplot(plt.bar(top_labels, top_prob))
    st.pyplot(bbox_inches='tight')
