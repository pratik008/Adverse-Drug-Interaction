import tensorflow as tf
from feature_generation import smiles_to_ECFP
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import os

# File that contains inference functions
#

def read_dict_from_csv(csv_file):
    '''Read  dictionary from a csv file

    Args :
        csv_file (str): Name and location of csv file to be read from
    
    Returns :
        d (dict): Dictionary read from file.
    '''
    
    if csv_file[-3:] == 'csv':
        d = {}
        try:
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                d = {row[0]:row[1] for row in reader}
        except IOError:
            print('I/O Error')
    else:
        print('Not a csv file.')

    return d


def predict_interaction(smiles, smiles_b, model = 'mlp', feature = 'ECFP', directory = ''):
    '''Use model to predict interactions

    Args :
    smiles (str): First SMILES string
    smiles_b (str): Second SMILES string
    model (str): Name of model used to train
    feature (str): Name of feature embedding used
    directory (str): Path to directory containing model

    Returns :
    prediction (numpy.ndarray): Array containing prediction from model
    '''

    model_path = os.path.join(directory, model + '_' + feature + '.h5')
    model = tf.keras.models.load_model(model_path)

    vec_a = smiles_to_ECFP(smiles)
    vec_b = smiles_to_ECFP(smiles_b)
    test = np.concatenate((vec_a, vec_b)).reshape((1, -1))
    prediction = model.predict(test)

    return prediction


def predict_from_files(candidates_file, drugs_file, target_file, model):
    '''Use model to predict interaction between candidates and drugs

    Args :
    candidates_file (str): Path to txt file with candidate SMILES strings
    drugs_file (str): Path to txt file with drug SMILES strings
    target_file (str): Path to csv file to write results to
    model (object): Pre-trained model to use to predict interactions from

    Returns :
    None
    '''

    candidates_list = []
    with open(candidates_file) as file:
        for line in file:
            candidates_list.append(line)
    print('Loaded drug candidates.')

    drugs_list = []
    with open(drugs_file) as file:
        for line in file:
            drugs_list.append(line)
    print('Loaded existing drugs')

    label_lookup = read_dict_from_csv(os.path.join('application', 'flaskapp', 'label_lookup.csv'))

    interactions_df = pd.DataFrame(columns = ['Candidate SMILES', 'Drug SMILES', \
        'Interaction 1', 'Probability 1', 'Interaction 2', 'Probability 2', \
            'Interaction 3', 'Probability 3'])

    print('Predicting interactions ...')
    for candidate in tqdm(candidates_list, desc = 'Candidates : '):
        vec_a = smiles_to_ECFP(candidate)
        if vec_a is not None:
            for drug in tqdm(drugs_list, desc = 'Drugs : '):
                vec_b = smiles_to_ECFP(drug)
                if vec_b is not None:    
                    test = np.concatenate((vec_a, vec_b)).reshape((1, -1))
                    prediction = model.predict(test)
                    top_labels, top_probs = get_top_n(prediction, 3)
                    top_labels = list(map(lambda x : label_lookup[str(x)], top_labels))
                    interactions_df = interactions_df.append({
                        'Candidate SMILES':candidate, 'Drug SMILES':drug, \
                            'Interaction 1':top_labels[0], 'Probability 1':top_probs[0], \
                                'Interaction 2':top_labels[1], 'Probability 2':top_probs[1], \
                                    'Interaction 3':top_labels[2], 'Probability 3':top_probs[2]}, \
                                        ignore_index = True)
        
    interactions_df.to_csv(target_file, index=False)


def get_top_n(arr, n):
    '''Return the top n elements and indices of a numpy array

    Args :
    arr (numpy.ndarray): Array that contains labels and corresponding probablilites
    n (int): Number of top values to return

    Returns :
    top_labels (list): List of numerical labels that have the top probabilities
    top_probs (list): Descending list of probabilities
    '''

    assert(type(n) == int and n > 0)
    arr_df = pd.DataFrame(data = arr[0], columns = ['Probabilities'])
    arr_df.sort_values('Probabilities', ascending = False, inplace = True)
    top_labels = list(arr_df[:n].index)
    top_probs = list(arr_df[:n]['Probabilities'])
    return top_labels, top_probs