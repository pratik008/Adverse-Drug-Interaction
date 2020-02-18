import tensorflow as tf
from scripts.feature_generation import smiles_to_ECFP, tokenize_SMILES
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import os
from keras.utils import CustomObjectScope
from keras.models import load_model
from scripts.model import AttentionWithContext
import argparse


# File that contains inference functions
#


class Inference:
    def __init__(self, druga_smiles, drugb_smiles, inference_model, inference_type):
        self.druga_smiles = druga_smiles
        self.drugb_smiles = drugb_smiles
        self.inference_model = inference_model
        self.inference_type = inference_type

        if self.druga_smiles is None:
            self.druga_smiles = 'CCC@HC@HC(=O)N1CCC[C@H]1C(=O)NC@@HC(=O)NC@@HC(=O)NC@@HC(=O)NC@@HC(O)=O'

        if self.drugb_smiles is None:
            self.drugb_smiles = 'CC(C)CC@HC(=O)NC@@HC(=O)N1CCC[C@H]1C(=O)NC@HC(N)=O'

        if self.inference_model is None:
            self.inference_model = 'mlp_train'

        if self.inference_type is None:
            self.inference_type = 'SMILES'


    def read_dict_from_csv(self,csv_file):
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
            except Exception as e:
                print('I/O Error', e)
        else:
            print('Not a csv file.')

        return d


    def predict_interaction(self):
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

        model_path = os.path.join('models', self.inference_type + '_' + self.inference_model + '.h5')
        try:
            ### Load an existing model if exists
            print('Name of the model to load: ', model_path)
            with CustomObjectScope({'AttentionWithContext': AttentionWithContext}):
                print(tf.device)
                with tf.device('/cpu:0'):
                    #new_model = load_model('test_model.h5')
                    model = load_model(model_path)
            print("Loading pretrained model :", model)
        except Exception as e:
            print(e)
            print('No saved model found.')

        label_lookup_path = os.path.join('helper_files', 'label_lookup.csv')
        label_lookup = self.read_dict_from_csv(label_lookup_path)

        if self.inference_type == 'ECFP':
            vec_a = smiles_to_ECFP(self.druga_smiles)
            vec_b = smiles_to_ECFP(self.drugb_smiles)
        elif self.inference_type == 'SMILES':
            vec_a = tokenize_SMILES(self.druga_smiles)
            vec_b = tokenize_SMILES(self.drugb_smiles)

        test = np.concatenate((vec_a, vec_b)).reshape((1, -1))

        test = pad_sequences(test, maxlen=512, padding='post')
        prediction = model.predict(test)
        top_labels, top_probs = self.get_top_n(prediction, 4)
        top_labels = list(map(lambda x : label_lookup[str(x)], top_labels))
        return top_labels, top_probs



    def predict_from_files(self,candidates_file: object, drugs_file: object, target_file: object) -> object:
        '''Use model to predict interaction between candidates and drugs

        Args :
            candidates_file (str): Path to txt file with candidate SMILES strings
            drugs_file (str): Path to txt file with drug SMILES strings
            target_file (str): Path to csv file to write results to
            model (object): Pre-trained model to use to predict interactions from

        Returns :
            None
        '''
        candidates_file_path = os.path.join('data', 'sample', candidates_file)
        drugs_file_path = os.path.join('data', 'sample', drugs_file)

        candidates_list = []
        with open(candidates_file_path) as file:
            for line in file:
                candidates_list.append(line)
        print('Loaded drug candidates.')

        drugs_list = []
        with open(drugs_file_path) as file:
            for line in file:
                drugs_list.append(line)
        print('Loaded existing drugs')

        print('********************')
        print(os.path)
        label_lookup_path = os.path.join('helper_files', 'label_lookup.csv')
        label_lookup = self.read_dict_from_csv(label_lookup_path)

        interactions_df = pd.DataFrame(columns = ['Candidate SMILES', 'Drug SMILES', \
            'Interaction 1', 'Probability 1', 'Interaction 2', 'Probability 2', \
                'Interaction 3', 'Probability 3'])


        model_path = os.path.join('models', self.inference_type + '_' + self.inference_model + '.h5')
        try:
            ### Load an existing model if exists
            print('Name of the model to load: ', model_path)
            with CustomObjectScope({'AttentionWithContext': AttentionWithContext}):
                print(tf.device)
                with tf.device('/cpu:0'):
                    # new_model = load_model('test_model.h5')
                    model = load_model(model_path)
            print("Loading pretrained model :", model)
        except Exception as e:
            print(e)
            print('No saved model found.')
            raise


        #model = tf.keras.models.load_model(model_file)

        print('Predicting interactions ...')
        for candidate in tqdm(candidates_list, desc = 'Candidates : '):
            vec_a = smiles_to_ECFP(candidate)
            if vec_a is not None:
                for drug in tqdm(drugs_list, desc = 'Drugs : '):
                    vec_b = smiles_to_ECFP(drug)
                    if vec_b is not None:
                        test = np.concatenate((vec_a, vec_b)).reshape((1, -1))
                        prediction = model.predict(test)
                        top_labels, top_probs = self.get_top_n(prediction, 3)
                        top_labels = list(map(lambda x : label_lookup[str(x)], top_labels))
                        interactions_df = interactions_df.append({
                            'Candidate SMILES':candidate, 'Drug SMILES':drug, \
                                'Interaction 1':top_labels[0], 'Probability 1':top_probs[0], \
                                    'Interaction 2':top_labels[1], 'Probability 2':top_probs[1], \
                                        'Interaction 3':top_labels[2], 'Probability 3':top_probs[2]}, \
                                            ignore_index = True)

        target_file_path = os.path.join('logs', target_file)
        interactions_df.to_csv(target_file_path, index=False)


    def get_top_n(self,arr: object, n: int) -> object:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for drug interactions.')
    parser.add_argument('-a', '--druga_smiles', help="SMILES string for Drug A")
    parser.add_argument('-b', '--drugb_smiles', help="SMILES string for Drug B")
    parser.add_argument('-m', '--inference_model', help="Inference Model Name")
    parser.add_argument('-t', '--inference_type', help="Inference Type")

    args = parser.parse_args()
    print(args)

    inference = Inference(args.druga_smiles, args.drugb_smiles, args.inference_model, args.inference_type)
    top_labels, top_prob = inference.predict_interaction()

    print(top_labels, top_prob)

    print("Test Predict from fIles")
    #infer_model = Inference()
    inference_from_file = Inference(druga_smiles=None, drugb_smiles=None, inference_model='mlp_train', inference_type='ECFP')
    inference_from_file.predict_from_files('candidates.txt', 'drugs.txt', 'inference_output.csv')

