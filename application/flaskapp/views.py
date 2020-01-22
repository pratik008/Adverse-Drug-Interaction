from flask import render_template
from flask import request
from flaskapp import app
import csv
from inference import *
from feature_generation import smiles_to_ECFP

# File that contains all the python processing and the views logic
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


SMILES_DICT = read_dict_from_csv('flaskapp/smiles_dictionary.csv')


@app.route('/', methods = ['GET'])
@app.route('/home')
def home():
    global SMILES_DICT

    return render_template('home.html', drug_list = SMILES_DICT.keys())


@app.route('/predict')
def predict():
    global SMILES_DICT
    smiles = request.args.get('SMILES').strip()
    drug_name = request.args.get('drug_name').strip()

    if smiles == '':
        return render_template('error.html', error = 'no_smiles')
    
    if drug_name == '':
        return render_template('error.html', error = 'no_drug')
    
    fp = smiles_to_ECFP(smiles)
    if fp is None:
        return render_template('error.html', error = 'invalid_smiles')
    
    if drug_name not in SMILES_DICT:
        return render_template('error.html', error = 'drug_not_found')


    prediction = predict_interaction(smiles, SMILES_DICT[drug_name], directory = 'flaskapp')

    top_labels, top_probs = get_top_n(prediction, 5)

    label_lookup = read_dict_from_csv('flaskapp/label_lookup.csv')

    top_labels = list(map(lambda x : label_lookup[str(x)], top_labels))

    return render_template('result.html', top_labels = top_labels, top_probs = top_probs, \
        smiles = smiles, drug_name = drug_name)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/help')
def help_page():
    return render_template('help.html')


@app.route('/credits')
def credits_page():
    return render_template('credits.html')


@app.route('/feedback')
def feedback_page():
    return render_template('feedback.html')