import argparse
from scripts.inference import *

from scripts.inference import predict_from_files

parser = argparse.ArgumentParser(description='Predict drug interactions.')
parser.add_argument('-c', '--candidates_file', help="Path to file with candidate SMILES strings\
     (txt file)")
parser.add_argument('-d', '--drugs_file', help="Path to file with drug SMILES strings \
    (txt file)")
parser.add_argument('-t', '--target_file', help="Path to target file to save interactions \
    (csv file)")
parser.add_argument('-m', '--model', help="Path to model to use for predictions \
    (csv file)", default=os.path.join('application', 'flaskapp', 'mlp_ECFP.h5'))
args = parser.parse_args()

candidates_file = args.candidates_file
drugs_file = args.drugs_file
target_file = args.target_file
model_file = args.model

if candidates_file is None or drugs_file is None or target_file is None:
    raise ValueError('Missing arguments')


if candidates_file[-4:] != '.txt':
    raise ValueError('Candidates file must be a txt file')

if drugs_file[-4:] != '.txt':
    raise ValueError('Drugs file must be a txt file')

if target_file[-4:] != '.csv':
    raise ValueError('Target file must be a csv file')

#model = tf.keras.models.load_model(args.model)

predict_from_files(candidates_file, drugs_file, target_file, model_file)