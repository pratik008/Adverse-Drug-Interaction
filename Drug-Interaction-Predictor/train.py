import argparse
from helper import *

parser = argparse.ArgumentParser(description='Train drug interactions.')
parser.add_argument('-s', '--train_style', help="Style of training - options are ECFP, SMILES, Transfer_Learning")
parser.add_argument('-m', '--train_model', help="training model name")
parser.add_argument('-t', '--test_size', help="% of test samples")
parser.add_argument('-e', '--epochs', help="Number of epochs to train")

args = parser.parse_args()

train_style = args.train_style
train_model = args.train_model
test_size = float(args.test_size)
epochs = int(args.epochs)

if train_style is None:
    raise ValueError('Missing arguments')

if train_style:
    if not(train_style == 'ECFP' or train_style == 'SMILES' or train_style == 'Transfer_Learning'):
        raise ValueError("Please enter a valid training style - choices are, 'SMILES' 'ECFP' 'Transfer_learning' ")

if train_model == 'mlp_train':
    train_model = mlp_train
elif train_model == 'lstm_train':
    train_model = lstm_train
elif train_model == 'rf_train':
    train_model = rf_train
elif train_model == 'cnn_1lstm_atten':
    train_model = cnn_1lstm_atten
elif train_model == 'cnn_2lstm_atten':
    train_model = cnn_2lstm_atten
elif train_model == 'cnn_3lstm_atten':
    train_model = cnn_3lstm_atten
elif train_model == 'cnn_xl_atten':
    train_model = cnn_xl_atten
elif train_model == 'cnn_xxl_atten':
    train_model = cnn_xxl_atten

if test_size is None:
    test_size = 0.25

if epochs is None:
    epochs = 10

#Read Data and Preprocess
x_train, x_test, y_train, y_test = read_and_preprocess(train_style, test_size)

#Train Data and Evaluate
train_and_evaluate(x_train, x_test, y_train, y_test, model_name=train_model, train_type=train_style, epochs=epochs)

#Load model and infer



