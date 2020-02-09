from inference import *

if __name__ == '__main__':
    print("inside Main of Test FIle")
    predict_from_files('../data/sample/candidates.txt', '../data/sample/drugs.txt', 'output.csv', 'models/SMILES_lstm_2layer_train.h5')
