#!/bin/sh
python -m scripts.train -s ECFP -m rf_train -e 10 -t 0.25
#python train.py -s SMILES -m lstm_train -e 10 -t 0.25
#python train.py -s SMILES -m rf_train -e 10 -t 0.25
#python train.py -s SMILES -m cnn_1lstm_atten -e 10 -t 0.25
#python train.py -s SMILES -m cnn_2lstm_atten -e 10 -t 0.25
