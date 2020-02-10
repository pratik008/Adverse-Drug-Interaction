#!/bin/sh
python train.py -s Transfer_Learning -m rf_train -e 100 -t 0.25
python train.py -s Transfer_Learning -m mlp_train -e 50 -t 0.25
python train.py -s ECFP -m rf_train -e 100 -t 0.25
python train.py -s ECFP -m mlp_train -e 50 -t 0.25
python train.py -s SMILES -m cnn_1lstm_atten -e 50 -t 0.25
python train.py -s SMILES -m cnn_2lstm_atten -e 50 -t 0.25
python train.py -s SMILES -m cnn_3lstm_atten -e 50 -t 0.25

