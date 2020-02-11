#!/bin/sh
python train.py -s SMILES -m cnn_xxl_atten -e 50 -t 0.25
python train.py -s SMILES -m cnn_xl_atten -e 50 -t 0.25

