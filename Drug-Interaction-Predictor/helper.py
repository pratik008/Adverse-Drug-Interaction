from rdkit import Chem
import streamlit as st

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



