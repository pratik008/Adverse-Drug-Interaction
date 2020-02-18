from rdkit.Chem import AllChem
import numpy as np

# File contains helper function for generating mol2vec feature vectors.
#

def mol2alt_sentence(mol, radius):
    '''Same as mol2sentence() except it only returns the alternating sentence
    
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures 
    as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence 
    with identifiers from all radii combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each 
    radius is smaller
    
    Args :
        mol (rdkit.Chem.rdchem.Mol): Molecule to convert to sentence.
        radius (float): Fingerprint radius for ECFP.
    
    Returns:
    list : alternating sentence combined
    '''

    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def sentence2vec(sentence, model, unseen=None):
    '''Generate vectors for a sentence.
    
    Vector is simply a sum of vectors for individual words.
    
    Args :
        sentence (list): Alternating sentence obtained from mol2altsentence
        model (word2vec.Word2Vec): Gensim word2vec model
        unseen (str, Optional): Keyword for unseen words. If None, those words are skipped.
            https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary
            -for-text-analysis-using-neural-networks/163032#163032

    Returns:
        np.array : vector corresponding to input sentence.
    '''
    
    keys = set(model.wv.vocab.keys())

    if unseen:
        unseen_vec = model.wv.word_vec(unseen)

    if unseen:
        vec = sum([model.wv.word_vec(y) if y in set(sentence) & keys
                        else unseen_vec for y in sentence])
    else:
        vec = sum([model.wv.word_vec(y) for y in sentence 
                        if y in set(sentence) & keys])
    
    return np.array(vec)