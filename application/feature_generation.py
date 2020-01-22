import rdkit as rd
from rdkit import Chem
#from rdkit.Chem import AllChem
#import numpy as np
from collections import Counter
from gensim.models import word2vec
from mol2vec_utils import *


# File to generate numerical features from smiles data and replace
# interaction labels by numerical ones
#


def smiles_to_ECFP(smiles, model = None, fp_radius = 2):
    '''Convert a SMILES representation to ECFP representation.

    Args :
        smiles (str): SMILES representation.
        fp_radius (int): Radius for which the Morgan fingerprint is to be computed.
    
    Returns :
        fparr (numpy.ndarray): Morgan fingerprint in the form of a NumPy array.
            Returns None if smiles is None or not readable.
    '''
    
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius)
        else:
            return None
        fparr = np.zeros((1,))
        rd.DataStructs.ConvertToNumpyArray(fp, fparr)
    else:
        return None

    fparr = fparr.astype('B') # Convert to byte to save memory
    
    return fparr


def smiles_to_mol2vec_vector(smiles, model, fp_radius = 2, uncommon = None):
    '''Convert a SMILES string to a Mol2Vec vector
    

    '''

    if uncommon:
        try:
            model[uncommon]
        except KeyError:
            raise KeyError('Selected word for uncommon: %s not in vocabulary' % uncommon)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        sentence = mol2alt_sentence(mol, fp_radius)
        vec = sentence2vec(sentence, model, unseen = uncommon)
        if vec.shape == (300,):
            return vec
        else:
            return None
    else:
        return None


def featurize_smiles_and_interactions(relation_list, smiles_feature_generator,\
     smiles_dict, label_map):
    '''Generate numerical features from smiles data and label interactions.

    The dictionary smiles_dict is used to find the SMILES representations of 
    drugs found in relation_list. The function smiles_feature_generator 
    is then applied to these SMILES representations to generate features to
    train the model. The dictionary label_map is used to convert the interaction 
    keywords in relation_list to numerical labels.


    Args :
        relation_list (list): List of Relation instances
        smiles_feature_generator (function): Function that maps a SMILES string 
            to some kind of numerical feature.
        smiles_dict (dict): Dictionary mapping drug names to SMILES strings.
        label_map (dict): Dictionary mapping interaction keywords to 
            numerical labels.

    Returns :
        smiles_feature_list (list): List of features converted from SMILES strings.
        interaction_label_list (list): List of interaction labels that will be the 
            target for classification.
        drug_pair_list (list): List of pairs of drug names for later reference.
    '''

    feature_dict = {}
    smiles_feature_list = []
    interaction_label_list = []
    drug_pair_list = []
    if smiles_feature_generator == smiles_to_mol2vec_vector:
        model = word2vec.Word2Vec.load('model_300dim.pkl')
    else:
        model = None

    for relation in relation_list:
        sub, obj, interaction = relation.get()
        sub_smiles, obj_smiles = smiles_dict[sub], smiles_dict[obj]

        if sub_smiles not in feature_dict:
            feature_dict[sub_smiles] = smiles_feature_generator(sub_smiles, model = model)
        sub_feature = feature_dict[sub_smiles]

        if obj_smiles not in feature_dict:
            feature_dict[obj_smiles] = smiles_feature_generator(obj_smiles, model = model)
        obj_feature = feature_dict[obj_smiles]

        interaction_label = label_map[interaction]

        if sub_feature is not None and obj_feature is not None:
            smiles_feature_list.append(np.concatenate((sub_feature, obj_feature)))
            interaction_label_list.append(interaction_label)
            drug_pair_list.append((sub, obj))
        

    return smiles_feature_list, interaction_label_list, drug_pair_list


def filter_less_frequent_labels(smiles_feature_list, interaction_label_list,\
     drug_pair_list, cutoff_freq):
    '''Filters out labels that appear below a certain frequency.

    Args :
        smiles_feature_list (list): List of numerical features (obtained from SMILES strings).
        interaction_label_list (list): List of numerical labels that are the target for 
            classification.
        drug_pair_list (list): List of pairs of drug names.
        cutoff_freq (list): Only interactions labels that appear above this number are kept.

    Returns :
        smiles_feature_list (list): Filtered list of features.
        interaction_label_list (list): Filtered list of interaction labels.
        drug_pair_list (list): Filtered list of pairs of drug names.
    '''
    
    assert(len(smiles_feature_list) == len(interaction_label_list))
    assert(len(drug_pair_list) == len(interaction_label_list))
    
    label_freq = Counter()
    for label in interaction_label_list:
        label_freq[label] += 1
    
    filter_count = 0
    index = 0
    while index < len(smiles_feature_list):
        if label_freq[interaction_label_list[index]] < cutoff_freq:
            del smiles_feature_list[index]
            del interaction_label_list[index]
            del drug_pair_list[index]
            filter_count += 1
        else:
            index += 1
    
    return smiles_feature_list, interaction_label_list, drug_pair_list, filter_count