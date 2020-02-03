import rdkit as rd
from rdkit import Chem
from collections import Counter
from gensim.models import word2vec
from mol2vec_utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from smiles_transformer.pretrain_trfm import TrfmSeq2seq
from smiles_transformer.build_vocab import WordVocab
from smiles_transformer.utils import *


# File to generate numerical features from smiles data and replace
# interaction labels by numerical ones
#



pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4


def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm) > 218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109] + sm[-109:]
    vocab = WordVocab.load_vocab('../data/vocab.pkl')
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1] * len(ids)
    padding = [pad_index] * (seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a, b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)


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

    #word2vec_model = word2vec.Word2Vec.load(model_path)

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


def tokenize_smiles_and_interactions(relation_list,
     smiles_dict, label_map, token_length=1024):
    '''Generate numerical vectors (tokens) from smiles data and label interactions.

        The dictionary smiles_dict is used to find the SMILEs representations of
        drugs found in relation_list. subject and object SMILEs strings are concatenated,
        Keras tokenizer is used to convert the SMILEs string into tokens, and padding is
        used to standardize the length. The dictionary label_map is used to convert the interaction
        keywords in relation_list to numerical labels.


        Args :
            relation_list (list): List of Relation instances
            smiles_dict (dict): Dictionary mapping drug names to SMILES strings.
            label_map (dict): Dictionary mapping interaction keywords to
                numerical labels.
            token_length: Length of each token (Concatenated SMILEs string)

        Returns :
            X_label: tokenized, concatenated druga SMILEs with Drugb SMILEs. (X)
            y_label: tokenized label for the interactions. (y)
        '''

    X_concatenate_smile = []
    y_label = []

    for relation in relation_list:
        sub, obj, interaction = relation.get()
        sub_smiles, obj_smiles = smiles_dict[sub], smiles_dict[obj]
        interaction_label = label_map[interaction]
        X_concatenate_smile.append(sub_smiles[:int(token_length/2)]+obj_smiles[:int(token_length/2)])
        y_label.append(interaction_label)

    # ==============Tokenizer - Convert SMILEs Charcaters to index================
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tk.fit_on_texts(X_concatenate_smile)
    print(tk.index_word)
    smile_seq = tk.texts_to_sequences(X_concatenate_smile)
    i = 0
    for item in smile_seq:
        assert len(X_concatenate_smile[i]) == len(smile_seq[i]), "Length of Tokenized string should be the same as length of input string."
        i = i + 1

    X_label = pad_sequences(smile_seq, maxlen=token_length, padding='post')

    return X_label, y_label


def featurize_smiles_and_interactions(relation_list, smiles_feature_generator,
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



def smiles_transformer_tokenize(relation_list,
     smiles_dict, label_map):
    '''Generate numerical vectors (tokens) from smiles data and label interactions.

        The dictionary smiles_dict is used to find the SMILEs representations of
        drugs found in relation_list. subject and object SMILEs strings are concatenated,
        Keras tokenizer is used to convert the SMILEs string into tokens, and padding is
        used to standardize the length. The dictionary label_map is used to convert the interaction
        keywords in relation_list to numerical labels.


        Args :
            relation_list (list): List of Relation instances
            smiles_dict (dict): Dictionary mapping drug names to SMILES strings.
            label_map (dict): Dictionary mapping interaction keywords to
                numerical labels.
            token_length: Length of each token (Concatenated SMILEs string)

        Returns :
            X_label: tokenized, concatenated druga SMILEs with Drugb SMILEs. (X)
            y_label: tokenized label for the interactions. (y)
        '''

    vocab = WordVocab.load_vocab('../data/vocab.pkl')

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 3)
    if torch.cuda.is_available():
        trfm.load_state_dict(torch.load('../save/trfm.pkl'), strict=False)
    else:
        trfm.load_state_dict(torch.load('../save/trfm.pkl', map_location=torch.device('cpu')), strict=False)
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))


    smiles_list = []
    for item in smiles_dict:
        smiles_list.append(smiles_dict[item])

    trfm_dict = {}
    x_split = [split(sm) for sm in smiles_list]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    trfm_dict = {}

    i = 0
    for item in smiles_dict:
        trfm_dict[item] = X[i]
        i = i + 1


    X_concatenate_smile = []
    y_label = []

    for relation in relation_list:
        sub, obj, interaction = relation.get()
        sub_smiles, obj_smiles = trfm_dict[sub], trfm_dict[obj]
        interaction_label = label_map[interaction]
        concat_smiles = np.concatenate((sub_smiles, obj_smiles), axis=0)
        X_concatenate_smile.append(concat_smiles)
        y_label.append(interaction_label)

    return X_concatenate_smile, y_label



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
    
    #return np.array(smiles_feature_list), np.array(interaction_label_list),\
    #  np.array(drug_pair_list), filter_count
    return smiles_feature_list, interaction_label_list, drug_pair_list, filter_count


def filter_less_frequent_labels_v2(relation_list, cutoff_freq):
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
    
    label_freq = Counter()
    for relation in relation_list:
        label_freq[relation.normalized_relation] += 1
    
    filter_count = 0
    index = 0
    while index < len(relation_list):
        if label_freq[relation_list[index].normalized_relation] < cutoff_freq:
            del relation_list[index]
            filter_count += 1
        else:
            index += 1
    
    return relation_list, filter_count