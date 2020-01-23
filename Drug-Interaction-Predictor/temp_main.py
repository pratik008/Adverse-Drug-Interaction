from read_data import *
from interaction_labelling import *
from feature_generation import *
from model import *
from helper import *
from sklearn.model_selection import StratifiedKFold


def main():
    # import XML Data - From link source
    drug_list, smiles_dict = read_from_file('../data/sample/full_database.xml')

    # preprocessing
    interactions = generate_interactions(drug_list, smiles_dict)
    relation_list = generate_relations(interactions)
    print("length of relation_list before preprocessing", len(relation_list))

    # More preprocessing
    relation_list, filter_count = remove_duplicates(relation_list)
    relation_list, filter_count2 = filter_unknowns(relation_list)
    relation_list, filter_count3 = filter_less_frequent_labels_v2(relation_list, 300)

    print("filtered duplicates: ", filter_count)
    print("filtered unknowns", filter_count2)
    print("filtered labels with less than x count", filter_count3)

    print("length of relation_list after removing duplicates", len(relation_list))

    # Create the labels from the relation_list
    label_map, label_lookup = generate_labels(relation_list)

    print(label_map)
    print(label_lookup)



    item = 'temp'
    smiles_tokenzied_dic = {}
    for item in smiles_dict:
        smiles, smiles_tokenized = process_and_tokenize(smiles_dict[item])
        smiles_tokenzied_dic.


    a, b = process_and_tokenize('CC(=O)O')
    print(a,b)

    print(len(b))


    print("Last Break Point")


if __name__ == '__main__':
    main()

