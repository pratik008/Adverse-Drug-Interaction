import numpy
from read_data import *
from inference import *
from interaction_labelling import *


def main():
    drug_list, smiles_dict = read_from_file('../data/sample/full_database.xml')
    print(drug_list)
    print(smiles_dict)

    interactions = generate_interactions(drug_list, smiles_dict)
    print(interactions)

    relation_list = generate_relations(interactions)
    print(relation_list)

    relation_list_new, filter_count = remove_duplicates(relation_list)
    relation_list_new2, filter_count2 = filter_unknowns(relation_list_new)

    print(filter_count)
    print(filter_count2)

    print(len(relation_list_new2))

    print("Last Break Point")
    #predict_from_files('../data/sample/candidates.txt'
    #                   ,'../data/sample/drugs.txt'
    #                   ,'../data/sample/output.csv'
    #                   ,'../application/flaskapp/mlp_ECFP.h5')


if __name__ == '__main__':
    main()

