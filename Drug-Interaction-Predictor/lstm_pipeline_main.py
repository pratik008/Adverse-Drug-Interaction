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

    # Tokenize smiles and interactions - create labels for training data
    X_label, y_label = tokenize_smiles_and_interactions(relation_list,smiles_dict,label_map)

    X_arr = np.array(X_label, dtype='float32')
    y_arr = np.array(y_label, dtype='float32')

    X_arr_small = X_arr[0:5000]
    y_arr_small = y_arr[0:5000]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

    # Stratified train test split with k folds
    for train_index, test_index in skf.split(X_arr_small, y_arr_small):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_arr_small[train_index], X_arr_small[test_index]
        y_train, y_test = y_arr_small[train_index], y_arr_small[test_index]
        model_rf = rf_train(X_train, y_train)
        accuracy, precision, recall, f1 = generate_model_report(model_rf, X_test, y_test)

    model = lstm_train(X_train,y_train)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


    print("Last Break Point")


if __name__ == '__main__':
    main()

