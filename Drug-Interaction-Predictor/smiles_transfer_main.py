from read_data import *
from interaction_labelling import *
from feature_generation import *
from model import *
import timeit
from sklearn.model_selection import train_test_split
from sklearn import metrics


# File to run everything and test functions
#vocab = WordVocab.load_vocab('../data/vocab.pkl')

# File to run everything and test functions

if __name__ == '__main__':

    start = timeit.default_timer()

    print('Reading drugs ...')
    # import XML Data - From link source
    drug_list, smiles_dict = read_from_file('../data/sample/full_database.xml')
    print('Drugs read : ', len(drug_list))

    print('Generating a list of interactions ...')
    interaction_list = generate_interactions(drug_list, smiles_dict)
    print('Interactions found : ', len(interaction_list))

    print('Generating relations ...')
    relation_list = generate_relations(interaction_list)
    clean_relation_list, filter_count = filter_unknowns(relation_list)
    print('Relations retained : ', len(clean_relation_list))
    print('Relations filtered : ', filter_count)

    clean_relation_list, filter_count = remove_duplicates(clean_relation_list)
    print('Relations left : ', len(clean_relation_list))
    print('Duplicates removed : ', filter_count)

    clean_relation_list, filter_count = filter_less_frequent_labels_v2(clean_relation_list, 50)
    print('Relations left : ', len(clean_relation_list))
    print('Pairs filtered : ', filter_count)

    # Create the labels from the relation_list
    label_map, label_lookup = generate_labels(clean_relation_list, save=False)

    cleaning = timeit.default_timer()
    print('Finished data ingestion and cleaning. Runtime : ', round((cleaning - start) / 60, 2), ' minutes')

    '''
    smiles_feature_list, interaction_label_list, drug_pair_list = \
        featurize_smiles_and_interactions(clean_relation_list, \
            smiles_to_mol2vec_vector, smiles_dict, label_map)

    smiles_feature_list, interaction_label_list, drug_pair_list, filter_count = \
        filter_less_frequent_labels(smiles_feature_list, \
            interaction_label_list, drug_pair_list, 50)'''

    # Transfer Features - from SMILEs_Transformer
    X_label, y_label = smiles_transformer_tokenize(clean_relation_list, smiles_dict, label_map)


    middle = timeit.default_timer()
    print('Finished feature generation. Runtime : ', round((middle - start) / 60, 2), ' minutes')

    # rint = random.randint(1, 1000)
    test_size = 0.25
    rint = 42
    x_train, x_test, y_train, y_test = train_test_split(X_label, \
                                                        y_label, test_size=test_size, random_state=rint, \
                                                        stratify=y_label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print('Number of training classification labels : ', len(set(y_train)))
    print('Number of test classification labels : ', len(set(y_test)))
    print('Number of training samples : ', len(x_train))
    print('Number of test samples : ', len(x_test))

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    print('\nTraining Random Forest with Transfer Learning ... ')
    model = rf_train(x_train, y_train)


    print('\nPrediction / Evaluation for Random Forest Model')
    y_pred = model.predict(x_test)
    print('shape of y_pred is : ', y_pred.shape)

    classes = sorted(list(set(y_test)))

    accuracy_per_class, precision_per_class, recall_per_class, f1score_per_class = \
        generate_model_report_per_class(y_test, y_pred, classes)

    mcc_score = metrics.matthews_corrcoef(y_test, y_pred)

    totalF1 = 0
    for item in f1score_per_class:
        totalF1 = totalF1 + f1score_per_class[item]
        print("F1 score for class ", item, " is : ", f1score_per_class[item])

    averageF1 = totalF1 / max(f1score_per_class)
    print("Average F1 score per class: ", averageF1)
    print("MCC Score: ", mcc_score)

    print('\nTraining mlp with Transfer Learning... ')
    model = mlp_train(x_train, y_train)

    print('\nPrediction / evaluation of mlp Model... ')
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1).reshape((y_pred.shape[0], 1))
    print('shape of y_pred is : ', y_pred.shape)

    classes = sorted(list(set(y_test)))

    accuracy_per_class, precision_per_class, recall_per_class, f1score_per_class = \
        generate_model_report_per_class(y_test, y_pred, classes)

    mcc_score = metrics.matthews_corrcoef(y_test, y_pred)

    totalF1 = 0
    for item in f1score_per_class:
        totalF1 = totalF1 + f1score_per_class[item]
        print("F1 score for class ", item, " is : ", f1score_per_class[item])

    averageF1 = totalF1 / max(f1score_per_class)
    print("Average F1 score per class: ", averageF1)
    print("MCC Score: ", mcc_score)

    stop = timeit.default_timer()
    print('Total runtime: ', round((stop - start) / 60, 2), ' minutes')

    # accuracy, precision, recall, f1 = generate_model_report(model, x_test, y_test)
