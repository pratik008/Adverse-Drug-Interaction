from read_data import *
from interaction_labelling import *
from feature_generation import *
from model import *
from helper import *
from sklearn.model_selection import StratifiedKFold
import timeit


def main():

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

    cleaning = timeit.default_timer()
    print('Finished data ingestion and cleaning. Runtime : ', round((cleaning - start)/60, 2), ' minutes')


    # Create the labels from the relation_list
    label_map, label_lookup = generate_labels(relation_list)

    # Tokenize smiles and interactions - create labels for training data
    X_label, y_label = tokenize_smiles_and_interactions(clean_relation_list,smiles_dict,label_map, token_length=512)


    middle = timeit.default_timer()
    print('Finished feature generation. Runtime : ', round((middle - start) / 60, 2), ' minutes')

    #rint = random.randint(1, 1000)
    test_size = 0.5
    rint = 42
    x_train, x_test, y_train, y_test = train_test_split(X_label, \
        y_label, test_size = test_size, random_state = rint, \
            stratify = y_label)

    print('Number of training classification labels : ', len(set(y_train)))
    print('Number of test classification labels : ', len(set(y_test)))
    print('Number of training samples : ', len(x_train))
    print('Number of test samples : ', len(x_test))


    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)


    x_train = x_train[0:1000]
    x_test = x_test[0:1000]
    y_train = y_train[0:1000]
    y_test = y_test[0:1000]


    print('Number of training classification labels : ', len(set(y_train)))
    print('Number of test classification labels : ', len(set(y_test)))
    print('Number of training samples : ', len(x_train))
    print('Number of test samples : ', len(x_test))


    print('\nTraining LSTM Model with tokenized SMILEs Strings ... ')
    #model = lstm_train(x_train,y_train)
    model = rf_train(x_train,x_test)

    print('\nPrediction / evaluation of mlp Model... ')
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1).reshape((y_pred.shape[0], 1))
    print('shape of y_pred is : ', y_pred.shape)

    len(y_test)


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


if __name__ == '__main__':
    main()

