from read_data import *
from interaction_labelling import *
from feature_generation import *
from model import *
import timeit
from sklearn.model_selection import train_test_split
from sklearn import metrics
from helper import *

def main():

    start = timeit.default_timer()

    print('Reading drugs ...')
    # import XML Data - From link source
    drug_list, smiles_dict = read_from_file('../data/sample/drug_split11.xml')
    #drug_list, smiles_dict = read_from_file('../data/sample/full_database.xml')
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

    print('print label mappings : ', label_map)

    # Tokenize smiles and interactions - create labels for training data
    X_label, y_label = tokenize_smiles_and_interactions(clean_relation_list,smiles_dict, label_map, token_length=512)

    middle = timeit.default_timer()
    print('Finished feature generation. Runtime : ', round((middle - start) / 60, 2), ' minutes')

    # rint = random.randint(1, 1000)
    test_size = 0.60
    print('inside train and evaluate - calling train test split')
    rint = 42
    x_train, x_test, y_train, y_test = train_test_split(X_label,
                                                        y_label, test_size=test_size, random_state=rint,
                                                        stratify=y_label)

    print('completed train test split')
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print('Number of training classification labels : ', len(set(y_train)))
    print('Number of test classification labels : ', len(set(y_test)))
    print('Number of training samples : ', len(x_train))
    print('Number of test samples : ', len(x_test))

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    traintest = timeit.default_timer()
    print('Finished train test split. Runtime : ', round((traintest - start) / 60, 2), ' minutes')

    #train_and_evaluate(x_train, y_train, x_test, y_test, rf_train)
    #train_and_evaluate(x_train, y_train, x_test, y_test, mlp_train)
    #train_and_evaluate(x_train, y_train, x_test, y_test, lstm_train)
    #train_and_evaluate(x_train, y_train, x_test, y_test, lstm_2layer_train)
    #train_and_evaluate(x_train, y_train, x_test, y_test, cnn_lstm_train)
    #train_and_evaluate(x_train, y_train, x_test, y_test, model_cnn)
    #train_and_evaluate(x_train, y_train, x_test, y_test, model_lstm_du)
    #train_and_evaluate(x_train, y_train, x_test, y_test, model_lstm_atten)
    #train_and_evaluate(x_train, y_train, x_test, y_test, cnn_2layer_lstm)
    train_and_evaluate(x_train, y_train, x_test, y_test, cnn_lstm_atten)
    #train_and_evaluate(x_train, y_train, x_test, y_test, lstm_2layer_du)

    stop = timeit.default_timer()
    print('Total runtime: ', round((stop - start) / 60, 2), ' minutes')


if __name__ == '__main__':
    main()

