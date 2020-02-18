from scripts.helper import *
import collections

if __name__ == '__main__':

    train_type = 'SMILES' #Transfer_Learning, #ECFP, #SMILES

    start = timeit.default_timer()

    print('Reading drugs ...')
    #import XML Data - From link source
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


    if train_type == 'ECFP':
        # Feature generarion - from SMILEs to ECFP
        X_label, y_label, drug_pair_list \
            = featurize_smiles_and_interactions(clean_relation_list, smiles_to_ECFP, smiles_dict, label_map)

    elif train_type == 'SMILES':
        # Tokenize smiles and interactions - create labels for training data
        X_label, y_label = tokenize_smiles_and_interactions(clean_relation_list, smiles_dict, label_map,
                                                            token_length=512)

    elif train_type == 'Transfer_Learning':
        # Transfer Features - from SMILEs_Transformer
        X_label, y_label = smiles_transformer_tokenize(clean_relation_list, smiles_dict, label_map)


    counter = collections.Counter(y_label)
    print(counter)
    y_label_dist = pd.DataFrame(counter.items())

    y_label_dist.to_csv("./logs/label_distribution.csv")


    middle = timeit.default_timer()
    print('Finished feature generation. Runtime : ', round((middle - start)/60, 2), ' minutes')

    #rint = random.randint(1, 1000)
    test_size = 0.25
    rint = 42
    x_train, x_test, y_train, y_test = train_test_split(X_label, \
        y_label, test_size = test_size, random_state = rint, \
            stratify = y_label)

    counter = collections.Counter(y_train)
    print(counter)
    y_train_dist = pd.DataFrame(counter.items())

    y_train_dist.to_csv("./logs/y_train_distribution.csv")


    traintest = timeit.default_timer()
    print('Finished train test split. Runtime : ', round((traintest - start) / 60, 2), ' minutes')

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print('Number of training classification labels : ', len(set(y_train)))
    print('Number of test classification labels : ', len(set(y_test)))
    print('Number of training samples : ', len(x_train))
    print('Number of test samples : ', len(x_test))

    if train_type == 'ECFP' or train_type == 'Transfer_Learning' or train_type == 'SMILES':
        # Use these for SMILES, Transfer_Learning and ECFP
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=rf_train, train_type=train_type)
        train_and_evaluate(x_train, y_train, x_test, y_test, model_name=mlp_train, train_type=train_type)

    # Use these for SMILES only
    if train_type == 'SMILES':
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=lstm_train, train_type=train_type)
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=lstm_2layer_train, train_type=train_type)
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=cnn_lstm_train, train_type=train_type)
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=model_cnn, train_type=train_type)
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=model_lstm_du, train_type=train_type)
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=model_lstm_atten, train_type=train_type)
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=cnn_2layer_lstm, train_type=train_type)
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=cnn_lstm_atten, train_type=train_type)
        #train_and_evaluate(x_train, y_train, x_test, y_test, model_name=lstm_2layer_du, train_type=train_type)
        train_and_evaluate(x_train, y_train, x_test, y_test, model_name=cnn_1lstm_atten, train_type=train_type)

    stop = timeit.default_timer()
    print('Total runtime: ', round((stop - start)/60, 2), ' minutes')

