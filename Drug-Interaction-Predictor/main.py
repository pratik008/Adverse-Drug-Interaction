from read_data import *
from interaction_labelling import *
from feature_generation import *
from model import *
import timeit
from sklearn.metrics import *

# File to run everything and test functions

if __name__ == '__main__':
    
    start = timeit.default_timer()
    save = False
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
    
    
    label_map, label_lookup = generate_labels(clean_relation_list, save = save)

    cleaning = timeit.default_timer()
    print('Finished data ingestion and cleaning. Runtime : ', round((cleaning - start)/60, 2), ' minutes')

    '''
    smiles_feature_list, interaction_label_list, drug_pair_list = \
        featurize_smiles_and_interactions(clean_relation_list, \
            smiles_to_mol2vec_vector, smiles_dict, label_map)

    smiles_feature_list, interaction_label_list, drug_pair_list, filter_count = \
        filter_less_frequent_labels(smiles_feature_list, \
            interaction_label_list, drug_pair_list, 50)'''

    #print('Number of drug pairs retained : ', len(smiles_feature_list))
    #print('Pairs filtered : ', filter_count)

    middle = timeit.default_timer()
    print('Finished feature generation. Runtime : ', round((middle - start)/60, 2), ' minutes')

    # Feature generarion - from SMILEs to ECFP
    smiles_feature_list, interaction_label_list, drug_pair_list \
        = featurize_smiles_and_interactions(clean_relation_list, smiles_to_ECFP, smiles_dict, label_map)

    #rint = random.randint(1, 1000)
    rint = 42
    x_train, x_test, y_train, y_test = train_test_split(smiles_feature_list, \
        interaction_label_list, test_size = 0.2, random_state = rint, \
            stratify = interaction_label_list)

    z_train, z_test, y_train, y_test = train_test_split(drug_pair_list, \
        interaction_label_list, test_size = 0.2, random_state = rint, \
            stratify = interaction_label_list)

    print('Number of classification labels : ', len(set(y_train)))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)


    #x_train_small = x_train[0:5000]
    #y_train_small = y_train[0:5000]
    #x_test_small = x_test[0:5000]
    #y_test_small = y_test[0:5000]

    x_train_small = x_train
    y_train_small = y_train
    x_test_small = x_test
    y_test_small = y_test

    rf_model = rf_train(x_train_small, y_train_small)
    #mol2vec_model = mlp_mol2vec_train(x_train_small, y_train_small)
    mlp_model = mlp_train(x_train_small, y_train_small)

    #Choose the model for further processing
    model = mlp_model
    #model = mol2vec_model

    if save:
        model.save('mlp_ECFP.h5')
    
    #model.evaluate(x_test_small, y_test_small)
    y_pred_small = model.predict(x_test_small)
    #y_pred = preds

    if (model == mlp_model) or (model == mol2vec_model) :
        y_pred_small = np.argmax(y_pred_small, axis = 1).reshape((y_pred_small.shape[0],1))

    y_pred_small.shape

    accuracy = accuracy_score(y_test_small, y_pred_small)
    precision = precision_score(y_test_small, y_pred_small, average = 'weighted')
    recall = recall_score(y_test_small, y_pred_small, average = 'weighted')
    f1 = f1_score(y_test_small, y_pred_small, average = 'weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    
    classes = sorted(list(set(y_test_small)))

    accuracy_per_class, precision_per_class, recall_per_class, f1score_per_class = \
        generate_model_report_per_class(y_test_small, y_pred_small, classes)

    mcc_score = metrics.matthews_corrcoef(y_test_small,y_pred_small)

    totalF1 = 0
    for item in f1score_per_class:
        totalF1 = totalF1 + f1score_per_class[item]
        print("F1 score for class ",item," is : ", f1score_per_class[item])

    averageF1 = totalF1/max(f1score_per_class)
    print("Average F1 score per class: ",averageF1)
    print("MCC Score: ", mcc_score)


    #for cls in classes:
    #    print(cls, accuracy_per_class[cls])
    
    #for cls in classes:
    #    print(cls, y_test_small.count(cls))
    
    stop = timeit.default_timer()
    print('Total runtime: ', round((stop - start)/60, 2), ' minutes')
    
    #accuracy, precision, recall, f1 = generate_model_report(model, x_test, y_test)
    