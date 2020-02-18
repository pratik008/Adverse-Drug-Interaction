import argparse
from scripts.helper import *
from scripts.model import *


class Train(object):
    def __init__(self, train_style, train_model_name, test_size, epochs=10):
        self.train_style = train_style
        self.train_model_name = train_model_name
        self.test_size = test_size
        self.epochs = epochs
        self.train_model = None

        if self.train_style is None:
            self.train_style = 'ECFP'

        if self.train_model_name is None:
            self.train_model_name = 'mlp_train'

        model = Model()
        self.train_model = model.get_model_from_name(self.train_model_name)

        if self.train_model is None:
            raise ValueError('Model \'{}\' not found in Class Model. Please select appropriate model name.'.format(self.train_model_name))

        if self.test_size is None:
            self.test_size = 0.25
        else:
            self.test_size = float(test_size)

        if self.epochs is None:
            self.epochs = 10
        else:
            self.epochs = int(epochs)

    '''
    if train_style:
        if not(train_style == 'ECFP' or train_style == 'SMILES' or train_style == 'Transfer_Learning'):
            raise ValueError("Please enter a valid training style - choices are, 'SMILES' 'ECFP' 'Transfer_learning' ")

    '''


    def print_train_choise(self):
        print(self.train_style)
        print(self.train_model_name)
        print(self.test_size)
        print(self.epochs)
        print(self.train_model)



    def read_and_preprocess(self):
        print("training with style {} test size {} ".format(self.train_style, self.test_size))

        train_type = self.train_style  # Transfer_Learning, #ECFP, #SMILES

        start = timeit.default_timer()

        print('Reading drugs ...')
        # import XML Data - From link source

        drug_file_path = os.path.join('data','sample','drug_split.xml')
        #drug_file_path = os.path.join('../data','sample','full_database.xml')
        drug_list, smiles_dict = read_from_file(drug_file_path)
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
        print('Finished feature generation. Runtime : ', round((middle - start) / 60, 2), ' minutes')

        # rint = random.randint(1, 1000)
        test_size = self.test_size
        rint = 42
        x_train, x_test, y_train, y_test = train_test_split(X_label, \
                                                            y_label, test_size=test_size, random_state=rint, \
                                                            stratify=y_label)

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

        return x_train, x_test, y_train, y_test


    def train_and_evaluate(self,x_train, x_test, y_train, y_test):


        model = None
        save_model_name = os.path.join('models', self.train_style+'_'+self.train_model.__name__ + '.h5')
        save_model_img = os.path.join('logs', self.train_style+'_'+self.train_model.__name__+'.png')
        save_metrics_name = os.path.join('logs', self.train_style+'_'+self.train_model.__name__)


        #### Training a Model
        print('\nTraining model {0} with {1:2d} training samples!'.format(self.train_model, x_train.shape[0]))

        if self.train_model.__name__ == 'rf_train':
            model = self.train_model(x_train, y_train)
            model.fit(x_train,y_train)

        else:
            earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto')
            mcp_save = ModelCheckpoint(save_model_name, save_best_only=True, monitor='val_loss', mode='auto')

            try:
                ### Load an existing model if exists
                print('Name of the model to load: ', save_model_name)
                with CustomObjectScope({'AttentionWithContext': AttentionWithContext}):
                    model = load_model(save_model_name)
                print("Loading pretrained model :", model)
            except:
                print('No saved model found.')
                model = self.train_model(x_train, y_train)

            plot_model(model, to_file=save_model_img)
            history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=128, validation_split=0.2, verbose=2, callbacks=[earlyStopping, mcp_save])
            print(model.summary())

        #### Evaluate the model
        print('\nPrediction / evaluation of Model {}: '.format(self.train_model))
        y_pred = model.predict(x_test)
        print('Shape of y_pred', y_pred.shape)

        if self.train_model.__name__ != 'rf_train':
            y_pred = np.argmax(y_pred, axis=1).reshape((y_pred.shape[0], 1))

        classes = sorted(list(set(y_test)))

        accuracy_per_class, precision_per_class, recall_per_class, f1score_per_class, mcc_score_per_class = \
            generate_model_report_per_class(y_test, y_pred, classes)

        metrics = pd.DataFrame()

        # Print F1 score per class
        print('F1 Score per class')
        pprint.pprint(f1score_per_class)
        print('MCC Score per class')
        pprint.pprint(mcc_score_per_class)

        print("Average F1 score per class: ",  sum(f1score_per_class.values()) / len(f1score_per_class.values()))
        print("Average accuracy per class: ",  sum(accuracy_per_class.values()) / len(accuracy_per_class.values()))
        print("Average precision per class: ",  sum(precision_per_class.values()) / len(precision_per_class.values()))
        print("Average recall per class: ",  sum(recall_per_class.values()) / len(recall_per_class.values()))
        print("Average mcc score per class: ",  sum(mcc_score_per_class.values()) / len(mcc_score_per_class.values()))

        metrics['f1score_per_class'] = f1score_per_class.values()
        metrics['precision_per_class'] = precision_per_class.values()
        metrics['recall_per_class'] = recall_per_class.values()
        metrics['mcc_score_per_class'] = mcc_score_per_class.values()
        metrics['accuracy_per_class'] = accuracy_per_class.values()


        if self.train_model.__name__ != 'rf_train':
            loss_metrics = pd.DataFrame()

            # Plot and save training & validation accuracy values
            if tf.test.is_gpu_available():
                loss_metrics['acc'] = history.history['acc']
                loss_metrics['val_acc'] = history.history['val_acc']
                loss_metrics['loss'] = history.history['loss']
                loss_metrics['val_loss'] = history.history['val_loss']
            else:
                loss_metrics['acc'] = history.history['accuracy']
                loss_metrics['val_acc'] = history.history['val_accuracy']
                loss_metrics['loss'] = history.history['loss']
                loss_metrics['val_loss'] = history.history['val_loss']

            print(loss_metrics)
            loss_metrics.to_csv(save_metrics_name+'_loss_metrics.csv')

        print(metrics)
        metrics.to_csv(save_metrics_name+'_metrics.csv')


        '''
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        '''
        return save_model_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train drug interactions.')
    parser.add_argument('-s', '--train_style', help="Style of training - options are ECFP, SMILES, Transfer_Learning")
    parser.add_argument('-m', '--train_model', help="training model name")
    parser.add_argument('-t', '--test_size', help="% of test samples")
    parser.add_argument('-e', '--epochs', help="Number of epochs to train")

    args = parser.parse_args()
    print(args)


    #Create Train Object
    train_model = Train(args.train_style, args.train_model, args.test_size, args.epochs)
    train_model.print_train_choise()

    # Read Data and Preprocess
    x_train, x_test, y_train, y_test = train_model.read_and_preprocess()

    #Train Model and Evaluate
    saved_model = train_model.train_and_evaluate(x_train, x_test, y_train, y_test)
    print("Training Complete! Model saved at : ",saved_model )



