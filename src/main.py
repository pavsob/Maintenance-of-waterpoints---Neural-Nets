import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import sys
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

######****** Functions *******#####
### Train function ###
def train(clf, X_train, y_train):

    # Fitting into the pipeline which performs all the transformations and finally the model fitting
    clf.fit(X_train, y_train)

    # Getting the predicted label
    y_train_hat = clf.predict(X_train)

    # Training accuracy score
    print("Accuracy on training set: " + str(accuracy_score(y_train, y_train_hat)))

    # Classification report
    target_names = ['functional needs repair', 'others']
    print(classification_report(y_train, y_train_hat, target_names=target_names))

    return clf

### Test function ###
def test(clf_loaded, X_test, y_test, Label, task, imbalance):
    if imbalance == 'smote':
        try:
            with open(task + "/NN-preprocessor.pkl","rb") as f:
                preprocessor_loaded = pickle.load(f)
        except FileNotFoundError:
            print("Train the model first")
            sys.exit()
        # Preprocess first
        X_test = preprocessor_loaded.transform(X_test)
        # calculate the predictions
        y_test_hat = clf_loaded.predict(X_test)
        print('washere')
    else:
        y_test_hat = clf_loaded.predict(X_test)

    # print accuracy
    print("Accuracy on test set: " + str(accuracy_score(y_test, y_test_hat)))

    # Confussion matrix
    print("Confussion matrix: ")
    print(str(confusion_matrix(y_test, y_test_hat)))

    # Precission-Recall curve
    plot_precision_recall_curve(clf_loaded, X_test, y_test)
    plt.show()

    # Classification report
    target_names = ['functional needs repair', 'others']
    print(classification_report(y_test, y_test_hat, target_names=target_names))

    # Saving predictions into txt file
    y_predicted_labels = Label.inverse_transform(y_test_hat)
    df_test_predicted = pd.DataFrame(y_predicted_labels, columns = ['Status_group'])
    df_test_predicted.to_csv(task + '/' + task +'_test_predicted_labels.txt', sep='\t')

### Predict function ###
def predict(X_test_nolabels, clf_loaded, id_column, imbalance):
    if imbalance == 'smote':
        try:
            with open(task + "/NN-preprocessor.pkl","rb") as f:
                preprocessor_loaded = pickle.load(f)
        except FileNotFoundError:
            print("Train the model first")
            sys.exit()
        # Preprocess first
        X_test_nolabels = preprocessor_loaded.transform(X_test_nolabels)
        # calculate the predictions
        y_test_nolabels_hat = clf_loaded.predict(X_test_nolabels)
    else:
        # calculate the predictions
        y_test_nolabels_hat = clf_loaded.predict(X_test_nolabels)

    # Pritnting IDs that needs repair
    # Functional needs repait is encoded under 0
    print("ID of pumps needing repair")
    index = 0
    for i in y_test_nolabels_hat:
        if i == 0:
            print(id_column[index])
        index+=1

### Duplicates ###
def duplicates(df):
    count=0
    for i in df.duplicated():
        if i==True:
            count+=1
    print('number of duplicates: ' + str(count))
    df.drop_duplicates(inplace = True)
    return df

### Cleaning function ###
def cleanData(df_test):
    # Dropping unnecessary columns
    df_test.drop(columns = ['id','num_private','recorded_by', 'date_recorded', 'scheme_name', 'wpt_name', 'extraction_type_group',
         'extraction_type_class', 'payment_type', 'quality_group', 'quantity_group', 'waterpoint_type_group',
         'source_type', 'source_class', 'district_code', 'region_code', 'management_group', 'scheme_management',
         'subvillage', 'construction_year', 'funder'], inplace=True)
    # Creating NaN values
    df_test.replace('unknown', np.nan, inplace = True)
    return df_test

### Under-sampling ###
# Manually created function that deletes surplus
def underSample(df):
    # minoriy index and length
    length_minority_class = len(df[df['status_group'] == 'functional needs repair'])
    index_minority = df[df['status_group'] == 'functional needs repair'].index
    # majority random choice delete to have the same length as minority class
    index_majority_class = df[df['status_group'] == 'others'].index
    index_random_maj = np.random.choice(index_majority_class, length_minority_class, replace=False)
    # Get undersampling indexes
    under_sample_index = np.concatenate([index_random_maj, index_minority])
    # Creating new dataframe with the indexes
    under_sample = df.loc[under_sample_index]
    return under_sample

######****** TASKS *******######

#### Task 1 function ####
def task1(act):
    Label = preprocessing.LabelEncoder()
    ##### Train part #####
    if act == 'train':
        # Loading the data train set
        df = pd.read_csv('task1/task1_train.csv')
        # Dropping ID, dropping num_private
        df.drop(['id','num_private'], inplace = True, axis=1)
        # Splitting input and output data
        y_train = Label.fit_transform(df['status_group'])
        X_train = df.drop(['status_group'], axis=1)
        # Setting up pipelines of transformers
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        # Putting it together in ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, make_column_selector(dtype_include=['float64', 'int64']))])
        # Model settings
        model = MLPClassifier(hidden_layer_sizes=[11,12], batch_size=100, activation='tanh', solver='sgd', learning_rate_init=0.001, early_stopping=True, n_iter_no_change=10, random_state=0)
        # Setting up the final pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf = train(clf, X_train, y_train)
        # Save model
        with open("task1/NN-model-task1.pkl","wb") as f:
            pickle.dump(clf,f)

    ##### Test part ####
    elif act == 'test':
        # Load saved model
        try:
            with open("task1/NN-model-task1.pkl","rb") as f:
                clf_loaded = pickle.load(f)
        except FileNotFoundError:
            print("Train the model first")
            sys.exit()
        # load a test set
        df_test = pd.read_csv("task1/task1_test.csv")
        y_test = Label.fit_transform(df_test['status_group'])
        X_test = df_test.drop(['id','num_private','status_group'], axis=1)
        # tets function
        test(clf_loaded, X_test, y_test, Label,task,'')

    ##### Predict part #####
    elif act == 'predict':
        # Load saved model
        try:
            with open("task1/NN-model-task1.pkl","rb") as f:
                clf_loaded = pickle.load(f)
        except FileNotFoundError:
            print("Train the model first")
            sys.exit()
        # load a test set with no labels
        df_test_nolabels = pd.read_csv("task1/task1_test_nolabels.csv")
        id_column = df_test_nolabels['id']
        X_test_nolabels = df_test_nolabels.drop(['id','num_private'], axis=1)
        predict(X_test_nolabels, clf_loaded, id_column, '')

    else:
        print("Invalid action argument. Choose action: train, test or predict")

#### Task 2-3 function ####
def task234(task, act, imbalance):

    # Loading data and cleaning the data + dealing with imbalanced data-oversamplin and undersampling options, SMOTE option in train part
    if act == 'predict':
        df = pd.read_csv(task + "/" + task + "_test_nolabels.csv")
        id_column = df['id']
        df = cleanData(df)
        X = df
    else:
        df = pd.read_csv(task + "/" + task + "_" + act + ".csv")
        df = cleanData(df)
        if act == 'train':
            df = duplicates(df)
        Label = preprocessing.LabelEncoder()
        ## Under-sampling using custom function
        if imbalance == 'under' and act == 'train':
            df = underSample(df)
        y = Label.fit_transform(df['status_group'])
        X = df.drop(['status_group'], axis=1)
        ## Over-sampling using imblearn
        if imbalance == 'over' and act == 'train':
            ros = RandomOverSampler(random_state = 0)
            X, y = ros.fit_resample(X, y)

    ##### Train part #####
    if act == 'train':
        # Setting up pipelines of transformers - imputes, scale and encode data
        # There are no missing numeric values but in case different data set has, it fills NaNs with mean values  
        num_transformer = Pipeline(steps=[('imputernum', SimpleImputer(missing_values=np.nan, strategy='mean')),
                                          ('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('imputercat', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

        # Putting it together in ColumnTransformer
        preprocessor = ColumnTransformer(
        transformers=[('cat', cat_transformer, make_column_selector(dtype_include=['object','bool'])),
                      ('num', num_transformer, make_column_selector(dtype_include=['float64', 'int64']))])
        ## Model settings
        model = MLPClassifier(hidden_layer_sizes=[10,10], batch_size=70, activation='relu', solver='sgd', learning_rate_init=0.001,
                              early_stopping=True, n_iter_no_change=50, random_state=0)
        
        ### Over-sampling - SMOTE
        if imbalance == 'smote':
            print('Shape before SMOTE')
            print(X.shape)
            print(y.shape)
            # Using SMOTE from imblearn
            smote = SMOTE(random_state = 0)
            preprocessor.fit(X)
            X = preprocessor.transform(X)
            X, y = smote.fit_resample(X,y)
            print('Shape after SMOTE')
            print(X.shape)
            print(y.shape)
            # We already preprocessed data; therefore, clf is only the model - then it is necessary to alter test and predict part to preprocess data separately when performing SMOTE
            clf = Pipeline(steps=[('classifier', model)])
            with open(task + "/NN-preprocessor.pkl","wb") as f:
                pickle.dump(preprocessor,f)
        else:
            # Setting up the final pipeline
            clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        clf = train(clf, X, y)
        # Save model
        with open(task + "/NN-model.pkl","wb") as f:
            pickle.dump(clf,f)

    ##### Test part ####
    elif act == 'test':
        # Load saved model
        try:
            with open(task + "/NN-model.pkl","rb") as f:
                clf_loaded = pickle.load(f)
        except FileNotFoundError:
            print("Train the model first")
            sys.exit()
        # tets function
        test(clf_loaded, X, y, Label, task, imbalance)


    ##### Predict part #####
    elif act == 'predict':
        # Load saved model
        try:
            with open(task + "/NN-model.pkl","rb") as f:
                clf_loaded = pickle.load(f)
        except FileNotFoundError:
            print("Train the model first")
            sys.exit()
        # load a test set with no labels
        predict(X, clf_loaded, id_column, imbalance)

    else:
        print("Invalid action argument. Choose action: train, test or predict")


##########******** Main Program ********##########
task = str(sys.argv[1])
act = str(sys.argv[2])

if task == 'task1':
    task1(act)
elif task == 'task2':
    imbalance = ''
    task234(task, act, imbalance)
elif task == 'task3':
    imbalance = ''
    task234(task, act, imbalance)
elif task == 'task4':
    # arguments: under | over | smote
    try:
        imbalance = str(sys.argv[3])
    except IndexError:
        print("Choose another parameter: under | over | smote")
        print("You have to train the parameter first and then use the same parameter for predict or test, otherwise it fails")
        sys.exit()
    task234(task, act, imbalance)
