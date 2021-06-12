import getopt
import sys
import pandas as pd
import numpy as np
import re
from numpy import mean
from numpy import std
import time
from IPython.display import display
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold, GridSearchCV
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from matplotlib import pyplot

'''
******************************************************************************************
*   This file compare stacking ensemble to each standalone classifier models for indicated
*   dataset with either standard settings or adjusted_settings when indicated.
*   Datasets allowed are found in /data directory.
*   Results are stored in /results directory.
******************************************************************************************
'''

#static variables
starbanner = '\n******************************************************************\n'
beginMsg = "Beginning Classification Modeling...\n"
completeMsg = "\n******** Completed Classification Modeling ********\n"
directory = '/content/CIS-700-final/results/'
RANDOM_SEED = 0

def display_time_elapsed(start):
    if start > 0 :
        s = time.time() - start
        hours, remainder = divmod(s, 3600)
        minutes, seconds = divmod(remainder, 60)
        te = 'Time Elapsed: '
        if(hours > 0):
            te += '{:02}hrs:'.format(int(hours))
        if(hours > 0 or minutes > 0):
            te += '{:02}mins:'.format(int(minutes))
        te += '{:02}secs\n'.format(int(seconds))
        print(te)

# get the dataset for the passed data location
def get_dataset(data_loc=None):
    dataset = pd.read_csv(data_loc)
    # if iris (originally used to test functionallity) return iris data set split, encoded/transformed
    if data_loc == 'data/iris.csv':
        X_train, y_train = dataset.iloc[:,0:4], dataset.iloc[:,4]
        encoder_object = LabelEncoder()
        y_train = encoder_object.fit_transform(y_train)
        return X_train, y_train, None, None
    else:
        # else is assumed eapoe data set
        # lets first clean it up a bit, we will start by converting all text to lowercase for standardization
        dataset['clean_text'] = dataset['Text'].map(lambda x: re.sub('[^a-zA-Z]',' ',x))
        dataset['clean_text'] = dataset['Text'].map(lambda x: x.lower())
        # here, we will split the data set by those who have sentiment values (training) and those who dont (test)
        train_clean = dataset[dataset.Sentiment != -12345]
        test_clean = dataset[dataset.Sentiment == -12345]
        test_clean.drop(['Sentiment'], axis=1, inplace=True)
        # setup the vectorizor, note TfidfVectorizer is of feature_extraction nature
        vect = TfidfVectorizer(ngram_range=(1,3))
        y_train = train_clean.Sentiment.values
        X_train_clean = train_clean.clean_text.values
         # Using original phrase, apply feature_extraction vectorizor
        X_tfidf = vect.fit_transform(X_train_clean)
        # split the data
        X_train , X_test, y_train , y_test = train_test_split(X_tfidf,y_train,test_size = 0.2)
        return X_train, y_train, X_test, y_test
 
# get a stacking ensemble of models
# return settings adjusted stacking ensemble of models if adjust_settings is set, else standard stacking ensemble of models
def get_stacking(X_train=None, y_train=None, adjust_settings=None):
    # define the base models
    level0 = list()
    if adjust_settings is not None:
        level0.append(RandomForestClassifier(n_estimators=15, random_state=1))
        level0.append(KNeighborsClassifier(n_neighbors=3))
        level0.append(LogisticRegression(random_state=1))
        level0.append(ExtraTreesClassifier(n_estimators=8, random_state=1))
        level0.append(DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=1, splitter='random', min_samples_split=3))
        level0.append(AdaBoostClassifier(n_estimators=200))
        # define meta learner model
        level1 = LogisticRegression(random_state=1)
    else:
        level0.append(RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED))
        level0.append(KNeighborsClassifier(n_neighbors=2))
        level0.append(LogisticRegression(random_state=RANDOM_SEED))
        level0.append(ExtraTreesClassifier(n_estimators=5, random_state=RANDOM_SEED))
        level0.append(DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED))
        level0.append(AdaBoostClassifier(n_estimators=100))
        # define meta learner model
        level1 = LogisticRegression(random_state=RANDOM_SEED)
    
    # define the stacking ensemble
    model = StackingClassifier(classifiers=level0, meta_classifier=level1, use_probas=True, average_probas=False)
    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train)
    return model
 
# get a list of models to evaluate
# return settings adjusted models if adjust_settings is set, else standard models
def get_models(X_train=None, y_train=None, adjust_settings=None):
    models = dict()
    if adjust_settings is not None:
        models['RandomForest'] = RandomForestClassifier(n_estimators=15, random_state=1)
        models['KNeighbors'] = KNeighborsClassifier(n_neighbors=3)
        models['LogisticRegression'] = LogisticRegression(random_state=1)
        models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=8, random_state=1)
        models['DecisionTree'] = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=1, splitter='random', min_samples_split=3)
        models['AdaBoost'] = AdaBoostClassifier(n_estimators=200)
    else:
        models['RandomForest'] = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
        models['KNeighbors'] = KNeighborsClassifier(n_neighbors=2)
        models['LogisticRegression'] = LogisticRegression(random_state=RANDOM_SEED)
        models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=5, random_state=RANDOM_SEED)
        models['DecisionTree'] = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED)
        models['AdaBoost'] = AdaBoostClassifier(n_estimators=100)
    models['Stacking'] = get_stacking(X_train,y_train, adjust_settings)
    return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X_train, y_train, adjust_settings=None):
	if adjust_settings is not None:
		cv = RepeatedKFold(n_splits=15, n_repeats=5, random_state=RANDOM_SEED)
	else:
		cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv)
	return scores


if __name__ == '__main__':

    argvals = ' '.join(sys.argv[1:])
    if argvals != '':
        opts, args = getopt.getopt(sys.argv[1:], "hg:t:d:o:p:a:")
        opt_arg = dict(opts)
        if '-h' in opt_arg.keys():
            print('usage: python stacking.py')
            print('       -W ignore python stacking.py')
            print('       -W ignore python stacking.py -d <your_data_location>')
            print('       -W ignore python stacking.py -d <your_data_location> -a <settings> ')
            sys.exit(0)

        data_loc = None
        if '-d' in opt_arg.keys():
            data_loc = opt_arg['-d']
        else:
            print('Unspecified Data Set: Defaulting to iris dataset')
            dataLoc = "data/iris.csv"

        adjust_settings = None
        if '-a' in opt_arg.keys():
            adjust_settings = opt_arg['-a']

    print(starbanner)
    print('\tStack Ensemble Classifier Example')
    print(starbanner)
    print(beginMsg)
    print('Recording Time Elapse...\n')
    if adjust_settings is not None:
        print('Running Classifiers With Adjusted Settings...\n')
    start = time.time()
    print('Accuracy\tVariance (+/-)\tClassifer\n')

    # define dataset
    X_train, y_train, X_test, y_test = get_dataset(data_loc)

    # get the models to evaluate
    models = get_models(X_train, y_train, adjust_settings)

    # evaluate the models and store results
    results, names = list(), list()
    
    adjusted = ''
    if adjust_settings is not None:
        adjusted = '_adjusted'

    log = open(directory + 'stacking_classifier_metrics' + adjusted + '.csv', 'w')
    #add log file headers
    log.write('Classifier,')
    log.write('Accuracy,')
    log.write('Variance')
    log.write('\n')
    log.flush()

    for name, model in models.items():
        scores = evaluate_model(model, X_train, y_train, adjust_settings)
        results.append(scores)
        names.append(name)
        acc_r = np.round(scores.mean(),4)
        std_r = np.round(scores.std(),4)
        log.write(name + ',')
        log.write(str(acc_r) + ',')
        log.write(str(std_r))
        log.write('\n')
        log.flush()
        print('%.3f\t\t%.3f\t\t%s  ' % (acc_r, std_r, name))

    log.close()

    print(completeMsg)
    display_time_elapsed(start)