import getopt
import sys

# compare ensemble to each standalone models for regression
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

# get the dataset
def get_dataset(data_loc=None):
    dataset = pd.read_csv(data_loc)
    if data_loc == 'data/iris.csv':
        X_train, y_train = dataset.iloc[:,0:4], dataset.iloc[:,4]
        encoder_object = LabelEncoder()
        y_train = encoder_object.fit_transform(y_train)
        return X_train, y_train, None, None
    else:
        dataset['clean_text'] = dataset['Text'].map(lambda x: re.sub('[^a-zA-Z]',' ',x))
        dataset['clean_text'] = dataset['Text'].map(lambda x: x.lower())
        train_clean = dataset[dataset.Sentiment != -12345]
        test_clean = dataset[dataset.Sentiment == -12345]
        test_clean.drop(['Sentiment'], axis=1, inplace=True)
        vect = TfidfVectorizer(ngram_range=(1,3))
        y_train = train_clean.Sentiment.values
        X_train_clean = train_clean.clean_text.values
        X_tfidf = vect.fit_transform(X_train_clean) # Using original phrase
        X_train , X_test, y_train , y_test = train_test_split(X_tfidf,y_train,test_size = 0.2)
        return X_train, y_train, X_test, y_test
 
# get a stacking ensemble of models
def get_stacking(X_train=None, y_train=None):
    # define the base models
    level0 = list()
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
def get_models(X_train=None, y_train=None):
    models = dict()
    models['RandomForest'] = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
    models['KNeighbors'] = KNeighborsClassifier(n_neighbors=2)
    models['LogisticRegression'] = LogisticRegression(random_state=RANDOM_SEED)
    models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=5, random_state=RANDOM_SEED)
    models['DecisionTree'] = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED)
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=100)
    models['Stacking'] = get_stacking(X_train,y_train)
    return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X_train, y_train):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv)
	return scores


if __name__ == '__main__':

    argvals = ' '.join(sys.argv[1:])
    if argvals != '':
        opts, args = getopt.getopt(sys.argv[1:], "hg:t:d:o:p:")
        opt_arg = dict(opts)
        if '-h' in opt_arg.keys():
            print('usage: python stacking.py')
            print('       -W ignore python stacking.py')
            print('       -W ignore python stacking.py -d <your_data_location>')
            sys.exit(0)

        data_loc = None
        if '-d' in opt_arg.keys():
            data_loc = opt_arg['-d']
        else:
            print('Unspecified Data Set: Defaulting to iris dataset')
            dataLoc = "data/iris.csv"

    print(starbanner)
    print('\tStack Ensemble Classifier Example')
    print(starbanner)
    print(beginMsg)
    print('Recording Time Elapse...\n')
    start = time.time()
    print('Accuracy\tVariance (+/-)\tClassifer\n')

    # define dataset
    X_train, y_train, X_test, y_test = get_dataset(data_loc)

    # get the models to evaluate
    models = get_models()

    # evaluate the models and store results
    results, names = list(), list()

    log = open(directory + 'stacking_classifier_metrics.csv', 'w')
    #add log1 file headers
    log.write('Classifier,')
    log.write('Accuracy,')
    log.write('Variance')
    log.write('\n')
    log.flush()

    for name, model in models.items():
        scores = evaluate_model(model, X_train, y_train)
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

    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()

    # save plot chart image as png
    pyplot.savefig(directory + 'stacking_classifier_decision_region_improved.png')

    print(completeMsg)
    display_time_elapsed(start)