# import warnings filter
import warnings

# compare ensemble to each standalone models for regression
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import time
from IPython.display import display
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from matplotlib import pyplot

#static variables
starbanner = '\n******************************************************************\n'
beginMsg = "Beginning Classification Modeling...\n"
completeMsg = "\n******** Completed Classification Modeling ********\n"
directory = '/content/CIS-700-final/results/'
RANDOM_SEED = 0
iris_dataset = pd.read_csv("data/iris.csv")


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
def get_dataset():
    X, y = iris_dataset.iloc[:,0:4], iris_dataset.iloc[:,4]
    encoder_object = LabelEncoder()
    y = encoder_object.fit_transform(y)
    return X, y
 
# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED))
    level0.append(KNeighborsClassifier(n_neighbors=2))
    level0.append(GaussianNB())
    level0.append(LogisticRegression(random_state=RANDOM_SEED))
    level0.append(ExtraTreesClassifier(n_estimators=5, random_state=RANDOM_SEED))
    level0.append(DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED))
    level0.append(AdaBoostClassifier(n_estimators=100))
    
    # define meta learner model
    level1 = LogisticRegression(random_state=RANDOM_SEED)
    # define the stacking ensemble
    model = StackingClassifier(classifiers=level0, meta_classifier=level1, use_probas=True, average_probas=False)
    return model
 
# get a list of models to evaluate
def get_models():
    models = dict()
    models['RandomForest'] = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
    models['KNeighbors'] = KNeighborsClassifier(n_neighbors=2)
    models['GaussianNB'] = GaussianNB()
    models['LogisticRegression'] = LogisticRegression(random_state=RANDOM_SEED)
    models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=5, random_state=RANDOM_SEED)
    models['DecisionTree'] = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED)
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=100)
    models['Stacking'] = get_stacking()
    return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
	return scores

print(starbanner)
print('\tStack Ensemble Classifier Example')
print(starbanner)
print(beginMsg)
print('Recording Time Elapse...\n')
start = time.time()
print('Accuracy\tVariance (+/-)\tClassifer\n')

# define dataset
X, y = get_dataset()

# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()

log = open(directory + 'stacking_classifier_metrics.csv', 'w')
#add log1 file headers
log.write('Classifier,')
log.write('Accuracy,')
log.write('Variance,')
log.write('\n')
log.flush()

for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    acc = mean(scores)
    std = std(scores)
    log.write(str(acc) + ',')
    log.write(str(std) + ',')
    #print('Accuracy: %.3f \t Variance: (+/-) (%.3f) \tClassifier: %s  ' % (mean(scores), std(scores), name))
    print('%.3f\t\t%.3f\t\t%s  ' % acc, std, name))

log.close()

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
display(pyplot.show())

# save plot chart image as png
pyplot.savefig(directory + 'stacking_classifier_decision_region_improved.png')

print(completeMsg)
display_time_elapsed(start)