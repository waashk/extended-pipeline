
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

base_estimators = {
    'knn': KNeighborsClassifier()
}

default_params = {
    'knn': 	{'n_neighbors': 30, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30, 
             'p': 2, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': -1}
}

default_tuning_params = {
    #'knn': [{'weights': ['uniform', 'distance'], 'n_neighbors': [5] + list(range(10, 110, 10))}] #[10 ... 100]
    'knn': [{'weights': ['uniform'], 'n_neighbors': [5] + list(range(10, 110, 10))}] #[10 ... 100]
}

info = {
    "name_class": "knn",
    "cv": 10,
    'n_jobs': -1,
    }

def findBestK(X_train, y_train):

    estimator = base_estimators[info['name_class']]

    tunning = default_tuning_params[info['name_class']]

    gs = GridSearchCV(estimator, tunning,
                              n_jobs=info['n_jobs'],
                              cv=info['cv'],
                              verbose=1,
                              scoring='f1_macro')
    gs.fit(X_train, y_train)
    print(gs.best_score_, gs.best_params_)

    return gs.best_params_['n_neighbors']


kvalues = {
    'webkb': 20,
    'mr': 90,
    'sst1': 5,
    'sst2': 90,
    'subj': 30,
    'mpqa': 5,
    'trec': 5,
    'ohsumed': 10,
    'wos5736': 70,
    'wos11967': 100,
    'cade12': 40,
}

def findBestKsaved(dataset):
    return kvalues[dataset]
    

