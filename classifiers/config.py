

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

base_estimators = {
    'svm': svm.SVC(),
    'knn': KNeighborsClassifier(),
}

default_params = {
    'svm': 	{'kernel': 'linear', 'C': 1, 'verbose': False, 'probability': False,
             'degree': 3, 'shrinking': True,
             'decision_function_shape': None,  # 'random_state': None,
             'tol': 0.001, 'cache_size': 25000, 'coef0': 0.0, 'gamma': 'auto',
             'class_weight': None, 'random_state': 1608637542},  # ,'max_iter':-1},
    'knn': 	{'n_neighbors': 30, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30,  # 30,
             'p': 2, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': -1},
}

default_tuning_params = {
    'svm': 	[{'C': 2.0 ** np.arange(-5, 15, 2)}],
    'knn': [{'weights': ['uniform', 'distance'], 'n_neighbors': [10, 25, 50, 75, 100]}],
}


def getClassifierInfo(classifier: str) -> dict:

    if classifier == 'svm':
        info = {
            "name_class": "svm",
            "cv": 10,
            'n_jobs': -1,
            "max_iter": 1000000,
        }
    if classifier == 'knn':
        info = {
            "name_class": "knn",
            "cv": 10,
            'n_jobs': -1,
        }
    return info