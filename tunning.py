from sklearn.model_selection import GridSearchCV
from sklearn.base import is_classifier
import numpy as np
import random

def findBestParams(clf, param_grid, X, y, nfold = 5):
    grid_search = GridSearchCV(clf, param_grid, cv=nfold, n_jobs=-1)
    grid_search.fit(X, y)

    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

    return grid_search.best_params_
