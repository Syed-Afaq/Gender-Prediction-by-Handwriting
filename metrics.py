from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
from myutils import cloneEstimator
from sklearn import base
import numpy as np
from feature_selection import *
from split_language import *

def getConfidences( clf, X ):
    if 'predict_proba' in dir(clf):
        return clf.predict_proba(X)[:,1]

    confidences = clf.decision_function(X)
    confidences -= np.min(confidences)
    confidences /= np.max(confidences)
    
    return confidences

def getTestScores( groundtruth, predicted, confidences ):
    return [accuracy_score(groundtruth, predicted), roc_auc_score(groundtruth, confidences), log_loss(groundtruth, confidences)]

def getKx2CVScores( clf, X, y, k = 5):
    accuracies = np.zeros((k, 2))
    AUCs = np.zeros((k, 2))
    logLoss = np.zeros((k, 2))

    for i in range(k):
        clf12 = cloneEstimator(clf)
        clf21 = cloneEstimator(clf)

        X1, X2, y1, y2 = train_test_split(X, y, test_size = 0.5)
        
        clf12.fit(X1, y1)
        clf21.fit(X2, y2)

        confidences21 = getConfidences(clf21, X1)
        confidences12 = getConfidences(clf12, X2)
        
        accuracies[i][0] = accuracy_score(y1, np.around(confidences21))
        accuracies[i][1] = accuracy_score(y2, np.around(confidences12))

        AUCs[i][0] = roc_auc_score(y1, confidences21)
        AUCs[i][1] = roc_auc_score(y2, confidences12)
        
        logLoss[i][0] = log_loss(y1, confidences21)
        logLoss[i][1] = log_loss(y2, confidences12)

    return [accuracies, AUCs, logLoss]
