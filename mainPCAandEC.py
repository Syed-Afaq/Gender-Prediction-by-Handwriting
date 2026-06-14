import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from feature_selection import *
from split_language import *
from myio import *
from metrics import *
from tunning import *

if __name__ == '__main__':
    ################################# I/O #################################
    print('Reading Files...')
    # Load dataset files (train, test, and answers)
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    answers = pd.read_csv('dataset/train_answers.csv')

    # Extract features and labels
    features_labels = list(train.columns.values[5:])
    X = train[features_labels].values
    Xtest = test[features_labels].values
    y_ = answers['male'].values
    y = y_[0:200]
    ytest = y_[200:]
    y = np.repeat(y, 4, axis=0)  # Repeat 4 times because of writer
    ytest = np.repeat(ytest, 4, axis=0)

    ########################### LANGUAGE PARTITION #############################
    print('Partition of Dataset by languages...')
    [X_arabic, X_english] = splitDataByLanguage(X)
    [y_arabic, y_english] = splitDataByLanguage(y)
    [X_arabic_test, X_english_test] = splitDataByLanguage(Xtest)
    [y_arabic_test, y_english_test] = splitDataByLanguage(ytest)

    ytest = y_english_test

    #### Feature selection ####
    nfeatures = 100

    # For English dataset
    feature_importance_english = getFeaturesImportance(
        RandomForestClassifier(n_estimators=500, max_features='sqrt', criterion='gini', n_jobs=-1),
        X_english, y_english
    )
    X_english = retainFeaturesByImportance(feature_importance_english, X_english, nfeatures)
    X_english_test = retainFeaturesByImportance(feature_importance_english, X_english_test, nfeatures)

    # For Arabic dataset
    feature_importance_arabic = getFeaturesImportance(
        RandomForestClassifier(n_estimators=500, max_features='sqrt', criterion='gini', n_jobs=-1),
        X_arabic, y_arabic
    )
    X_arabic = retainFeaturesByImportance(feature_importance_arabic, X_arabic, nfeatures)
    X_arabic_test = retainFeaturesByImportance(feature_importance_arabic, X_arabic_test, nfeatures)

    ######## RandomForest Classifier #######
    rf_clf = RandomForestClassifier(n_estimators=500, max_features='sqrt', criterion='gini', n_jobs=-1)

    ############################### Prediction for RandomForest Classifier ###################################
    english_confidences = []
    print('English Prediction...')
    rf_clf.fit(X_english, y_english)
    english_confidences = getConfidences(rf_clf, X_english_test)

    arabic_confidences = []
    print('Arabic Prediction...')
    rf_clf.fit(X_arabic, y_arabic)
    arabic_confidences = getConfidences(rf_clf, X_arabic_test)

    avg = (english_confidences + arabic_confidences) / 2.0
    logloss = log_loss(ytest, avg)

    print("Random Forest Classifier, Accuracy: %0.3f, AUC: %0.3f, Logloss: %0.3f" % (
        accuracy_score(ytest, np.around(avg)),
        roc_auc_score(ytest, avg),
        logloss
    ))

    ############################ Save the Model with Joblib #########################
    # Save the trained model using joblib
    print("Saving the trained model...")
    joblib.dump(rf_clf, 'gender_handwriting_model.pkl')

    # To load the model later, you would use:
    # rf_clf = joblib.load('gender_handwriting_model.pkl')
