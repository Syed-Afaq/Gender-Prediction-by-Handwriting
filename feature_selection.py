import numpy as np
import cv2

def getRankings(clf, ranks, key):
    clf_ = clf[key]
    for i in range(0, ranks, 1):
        clf_.fit(X1, y1)
        feature_importance = clf_.feature_importances_
        saveToMatlabFile(key + '_imp_' + str(i), {'importance': feature_importance})

def getFeaturesImportance(clf, X, y):
    clf.fit(X, y)
    feature_importance = clf.feature_importances_
    return feature_importance

def retainFeaturesByImportance(feature_importance, X1, number_features):
    sorted_idx = np.argsort(feature_importance)[::-1]
    if X1.ndim == 1:  # If X1 is 1D, reshape it to 2D
        X1 = X1.reshape(1, -1)
    X1 = X1[:, sorted_idx[:number_features]]  # Ensure it selects the correct number of features
    return X1

def retainFeaturesByThreshold(feature_importance, X1, fi_threshold):
    important_idx = np.where(feature_importance > fi_threshold)[0]
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    if X1.ndim == 1:  # If X1 is 1D, reshape it to 2D
        X1 = X1.reshape(1, -1)
    X1 = X1[:, important_idx][:, sorted_idx]
    return X1

def extract_features_from_image(image):
    # Example feature extraction: flattening the image into a vector
    image_resized = cv2.resize(image, (100, 100))  # Resize the image to a fixed size
    features = image_resized.flatten()  # Flatten the image into a 1D vector (you can replace this with actual features)
    return features
