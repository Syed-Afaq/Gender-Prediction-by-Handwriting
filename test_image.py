import joblib
import cv2
import numpy as np
from feature_selection import getFeaturesImportance, retainFeaturesByImportance, extract_features_from_image

# Load your pre-trained model
def load_model(model_path):
    return joblib.load(model_path)

# Function to preprocess the image
def preprocess_image(image_path, model, X_train, y_train):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features = extract_features_from_image(image)

    # Get feature importance using training data
    feature_importance = getFeaturesImportance(model, X_train, y_train)

    # Select the top 100 features
    features_selected = retainFeaturesByImportance(feature_importance, features.reshape(1, -1), 100)

    return features_selected

# Function to predict the gender based on image
def predict_gender(model, image_path, X_train, y_train):
    features = preprocess_image(image_path, model, X_train, y_train)
    prediction = model.predict(features)
    return prediction

# Main execution
if __name__ == '__main__':
    # Load the trained model
    model_path = 'gender_handwriting_model.pkl'
    model = load_model(model_path)

    # Load or define your training data (X_train and y_train)
    X_train = np.random.rand(100, 10)  # Example: 100 samples, 10 features (Replace with actual data)
    y_train = np.random.randint(0, 2, 100)  # Example: 100 labels (0 or 1)

    # Specify the image path to test
    image_path = 'testImage6.jpg'

    # Predict the gender based on the image
    prediction = predict_gender(model, image_path, X_train, y_train)
    print(f'Predicted Gender: {prediction}')
