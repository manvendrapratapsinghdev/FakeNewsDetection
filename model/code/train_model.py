import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from feature_extraction import preprocess_and_extract_features

def train_model():
    """
    Train a logistic regression model on the preprocessed and vectorized dataset.

    Saves the trained model to a file and prints evaluation metrics.
    """
    # Preprocess and extract features
    features_train, features_test, target_train, target_test, vectorizer = preprocess_and_extract_features()

    # Initialize the model
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Train the model
    model.fit(features_train, target_train)

    # Make predictions
    predictions = model.predict(features_test)

    # Evaluate the model
    accuracy = accuracy_score(target_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(target_test, predictions))

    # Save the trained model and vectorizer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../model/logistic_regression_model.pkl")
    vectorizer_path = os.path.join(script_dir, "../model/tfidf_vectorizer.pkl")

    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Trained model saved to {model_path}")
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")

def train_svm_model():
    """
    Train a Support Vector Machine (SVM) model on the preprocessed and vectorized dataset.

    Saves the trained model to a file and prints evaluation metrics.
    """
    # Preprocess and extract features
    features_train, features_test, target_train, target_test, _ = preprocess_and_extract_features()

    # Use a subset of the data for faster training and prediction
    features_train = features_train[:5000]
    target_train = target_train[:5000]
    features_test = features_test[:1000]
    target_test = target_test[:1000]

    # Initialize the SVM model with hyperparameters
    model = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)

    # Train the model
    model.fit(features_train, target_train)

    # Make predictions
    predictions = model.predict(features_test)

    # Evaluate the model
    accuracy = accuracy_score(target_test, predictions)
    print(f"SVM Model Accuracy: {accuracy:.2f}")
    print("SVM Classification Report:")
    print(classification_report(target_test, predictions))

    # Save the trained model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../model/svm_model.pkl")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Trained SVM model saved to {model_path}")

if __name__ == "__main__":
    train_model()
    train_svm_model()