import os
import joblib
from sklearn.metrics import accuracy_score, classification_report
from feature_extraction import preprocess_and_extract_features

def compare_models():
    """
    Compare the accuracy of two models using the same test dataset.
    """
    # Preprocess and extract features
    _, features_test, _, target_test, _ = preprocess_and_extract_features()

    # Define paths to the models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model1_path = os.path.join(script_dir, "../model/logistic_regression_model.pkl")
    model3_path = os.path.join(script_dir, "../model/svm_model.pkl")

    # Load the models
    model1 = joblib.load(model1_path)
    model3 = joblib.load(model3_path)

    # Evaluate Model 1
    predictions1 = model1.predict(features_test)
    accuracy1 = accuracy_score(target_test, predictions1)
    print("Model 1 Accuracy:", accuracy1)
    print("Model 1 Classification Report:")
    print(classification_report(target_test, predictions1))

    # Evaluate Model 3
    predictions3 = model3.predict(features_test)
    accuracy3 = accuracy_score(target_test, predictions3)
    print("Model 3 Accuracy:", accuracy3)
    print("Model 3 Classification Report:")
    print(classification_report(target_test, predictions3))

    # Compare and decide
    accuracies = [accuracy1, accuracy3]
    best_model_index = accuracies.index(max(accuracies)) + 1
    print(f"Model {best_model_index} is the best for further processing based on accuracy.")

if __name__ == "__main__":
    compare_models()
