import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def extract_features(features_train, features_test):
    """
    Extract features from text data using TF-IDF vectorization.

    Parameters:
        features_train (pd.Series): Training text data.
        features_test (pd.Series): Testing text data.

    Returns:
        features_train_vectorized: Vectorized training features.
        features_test_vectorized: Vectorized testing features.
        vectorizer: The fitted TF-IDF vectorizer.
    """
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # Fit and transform the training data, transform the testing data
    features_train_vectorized = vectorizer.fit_transform(features_train)
    features_test_vectorized = vectorizer.transform(features_test)

    print("Feature extraction complete. Features vectorized using TF-IDF.")

    return features_train_vectorized, features_test_vectorized, vectorizer

def preprocess_and_extract_features():
    """
    Preprocess the fake news dataset and extract features.

    Returns:
        features_train_vectorized: Vectorized training features.
        features_test_vectorized: Vectorized testing features.
        target_train: Training target labels.
        target_test: Testing target labels.
        vectorizer: The fitted TF-IDF vectorizer.
    """
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(script_dir, "../dataset/Fake.csv")
    true_path = os.path.join(script_dir, "../dataset/True.csv")

    # Load datasets
    fake_data = pd.read_csv(fake_path)
    true_data = pd.read_csv(true_path)

    # Add labels
    fake_data['label'] = 0  # Fake news
    true_data['label'] = 1  # True news

    # Combine datasets
    data = pd.concat([fake_data, true_data], ignore_index=True)

    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Debugging: Check for None values in the 'text' column
    print("Checking for None values in 'text' column before processing:", data['text'].isnull().sum())

    # Ensure all None or NaN values are replaced with an empty string
    data['text'] = data['text'].fillna('')

    # Debugging: Verify no None values remain
    print("Checking for None values in 'text' column after processing:", data['text'].isnull().sum())

    # Precompute stop words
    vectorizer = TfidfVectorizer()
    stop_words = vectorizer.get_stop_words() if vectorizer.get_stop_words() else set()

    # Clean the text
    data['text'] = data['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Define the target variable
    target_column = 'label'

    # Split into features (X) and target (y)
    features = data['text']
    target = data[target_column]

    # Split into training and testing sets
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Extract features
    features_train_vectorized, features_test_vectorized, vectorizer = extract_features(features_train, features_test)

    print("Preprocessing and feature extraction complete.")

    return features_train_vectorized, features_test_vectorized, target_train, target_test, vectorizer

if __name__ == "__main__":
    preprocess_and_extract_features()
