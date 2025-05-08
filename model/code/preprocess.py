import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Define a function to clean text
def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

def preprocess_fake_news_data():
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

    # Clean the text
    data['text'] = data['text'].apply(clean_text)

    # Define the target variable
    target_column = 'label'

    # Split into features (X) and target (y)
    features = data['text']
    target = data[target_column]

    # Split into training and testing sets
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Save preprocessed data
    train_path = os.path.join(script_dir, "../dataset/train.csv")
    test_path = os.path.join(script_dir, "../dataset/test.csv")
    pd.DataFrame({'text': features_train, 'label': target_train}).to_csv(train_path, index=False)
    pd.DataFrame({'text': features_test, 'label': target_test}).to_csv(test_path, index=False)

    print("Preprocessing complete. Training and testing data saved.")

    # Return processed data
    return features_train, features_test, target_train, target_test

if __name__ == "__main__":
    preprocess_fake_news_data()
