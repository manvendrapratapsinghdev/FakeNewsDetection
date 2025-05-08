import pandas as pd
import numpy as np

def analyze_dataset():
    # Load the datasets
    fake_data = pd.read_csv('/Users/d111879/Angular/Tensor/fake_news/model/dataset/Fake.csv')
    true_data = pd.read_csv('/Users/d111879/Angular/Tensor/fake_news/model/dataset/True.csv')

    # Combine datasets for analysis
    fake_data['label'] = 'fake'
    true_data['label'] = 'true'
    data = pd.concat([fake_data, true_data], ignore_index=True)

    # 1. Data Overview
    print("Data Overview:")
    print(data.info())
    print("\nData Types:")
    print(data.dtypes)
    print("\nSummary Statistics:")
    print(data.describe())
    print("\nFirst 5 Rows:")
    print(data.head())

    # 2. Data Quality Checks
    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nDuplicate Rows:")
    print(data.duplicated().sum())

    # 3. Data Distribution
    # Numerical features (if any)
    if not data.select_dtypes(include=[np.number]).empty:
        print("Numerical Feature Distributions:")
        print(data.select_dtypes(include=[np.number]).describe())

    # Categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"Distribution of {col}:")
        print(data[col].value_counts())

    # 4. Feature Relationships
    # Correlation heatmap for numerical features
    if not data.select_dtypes(include=[np.number]).empty:
        print("Correlation Matrix:")
        print(data.corr())

    # Pairwise relationships (if applicable)
    if not data.select_dtypes(include=[np.number]).empty:
        print("Pairwise Relationships Summary:")
        print(data.select_dtypes(include=[np.number]).corr())

    # 5. Target Variable Analysis
    print("Distribution of Target Variable (Label):")
    print(data['label'].value_counts())

    # 6. Outlier Detection
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        print(f"Outliers in {col}:")
        print(data[col].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))

    # 7. Text Data Analysis (if applicable)
    if 'text' in data.columns:
        data['text_length'] = data['text'].apply(len)
        print("Text Length Distribution:")
        print(data['text_length'].describe())

        fake_text = " ".join(fake_data['text'].dropna())
        true_text = " ".join(true_data['text'].dropna())
        print("Sample Fake Text:", fake_text[:500])
        print("Sample True Text:", true_text[:500])

    # Save the cleaned and analyzed data (optional)
    data.to_csv('/Users/d111879/Angular/Tensor/fake_news/model/dataset/combined_cleaned.csv', index=False)
    print("Data analysis completed and combined dataset saved.")

analyze_dataset()
