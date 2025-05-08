import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the datasets
fake_data = pd.read_csv('/Users/d111879/Angular/Tensor/fake_news/model/dataset/Fake.csv')
true_data = pd.read_csv('/Users/d111879/Angular/Tensor/fake_news/model/dataset/True.csv')

# Combine datasets for analysis
fake_data['label'] = 'fake'
true_data['label'] = 'true'
data = pd.concat([fake_data, true_data], ignore_index=True)

# Visualization code moved from understand.py
# 3. Data Distribution
# Numerical features (if any)
if not data.select_dtypes(include=[np.number]).empty:
    data.select_dtypes(include=[np.number]).hist(figsize=(10, 8))
    plt.suptitle("Numerical Feature Distributions")
    plt.show()

# Categorical features
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=col, order=data[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

# 4. Feature Relationships
# Correlation heatmap for numerical features
if not data.select_dtypes(include=[np.number]).empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

# Pairwise relationships (if applicable)
if not data.select_dtypes(include=[np.number]).empty:
    sns.pairplot(data.select_dtypes(include=[np.number]))
    plt.suptitle("Pairwise Relationships")
    plt.show()

# 5. Target Variable Analysis
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='label', order=data['label'].value_counts().index)
plt.title("Distribution of Target Variable (Label)")
plt.show()

# 6. Outlier Detection
numerical_cols = data.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=data, x=col)
    plt.title(f"Outliers in {col}")
    plt.show()

# 7. Text Data Analysis (if applicable)
if 'text' in data.columns:
    data['text_length'] = data['text'].apply(len)
    plt.figure(figsize=(8, 4))
    sns.histplot(data['text_length'], bins=50, kde=True)
    plt.title("Distribution of Text Length")
    plt.show()

    # Word Cloud for Fake News
    fake_text = ' '.join(fake_data['text'])
    wordcloud_fake = WordCloud(width=800, height=400).generate(fake_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_fake, interpolation='bilinear')
    plt.title("Word Cloud for Fake News")
    plt.axis("off")
    plt.show()

    # Word Cloud for True News
    true_text = ' '.join(true_data['text'])
    wordcloud_true = WordCloud(width=800, height=400).generate(true_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_true, interpolation='bilinear')
    plt.title("Word Cloud for True News")
    plt.axis("off")
    plt.show()