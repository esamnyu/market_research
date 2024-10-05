# tody_cleaning_reviews.py

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

import spacy
import nltk
from nltk.corpus import stopwords

from textblob import TextBlob
from collections import Counter

# Initialize spaCy English model
nlp = spacy.load('en_core_web_sm')

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the reviews from the Excel file
# Adjust the file path as needed
df = pd.read_excel('/Users/ethansam/wheel_strategy_project/tody_cleaning_reviews.xlsx')

# Ensure the correct column name for reviews
# Replace 'Review Text' with the actual column name if different
if 'Review Text' in df.columns:
    review_column = 'Review Text'
elif 'Review' in df.columns:
    review_column = 'Review'
else:
    raise ValueError("The DataFrame does not contain a 'Review Text' or 'Review' column.")

# Preprocess the reviews
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text

# Apply the preprocessing function
df['Cleaned_Review'] = df[review_column].astype(str).apply(preprocess_text)

# Remove stop words
stop_words = set(stopwords.words('english'))
df['Cleaned_Review'] = df['Cleaned_Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Optional: Lemmatization
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

df['Cleaned_Review'] = df['Cleaned_Review'].apply(lemmatize_text)

# Sentiment Analysis
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Apply the function
df['Polarity'] = df['Cleaned_Review'].apply(get_polarity)

# Classify sentiments
df['Sentiment'] = df['Polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# View sentiment counts
print("Sentiment analysis results:")
sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)
print("\n")

# Keyword Extraction using spaCy
def extract_keywords(text):
    doc = nlp(text)
    # Extract nouns and proper nouns that are not stop words
    keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop]
    return keywords

df['Keywords'] = df['Cleaned_Review'].apply(extract_keywords)

# Flatten list of keywords for all reviews
all_keywords = [keyword for sublist in df['Keywords'] for keyword in sublist]

# Remove the word cloud generation and replace it with more human-readable outputs

# ---------------------------------------------
# Generate Summary and Insights
# ---------------------------------------------

print("\n--- Summary and Insights ---\n")

# Step 1: Analyze Sentiment Distribution (already done)
print("Sentiment Distribution:")
print(sentiment_counts)
print("\n")

# Plot sentiment distribution
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.show()

# Step 2: Identify Common Themes in Positive Reviews
# Filter positive reviews
positive_reviews = df[df['Sentiment'] == 'Positive']

# Combine all keywords from positive reviews
positive_keywords = [keyword for sublist in positive_reviews['Keywords'] for keyword in sublist]

# Count the frequency of each keyword
positive_keyword_counts = Counter(positive_keywords)

# Get the top
