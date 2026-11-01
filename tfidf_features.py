
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import tokenize_text

# Load our own dataset
df = pd.read_csv("dataset/documents_dataset.csv")

# Get the text examples
texts = df["text"].astype(str)

# Create the TF-IDF feature extractor
vectorizer = TfidfVectorizer(
    tokenizer=tokenize_text,
    token_pattern=None,
    lowercase=False
)

# Convert text into numerical features
tfidf_matrix = vectorizer.fit_transform(texts)

print("Total text examples:", len(texts))
print("Number of unique word features:", len(vectorizer.get_feature_names_out()))
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Display some words learned from the dataset
print("\nFirst 20 vocabulary words:")
print(vectorizer.get_feature_names_out()[:20])

# Display the numerical representation of the first example
print("\nFirst text example:")
print(texts.iloc[0])

print("\nIts TF-IDF numerical features:")
print(tfidf_matrix[0].toarray())
