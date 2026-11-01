
import re

def clean_text(text):
    # Convert all letters to lowercase
    text = text.lower()

    # Remove punctuation and special symbols
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    # Clean the text first
    cleaned_text = clean_text(text)

    # Split the cleaned text into individual words
    tokens = re.findall(r"\b[a-z0-9]+\b", cleaned_text)

    return tokens


# Test our preprocessing functions
if __name__ == "__main__":
    sample_text = "Artificial Intelligence, learns from DATA!"

    print("Original text:")
    print(sample_text)

    print("\nCleaned text:")
    print(clean_text(sample_text))

    print("\nTokens:")
    print(tokenize_text(sample_text))