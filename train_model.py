
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import tokenize_text

# Step 1: Load our own dataset
df = pd.read_csv("dataset/documents_dataset.csv")

X = df["text"].astype(str)
y = df["category"]

# Step 2: Divide the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

# Step 3: Create the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            tokenizer=tokenize_text,
            token_pattern=None,
            lowercase=False
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000, random_state=42)
    )
])

# Step 4: Train the model using our training data
print("\nTraining the model...")
model.fit(X_train, y_train)

print("Training completed!")

# Step 5: Predict categories for unseen test examples
y_pred = model.predict(X_test)

# Step 6: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Save the trained model
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/trained_model.joblib")

print("\nTrained model saved successfully!")
print("Location: models/trained_model.joblib")
