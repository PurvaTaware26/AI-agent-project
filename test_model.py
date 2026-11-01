
import joblib

# Load the model we trained ourselves
model = joblib.load("models/trained_model.joblib")

# New sentences that were not in our training dataset
test_sentences = [
    "Neural networks learn patterns from examples.",
    "Python functions help organize reusable code.",
    "Virtual machines run on cloud infrastructure.",
    "Encryption helps protect confidential information.",
    "Data visualization helps us understand trends."
]

print("AI DOCUMENT INTELLIGENCE AGENT")
print("-" * 40)

for sentence in test_sentences:
    # Predict the category
    prediction = model.predict([sentence])[0]

    # Get the highest predicted probability
    probabilities = model.predict_proba([sentence])[0]
    confidence = max(probabilities) * 100

    print("\nInput:", sentence)
    print("Predicted category:", prediction)
    print("Model confidence:", round(confidence, 2), "%")
