
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent import ask_agent

DATASET_FOLDER = Path(__file__).parent / "dataset"


def load_qa_data():
    dataset_files = list(
        DATASET_FOLDER.glob("*_qa_training_data.csv")
    )

    if not dataset_files:
        raise FileNotFoundError(
            f"No Q&A datasets found in {DATASET_FOLDER}"
        )

    dataframes = []

    for file_path in dataset_files:
        data = pd.read_csv(file_path)

        required_columns = {"question", "answer", "topic"}

        if not required_columns.issubset(data.columns):
            raise ValueError(
                f"{file_path.name} must contain "
                "question, answer, and topic columns."
            )

        dataframes.append(
            data.dropna(subset=["question", "answer"])
        )

    return pd.concat(dataframes, ignore_index=True)


def answer_question(question):
    """Match relevant questions or search uploaded documents."""

    data = load_qa_data()
    question = question.strip()

    if not question:
        return "Please enter a question."

    questions = data["question"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        strip_accents="unicode"
    )

    question_matrix = vectorizer.fit_transform(questions)
    user_vector = vectorizer.transform([question])

    if user_vector.nnz > 0:
        scores = cosine_similarity(
            user_vector, question_matrix
        )[0]

        best_index = scores.argmax()
        best_score = scores[best_index]
        row = data.iloc[best_index]

        # Check whether the question contains meaningful
        # words that also appear in the matched question.
        stop_words = {
            "what", "when", "where", "which", "who",
            "why", "how", "does", "do", "is", "are",
            "the", "a", "an", "of", "in", "on", "for",
            "to", "and", "explain", "define", "describe",
            "tell", "me", "about"
        }

        user_words = {
            word.lower().strip(".,?!")
            for word in question.split()
        } - stop_words

        matched_words = {
            word.lower().strip(".,?!")
            for word in str(row["question"]).split()
        } - stop_words

        common_words = user_words.intersection(matched_words)

        # Accept a dataset answer only when there is
        # meaningful topic overlap and sufficient similarity.
        if best_score >= 0.30 and common_words:
            return (
                f"Answer:\n{row['answer']}\n\n"
                f"Topic: {row['topic']}\n"
                f"Matched question: {row['question']}\n"
                f"Question similarity: {best_score:.2f}\n\n"
                "Source: Your question-answer dataset"
            )

    # No suitable dataset match: search uploaded documents.
    document_answer = ask_agent(question)

    # If the documents also cannot answer, explain that
    # the information was not found.
    if (
        not document_answer
        or "could not find" in document_answer.lower()
        or "no relevant" in document_answer.lower()
    ):
        return (
            "I couldn't find a suitable answer in your "
            "Q&A dataset or uploaded documents. "
            "Please add relevant notes or training examples."
        )

    return document_answer

if __name__ == "__main__":
    print("AI Document Intelligence Agent")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            break

        if question:
            try:
                print("\n" + answer_question(question))
            except Exception as error:
                print(f"Error: {error}")
