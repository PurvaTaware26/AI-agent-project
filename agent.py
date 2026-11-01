
import re
from pathlib import Path

import pymupdf as fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import tokenize_text

DOCUMENTS_FOLDER = Path("documents")
RELEVANCE_THRESHOLD = 0.25


def read_documents():
    """Read text from TXT and PDF files."""
    documents = []

    for file_path in DOCUMENTS_FOLDER.iterdir():
        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

        elif file_path.suffix.lower() == ".pdf":
            blocks = []

            with fitz.open(file_path) as pdf:
                for page in pdf:
                    page_blocks = page.get_text("blocks")

                    for block in page_blocks:
                        block_text = block[4].strip()

                        if block_text:
                            blocks.append(block_text)

            # Keep separate PDF text blocks separate
            text = "\n\n".join(blocks)

        else:
            continue

        if text.strip():
            documents.append({
                "filename": file_path.name,
                "text": text
            })

    return documents

def create_passages(documents):
    """Combine nearby PDF paragraphs into useful passages."""

    passages = []

    for document in documents:
        text = document["text"]
        text = text.replace("\x00", " ")
        text = text.replace("\ufffd", " ")

        # Separate the original text blocks
        paragraphs = re.split(r"\n\s*\n", text)

        # Clean each paragraph but preserve its position
        cleaned = []

        for paragraph in paragraphs:
            paragraph = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph)
            paragraph = re.sub(r"\s+", " ", paragraph).strip()

            if len(paragraph.split()) >= 2:
                cleaned.append(paragraph)

        # Combine nearby paragraphs so lists and explanations stay together
        group_size = 6
        step_size = 3

        for i in range(0, len(cleaned), step_size):
            group = cleaned[i:i + group_size]
            passage_text = " ".join(group)

            if len(passage_text.split()) >= 5:
                passages.append({
                    "filename": document["filename"],
                    "text": passage_text
                })

            if i + group_size >= len(cleaned):
                break

    return passages


def search_documents(question, passages):
    """Rank passages by topic relevance and avoid contents pages."""

    if not passages:
        return None, 0.0

    texts = [p["text"] for p in passages]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_text,
        token_pattern=None,
        lowercase=False
    )

    matrix = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([question])

    if question_vector.nnz == 0:
        return None, 0.0

    similarities = cosine_similarity(
        question_vector, matrix
    )[0]

    question_words = set(tokenize_text(question))

    common_words = {
        "what", "is", "are", "was", "were", "the", "a", "an",
        "of", "in", "on", "for", "to", "how", "why", "who",
        "when", "where", "do", "does", "can", "explain", "tell",
        "me", "about", "please", "used", "and", "five", "list",
        "give", "describe"
    }

    keywords = question_words - common_words

    # Prefer passages that contain the actual topic.
    ranked = []

    for i, passage in enumerate(passages):
        text = passage["text"]
        words = set(tokenize_text(text))
        overlap = keywords & words

        if not overlap:
            continue

        score = float(similarities[i])

        # Boost passages containing the specific topic phrase.
        lower_text = text.lower()
        lower_question = question.lower()

        if "essential characteristics" in lower_question:
            if "essential characteristics" in lower_text:
                score += 0.35
            if "table of contents" in lower_text:
                score -= 0.5

        # Penalize likely contents lists with many headings.
        if text.count(",") >= 6 and len(text.split()) < 100:
            score -= 0.15

        ranked.append((score, i))

    if not ranked:
        return None, 0.0

    ranked.sort(reverse=True)

    best_score, best_index = ranked[0]

    if similarities[best_index] < RELEVANCE_THRESHOLD:
        return None, float(similarities[best_index])

    # Include a second passage only when it has a strong topic match.
    # Start with the single most relevant passage
    selected = [best_index]

    # Remove duplicate passages and preserve document order.
    selected.sort()

    result_text = []
    seen = set()

    for index in selected:
        text = passages[index]["text"].strip()

        if text not in seen:
            result_text.append(text)
            seen.add(text)

    filenames = list(dict.fromkeys(
        passages[i]["filename"] for i in selected
    ))

    return {
        "filename": ", ".join(filenames),
        "text": "\n\n".join(result_text)
    }, float(similarities[best_index])


def ask_agent(question):
    documents = read_documents()
    passages = create_passages(documents)

    if not passages:
        return "No readable documents found in the documents folder."

    result, score = search_documents(question, passages)

    if result is None:
        return (
            "I could not find sufficient relevant information "
            "in the available documents to answer this question."
        )

    return (
        f"Answer from your documents:\n{result['text']}\n\n"
        f"Source: {result['filename']}\n"
        f"Retrieval similarity: {score:.2f}\n\n"
        "Note: This is a retrieved passage, not a newly generated answer."
    )


def summarize_documents(documents, sentence_count=3):
    """Create an extractive summary from available documents."""

    sentences = []

    for document in documents:
        parts = re.split(
            r"(?<=[.!?])\s+|\n+",
            document["text"]
        )

        for part in parts:
            part = part.strip()

            if len(part.split()) >= 4:
                sentences.append({
                    "text": part,
                    "filename": document["filename"]
                })

    if not sentences:
        return "No suitable text found to summarize."

    # Convert sentences into numerical TF-IDF features
    texts = [item["text"] for item in sentences]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_text,
        token_pattern=None,
        lowercase=False
    )

    matrix = vectorizer.fit_transform(texts)

    # Give each sentence a score based on its TF-IDF values
    scores = matrix.sum(axis=1).A1

    # Select the highest-scoring sentences
    count = min(sentence_count, len(sentences))
    selected_indices = sorted(
        scores.argsort()[-count:]
    )

    summary = [
        sentences[index]["text"]
        for index in selected_indices
    ]

    return "\n".join(
        f"{number}. {sentence}"
        for number, sentence in enumerate(summary, start=1)
    )

if __name__ == "__main__":
    print("AI DOCUMENT INTELLIGENCE AGENT")
    print("Ask a question about your documents.")
    print("Type exit to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "exit":
            break
        
        if question.lower() == "summary":
            documents = read_documents()
            print("\nDOCUMENT SUMMARY\n")
            print(summarize_documents(documents))
            print("\n" + "-" * 50 + "\n")
            continue

        if not question:
            print("Please enter a question.\n")
            continue

        print("\n" + ask_agent(question))
        print("\n" + "-" * 50 + "\n")
