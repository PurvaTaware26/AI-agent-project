
import streamlit as st
from pathlib import Path
from qa_engine import answer_question
from agent import ask_agent, read_documents, summarize_documents
st.set_page_config(
    page_title="AI Document Intelligence Agent",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Intelligence Agent")

st.write(
    "An AI project powered by a machine-learning model "
    "trained on our own dataset."
)



# Sidebar
st.sidebar.header("Project Features")
st.sidebar.write("• Document question answering")
st.sidebar.write("• Extractive summarization")
st.sidebar.write("• PDF and TXT document support")
st.sidebar.write("• Rejection of unsupported questions")
st.sidebar.write("• Custom-trained text classifier")

st.sidebar.caption(
    "Training dataset: dataset/documents_dataset.csv"
)

# Tabs
ask_tab, summary_tab, files_tab = st.tabs(
    ["Ask Questions", "Summarize Documents", "Available Documents"]
)

with ask_tab:
    st.subheader("Ask a question")

    question = st.text_input(
        "Enter a question about your documents",
        placeholder="Example: What is TF-IDF used for?"
    )

    if st.button("Search Documents", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching your documents..."):
                answer = answer = answer_question(question)

            st.markdown("### Answer")
            st.write(answer)

with summary_tab:
    st.subheader("Document Summary")

    number_of_sentences = st.slider(
        "Number of sentences",
        min_value=1,
        max_value=10,
        value=3
    )

    if st.button("Generate Summary"):
        with st.spinner("Reading your documents..."):
            documents = read_documents()

            if not documents:
                st.warning("Add a TXT or PDF file to the documents folder.")
            else:
                summary = summarize_documents(
                    documents,
                    sentence_count=number_of_sentences
                )
                st.write(summary)

with files_tab:
    st.subheader("Upload Your Documents")

    documents_folder = Path("documents")
    documents_folder.mkdir(parents=True, exist_ok=True)

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = documents_folder / Path(uploaded_file.name).name

            file_path.write_bytes(uploaded_file.getvalue())

        st.success("Your documents have been saved successfully!")

    st.divider()
    st.subheader("Available Documents")

    documents = read_documents()

    if documents:
        for document in documents:
            st.write("📄", document["filename"])
            st.caption(
                f"{len(document['text'].split())} words"
            )
    else:
        st.info("Upload a PDF or TXT document to get started.")


st.divider()

st.caption(
    "Developed as an Introduction to AI project | "
    "Python • Scikit-learn • TF-IDF • Streamlit"
)
