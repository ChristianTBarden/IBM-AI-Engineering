import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline
import gradio as gr

# NEW HuggingFace LangChain integration
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

# Avoid Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Model Config ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"  # You can use "google/flan-t5-small" if memory is tight


def document_loader(file_path):
    """Load PDF and return documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(docs):
    """Split docs into smaller chunks (to avoid token limit issues)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # reduced from 1000 for safer T5 token limits
        chunk_overlap=50
    )
    return splitter.split_documents(docs)


def create_embeddings():
    """Return HuggingFace embeddings using CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )


def create_vectorstore(chunks):
    """Create Chroma vector DB from doc chunks."""
    embeddings = create_embeddings()
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def create_qa_chain(vectorstore):
    """Set up local LLM-based QA chain using HuggingFace Pipeline."""
    retriever = vectorstore.as_retriever()

    local_pipeline = pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        max_length=512,
        do_sample=False  # deterministic output
        # No return_full_text (not supported in text2text-generation)
    )

    llm = HuggingFacePipeline(pipeline=local_pipeline)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def retriever_qa(pdf_file, question):
    """Main function to process PDF + answer the question."""
    if not pdf_file:
        return "Please upload a PDF file."
    if not question.strip():
        return "Please enter a question."

    docs = document_loader(pdf_file.name)
    chunks = split_documents(docs)
    vectordb = create_vectorstore(chunks)
    qa_chain = create_qa_chain(vectordb)

    response = qa_chain.invoke({"query": question})
    return response.get("result", "No answer found.")


# Gradio UI
interface = gr.Interface(
    fn=retriever_qa,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a question")],
    outputs="text",
    title="📄 PDF Q&A (Offline)",
    description="Ask questions from a PDF using an offline Hugging Face model. No API needed!"
)

if __name__ == "__main__":
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"LLM: {LLM_MODEL_NAME}")
    interface.launch()
