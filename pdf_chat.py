
# Install dependencies:
# pip install langchain langchain-groq langchain-chroma \
#   langchain-huggingface langchain-community langchain-text-splitters \
#   pypdf sentence-transformers gradio python-dotenv

import os

import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# Check that the API key exists
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )

# Initialize Groq chat model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant. Answer the question using ONLY
the context below.

If the context doesn't contain the answer, say:
"I couldn't find that in the document."

After your answer, list the page numbers you used as:
Sources: page X, page Y.

Context:
{context}

Question: {question}
""")

# Build a searchable vector index from the PDF
def build_index(pdf_path):
    pages = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(pages)

    embedder = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
    )

    return db


# Retrieve relevant chunks and ask Groq
def ask(db, question):
    chunks = db.similarity_search(question, k=4)

    if not chunks:
        return "I couldn't find that in the document."

    context = "\n\n".join(
        f"[page {chunk.metadata['page'] + 1}] "
        f"{chunk.page_content}"
        for chunk in chunks
    )

    chain = prompt | model

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response.content


# Store the currently indexed PDF
state = {"db": None}


def upload(pdf):
    if pdf is None:
        return "Please upload a PDF file."

    try:
        state["db"] = build_index(pdf.name)
        return "✅ PDF indexed! Ask me anything about it."
    except Exception as e:
        state["db"] = None
        return f"❌ Failed to index PDF: {e}"


def chat(message, history):
    if state["db"] is None:
        return "Please upload a PDF first 📄"

    if not message or not message.strip():
        return "Please enter a question."

    try:
        return ask(state["db"], message)
    except Exception as e:
        return f"❌ Error generating answer: {e}"


# Gradio interface
with gr.Blocks(title="📄 Chat with your PDF") as demo:
    gr.Markdown("## 📄 Chat with your PDF (powered by Groq + RAG)")

    pdf = gr.File(
        label="Upload a PDF",
        file_types=[".pdf"],
        type="filepath",
    )

    status = gr.Markdown()

    pdf.upload(
        fn=upload,
        inputs=pdf,
        outputs=status,
    )

    gr.ChatInterface(fn=chat)

if __name__ == "__main__":
    demo.launch(share=False)
