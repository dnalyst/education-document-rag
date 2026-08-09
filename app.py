import os
from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================================================
# Configuration
# ==========================================================

DEFAULT_PDF = "data/UDISE_2025_26_Existing_Structure.pdf"
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-flash-latest"


# ==========================================================
# Gemini Client
# ==========================================================

def initialize_gemini():

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please check your .env file."
        )

    client = genai.Client(api_key=api_key)

    print("Gemini Connected")

    return client


# ==========================================================
# Embedding Model
# ==========================================================

def load_embedding_model():

    print("Loading Embedding Model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print(" Embedding Model Loaded")

    return embeddings


# ==========================================================
# Build Vector Store
# ==========================================================

def build_vector_store(pdf_path, embeddings):

    print("\nLoading PDF...")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(" PDF Loaded Successfully")
    print(f"Number of Pages : {len(documents)}")

    print("\nSplitting PDF into Chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Number of Chunks : {len(chunks)}")

    print("\nCreating FAISS Vector Store...")

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_db.save_local(VECTOR_STORE_PATH)

    print(" Vector Store Created")
    print(" Vector Store Saved")

    return vector_db


# ==========================================================
# Load Existing Vector Store
# ==========================================================

def load_vector_store(embeddings):

    print("\nLoading Existing Vector Store...")

    vector_db = FAISS.load_local(
        folder_path=VECTOR_STORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    print(" Existing Vector Store Loaded")

    return vector_db


# ==========================================================
# Ask Question
# ==========================================================

def ask_question(client, vector_db, question):

    docs = vector_db.similarity_search(
        question,
        k=3
    )

    context_parts = []

    for doc in docs:
        page_number = doc.metadata.get("page", "Unknown")
        context_parts.append(
        f"[Page {page_number + 1}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an education-data document assistant.

Your task is to answer the user's question using ONLY the
retrieved content from the provided UDISE+ document.

Rules:
1. Do not use outside knowledge.
2. Do not invent statistics, figures, rankings, trends, or claims.
3. If the retrieved context does not contain enough information
   to answer the question, say:
   "I could not find enough information to answer this from
   the provided UDISE+ document."
4. When the document provides a specific figure or statistic,
   reproduce it accurately.
5. Keep the answer clear and concise.
6. If useful, mention the relevant page number from the retrieved
   document.

Retrieved context:
{context}

User question:
{question}
"""
    
    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        print("\n" + "=" * 60)
        print("Assistant")
        print("=" * 60)
        print(response.text)

        print("\n" + "-" * 60)
        print("Retrieved Sources")
        print("-" * 60)

        for doc in docs:
            page_number = doc.metadata.get("page", "Unknown")
            print(f"Page {page_number + 1}")

    except Exception as e:

        print(f"\nError: {e}")


# ==========================================================
# Main Application
# ==========================================================

def main():

    print("=" * 60)
    print("                 PDF RAG ASSISTANT")
    print("=" * 60)

    # Initialize Gemini
    client = initialize_gemini()

    # Load Embedding Model
    embeddings = load_embedding_model()

    # Select PDF
    pdf_path = DEFAULT_PDF

    print(f"\nUsing document: {pdf_path}")

    # Check whether vector database already exists
    faiss_file = os.path.join(
        VECTOR_STORE_PATH,
        "index.faiss"
    )

    if os.path.exists(faiss_file):

        vector_db = load_vector_store(embeddings)

    else:

        vector_db = build_vector_store(
            pdf_path,
            embeddings
        )

    print("\n" + "=" * 60)
    print("Ask questions from your PDF.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":

            print("\nThank you for using PDF RAG Assistant!")

            break

        if not question:

            print("Please enter a question.")

            continue

        ask_question(
            client,
            vector_db,
            question
        )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\n\nProgram terminated by user.")

    except Exception as e:

        print(f"\n{e}")