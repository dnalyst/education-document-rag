# from dotenv import load_dotenv
# import os
# from google import genai

# load_dotenv()

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# print("Testing...")

# response = client.models.generate_content(
#     model="gemini-flash-latest",
#     contents="Reply with exactly one word: Hello"
# )

# print(response.text)



import os
from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

print("✅ Gemini Connected")

# -----------------------------
# Load PDF
# -----------------------------
pdf_path = "data/Microsoft Certified Azure F_ (Z-Library)-1.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

# -----------------------------
# Split into Chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)


print(f"✅ PDF Loaded Successfully!")

print(f"Number of pages = {len(documents)}")

print(f"Number of Chunks = {len(chunks)}")

print("Creating Embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Embedding Model Loaded!")

print("Creating Vector Database...")

vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("✅ Vector Database Created!")

# ==========================
# Ask a Question
# ==========================

# question = "What is Azure Virtual Machine?"

# docs = vector_db.similarity_search(
#     question,
#     k=3
# )

# print(f"\nRetrieved {len(docs)} Chunks\n")
# for i, doc in enumerate(docs):
#     print(f"\n========== Chunk {i+1} ==========")
#     print(doc.page_content[:500])

# context = "\n\n".join([doc.page_content for doc in docs])

# print("\nContext Created Successfully!\n")

# prompt = f"""
# You are an AI assistant.

# Answer ONLY from the context below.

# If the answer is not present, say:
# 'I could not find the answer in the provided document.'

# Context:
# {context}

# Question:
# {question}
# """

# response = client.models.generate_content(
#     model="gemini-flash-latest",
#     contents=prompt
# )

# print("\n========== FINAL ANSWER ==========\n")
# print(response.text)

# ==========================
# Chat Loop
# ==========================

while True:

    question = input("\nAsk your question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("\nThank you for using PDF RAG Assistant!")
        break

    docs = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant.

Answer ONLY from the context below.

If the answer is not present, say:
'I could not find the answer in the provided document.'

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",   # Use the model that worked for you
        contents=prompt
    )

    print("\n========== ANSWER ==========\n")
    print(response.text)