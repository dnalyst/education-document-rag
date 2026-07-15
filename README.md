# PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that enables users to ask natural language questions about a PDF document. The application retrieves the most relevant information from the document using semantic search and generates accurate answers using Google's Gemini model.

---

## Features

- Load any PDF document
- Split documents into semantic chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in a FAISS vector database
- Automatically reuse an existing vector database
- Retrieve the most relevant chunks using semantic similarity search
- Generate context-aware answers using Google Gemini
- Interactive command-line interface

---

## Tech Stack

- Python
- Google Gemini API
- LangChain
- Hugging Face Sentence Transformers
- FAISS Vector Database
- PyPDF
- python-dotenv

---

## Project Structure

```text
pdf-RAG/
│
├── app.py
├── app_learning.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   └── sample.pdf
│
└── vector_store/
    ├── index.faiss
    └── index.pkl
```

---

## RAG Pipeline

```text
                User Question
                      │
                      ▼
          Semantic Similarity Search
                (FAISS Vector DB)
                      │
                      ▼
          Top-K Relevant PDF Chunks
                      │
                      ▼
             Prompt Construction
          (Context + User Question)
                      │
                      ▼
              Google Gemini LLM
                      │
                      ▼
              Context-Aware Answer
```

---

## Workflow

1. Load the PDF document.
2. Split the document into semantic chunks.
3. Generate embeddings using Sentence Transformers.
4. Store embeddings in a FAISS vector database.
5. Retrieve the most relevant chunks for the user's query.
6. Generate an answer using Google Gemini based only on the retrieved context.

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
GOOGLE_API_KEY=YOUR_API_KEY
```

Run the application

```bash
python app.py
```

---

## Future Improvements

- Support multiple PDF documents
- Streamlit web interface
- Conversation history
- Source citation with page numbers
- Hybrid search (keyword + semantic search)
- Metadata filtering

---

## Learning Outcome

This project demonstrates an end-to-end implementation of a Retrieval-Augmented Generation (RAG) pipeline, covering document preprocessing, text chunking, embedding generation, vector search with FAISS, prompt engineering, and response generation using Google Gemini.