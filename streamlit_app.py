# import streamlit as st

# # -----------------------------
# # Page Configuration
# # -----------------------------
# st.set_page_config(
#     page_title="PDF RAG Assistant",
#     page_icon="📄",
#     layout="wide"
# )

# # -----------------------------
# # Sidebar
# # -----------------------------
# with st.sidebar:
#     st.title("📄 PDF RAG Assistant")
#     st.markdown("---")
#     st.write("Upload a PDF and ask questions about it.")
#     st.button("🗑️ Clear Chat")

# # -----------------------------
# # Main Page
# # -----------------------------
# st.title("🤖 PDF RAG Assistant")

# st.write("Ask questions about your PDF using Google Gemini and FAISS.")

# uploaded_file = st.file_uploader(
#     "Upload your PDF",
#     type=["pdf"]
# )

# question = st.text_input(
#     "Ask a question"
# )

# if st.button("Ask"):
#     if not uploaded_file:
#         st.warning("Please upload a PDF first.")
#     elif not question:
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Generating answer..."):
#             st.success("UI is working! Backend will be connected next.")

import os
import streamlit as st

from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DEFAULT_PDF = "data/UDISE_2025_26_Existing_Structure.pdf"

VECTOR_STORE_PATH = "vector_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_MODEL = "gemini-flash-latest"

@st.cache_resource
def initialize_gemini():

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found."
        )

    return genai.Client(api_key=api_key)

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

@st.cache_resource
def build_vector_store(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = load_embedding_model()

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_db.save_local(VECTOR_STORE_PATH)

    return vector_db
@st.cache_resource
def load_vector_store():

    embeddings = load_embedding_model()

    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def ask_question(client, vector_db, question):

    docs = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are an AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context, reply:

"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text

# ==========================================================
# Streamlit UI
# ==========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    text-align: center;
    color: #2563EB;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    color: white;

}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: white !important;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

section[data-testid="stSidebar"] .stButton > button {
    background-color: #2563EB;
    color: white !important;
    border: none;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #1D4ED8;
    color: white !important;
}

.stTextInput input {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
st.title("📄 PDF AI Assistant")

st.caption("Chat with your PDF using Retrieval-Augmented Generation (RAG).")




# Sidebar
with st.sidebar:

    st.header("⚙️ Settings")

    pdf_path = st.text_input(
        "PDF Path",
        DEFAULT_PDF
    )

    if st.button("📂 Load PDF"):
        st.session_state["load_pdf"] = True

    st.divider()

    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### ⚡ Powered by")

    st.markdown("""
- 🤖 Google Gemini
- 🦜 LangChain
- 📚 FAISS
""")
# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []


client = initialize_gemini()

embeddings = load_embedding_model()

# Load Vector Store
if st.session_state.get("load_pdf"):

    faiss_file = os.path.join(
        VECTOR_STORE_PATH,
        "index.faiss"
    )

    with st.spinner("Loading PDF..."):

        if os.path.exists(faiss_file):

            vector_db = load_vector_store()

        else:

            vector_db = build_vector_store(pdf_path)

    st.session_state["vector_db"] = vector_db

    st.success("PDF Loaded Successfully!")

# Question
st.subheader("💬 Chat")

question = st.text_input(
    "Ask a Question",
    placeholder="Type your question here..."
)

if st.button("🤖 Ask AI"):

    if "vector_db" not in st.session_state:

        st.warning("Please load a PDF first.")

    elif not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            answer = ask_question(
                client,
                st.session_state["vector_db"],
                question
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])