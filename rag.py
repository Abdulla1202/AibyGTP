from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import shutil
import certifi
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

from pypdf import PdfReader
import docx2txt

# Ensure uploads directory exists
Path("uploads").mkdir(exist_ok=True)

# ─── 100% LOCAL EMBEDDINGS (Zero API Calls) ───────────────────────

class LocalEmbeddings(Embeddings):
    """
    Strictly local embeddings using ChromaDB's default model.
    This class does NOT use any Google/OpenAI APIs.
    """
    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} documents locally...")
        return self._ef(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ef([text])[0]

def _init_vectorstore():
    """Initialize vectorstore using ONLY LocalEmbeddings."""
    db_path = Path("chroma_db")
    db_path.mkdir(exist_ok=True)

    # Use the LocalEmbeddings class defined above
    embeddings = LocalEmbeddings()

    return Chroma(
        collection_name="agentic_chatbot_docs",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

# Global vectorstore instance
vectorstore = _init_vectorstore()

# ─── File Processing ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text: return ""
    import re
    return re.sub(r'\s+', ' ', text).strip()

def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        if suffix == ".docx":
            return docx2txt.process(file_path)
        if suffix in [".txt", ".md", ".py", ".csv"]:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Read error: {e}")
        return ""
    raise ValueError("Unsupported file type.")

def add_document_to_rag(file_path: str, thread_id: str):
    try:
        logger.info(f"Processing locally: {file_path} (Thread: {thread_id})")
        p = Path(file_path)
        text = clean_text(read_file_text(str(p)))
        if not text: return {"success": False, "error": "No text found"}

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        docs = [
            Document(page_content=c, metadata={"thread_id": thread_id, "source": p.name})
            for c in chunks
        ]

        vectorstore.add_documents(docs)
        logger.info(f"Successfully added {len(docs)} chunks locally.")
        return {"success": True, "filename": p.name, "chunks": len(docs)}
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        return {"success": False, "error": str(e)}

def retrieve_from_rag(query: str, thread_id: str, k: int = 5) -> str:
    try:
        docs = vectorstore.similarity_search(query, k=k, filter={"thread_id": thread_id})
        if not docs:
            docs = vectorstore.similarity_search(query, k=k)

        if not docs: return "No documents found."

        return "\n\n".join([f"[Source {i+1}: {d.metadata.get('source')}]\n{clean_text(d.page_content)}" for i, d in enumerate(docs)])
    except Exception as e:
        logger.error(f"Retrieval Error: {e}")
        return "Error searching documents."
