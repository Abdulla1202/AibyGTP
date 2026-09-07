from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import shutil
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings


from pypdf import PdfReader
import docx2txt


Path("uploads").mkdir(exist_ok=True)


def _init_vectorstore():
    """Initialize vectorstore with Google Embeddings for higher quality retrieval."""
    db_path = Path("chroma_db")

    # Wipe old DB if it doesn't have the marker for Google Embeddings
    marker = db_path / ".google_embeddings"
    if db_path.exists() and not marker.exists():
        shutil.rmtree(db_path, ignore_errors=True)
        print("[OK] Cleared old chroma_db to switch to Google Embeddings")

    db_path.mkdir(exist_ok=True)
    marker.touch()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(
        collection_name="agentic_chatbot_docs",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )


vectorstore = _init_vectorstore()


# ─── File Reading ─────────────────────────────────────────────

def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix == ".csv":
        import csv
        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return ""
            header = rows[0]
            data_rows = rows[1:]
            text_output = []
            for row in data_rows:
                row_str = " | ".join([f"{header[i]}: {val}" for i, val in enumerate(row) if i < len(header)])
                text_output.append(row_str)
            return "\n".join(text_output)

    if suffix in [".txt", ".md", ".py"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")


# ─── Add Document ─────────────────────────────────────────────

def add_document_to_rag(file_path: str, thread_id: str):
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name
            }
        )
        for chunk in chunks
    ]

    vectorstore.add_documents(docs)

    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }


# ─── Search Documents ────────────────────────────────────────

def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    # First try thread-specific docs
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={"thread_id": thread_id}
    )

    # Fallback: search ALL uploaded docs if thread-specific found nothing
    if not docs:
        try:
            docs = vectorstore.similarity_search(query, k=k)
        except Exception:
            docs = []

    if not docs:
        return "No relevant uploaded document content found. Please upload a document first."

    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(results)