import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector


load_dotenv()


def get_embeddings():
    """Choose embedding model based on available API keys"""
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key and openai_key.strip():
        print("Using OpenAI embeddings")
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    elif google_key and google_key.strip():
        print("Using Google embeddings")
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        )
    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env"
        )


def ingest_pdf():
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "document.pdf"

    docs = PyPDFLoader(str(pdf_path)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, add_start_index=False
    ).split_documents(docs)

    if not splits:
        raise ValueError("No document splits were created.")

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = get_embeddings()

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVERCTOR_COLLECTION"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    store.add_documents(enriched, ids=ids)


if __name__ == "__main__":
    ingest_pdf()
