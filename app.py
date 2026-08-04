import os
import shutil

from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

# =====================================================
# Configuration
# =====================================================

DOCUMENTS_PATH = "documents"
CHROMA_DB_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def main():

    # =================================================
    # Check Documents Folder
    # =================================================

    if not Path(DOCUMENTS_PATH).exists():
        raise FileNotFoundError(
            f"Folder '{DOCUMENTS_PATH}' does not exist."
        )

    # =================================================
    # Delete Existing Chroma Database
    # =================================================

    if Path(CHROMA_DB_PATH).exists():
        print("Deleting old Chroma database...")
        shutil.rmtree(CHROMA_DB_PATH)

    # =================================================
    # Load All PDFs using PyMuPDFLoader
    # =================================================

    documents = []

    pdf_files = list(Path(DOCUMENTS_PATH).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found inside documents folder."
        )

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyMuPDFLoader(str(pdf_file))

        documents.extend(loader.load())

    print(f"\nLoaded {len(documents)} pages.")

    # =================================================
    # Split into Chunks
    # =================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # =================================================
    # Load Embedding Model
    # =================================================

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # =================================================
    # Create Chroma Vector Database
    # =================================================

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    print("\n✅ Vector Database Created Successfully!")
    print(f"📂 Database Location: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()