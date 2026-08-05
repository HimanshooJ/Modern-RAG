#Vector Store Module

import shutil
from pathlib import Path
from langchain_chroma import Chroma
from src.config import CHROMA_DB_PATH

def create_vectorstore(chunks, embeddings):
    """
    Create a new Chroma vector database.
    """

    db_path = Path(CHROMA_DB_PATH)

    if db_path.exists():
        print("Deleting old Chroma database...")
        shutil.rmtree(db_path)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    print("\n Vector Database Created Successfully!")
    print(f" Database Location : {CHROMA_DB_PATH}")


def load_vectorstore(embeddings):
    """
    Load an existing Chroma database.
    """

    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )