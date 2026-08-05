#Build Chroma Vector Database

from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore

def main():

    print("=" * 80)
    print("Building Vector Database")
    print("=" * 80)

    # -------------------------------------------------
    # Load Documents
    # -------------------------------------------------

    documents = load_documents()

    # -------------------------------------------------
    # Split Documents
    # -------------------------------------------------

    chunks = split_documents(documents)

    # -------------------------------------------------
    # Load Embedding Model
    # -------------------------------------------------

    embeddings = get_embeddings()

    # -------------------------------------------------
    # Create Vector Database
    # -------------------------------------------------

    create_vectorstore(
        chunks,
        embeddings
    )

    print("\nDatabase is ready!")


if __name__ == "__main__":
    main()