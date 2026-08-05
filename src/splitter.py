#Document Splitter Module

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def split_documents(documents):
    """
    Split documents into smaller chunks.

    Parameters
    ----------
    documents : list
        List of LangChain Document objects.

    Returns
    -------
    list
        List of chunked Document objects.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.\n")

    return chunks