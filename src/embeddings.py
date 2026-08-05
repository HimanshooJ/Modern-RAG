#Embedding Model Module

from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL

def get_embeddings():
    """
    Load the HuggingFace embedding model.

    Returns
    -------
    HuggingFaceEmbeddings
        Initialized embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )