#Retriever Module

from src.config import TOP_K

def get_retriever(vectorstore):
    """
    Create a retriever from the vector store.

    Parameters
    ----------
    vectorstore : Chroma
        Initialized Chroma vector database.

    Returns
    -------
    BaseRetriever
        LangChain retriever.
    """

    return vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )