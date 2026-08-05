from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from src.config import DOCUMENTS_PATH


def load_documents():
    """
    Loads all PDF files from the documents folder.

    Returns
    -------
    list
        List of LangChain Document objects.
    """

    documents = []

    documents_path = Path(DOCUMENTS_PATH)

    if not documents_path.exists():
        raise FileNotFoundError(
            f"Folder '{DOCUMENTS_PATH}' does not exist."
        )

    pdf_files = list(documents_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found inside the documents folder."
        )

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyMuPDFLoader(str(pdf_file))

        documents.extend(loader.load())

    print(f"\nLoaded {len(documents)} pages.\n")

    return documents