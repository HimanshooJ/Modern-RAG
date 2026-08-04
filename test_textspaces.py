from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("documents/Himanshoo_RESUME.pdf")

docs = loader.load()

print(docs[0].page_content)