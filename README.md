# Modern-RAG

A simple Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **ChromaDB**, **Hugging Face Embeddings**, and **Google Gemini**.

I built this project to understand how a RAG pipeline actually works instead of relying on an end-to-end framework. The idea was to build every major component separately from loading PDFs to retrieving relevant context and generating answers with an LLM.

---

## Features

- 📄 Load information from PDF documents
- ✂️ Split documents into semantic chunks
- 🧠 Generate embeddings using Sentence Transformers
- 🗄️ Store embeddings in Chroma Vector Database
- 🔍 Retrieve relevant chunks using semantic search
- 🤖 Generate context-aware answers using Gemini
- 🧩 Modular project structure for easier understanding and future improvements
- 🔁 Basic retry logic for temporary API failures

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Google Gemini 3.5 Flash
- PyMuPDF
- python-dotenv

---

## Project Structure

```
Modern-RAG/
│
├── app.py              # Creates the vector database
├── chat.py             # CLI chatbot
│
├── documents/          # PDF files
├── chroma_db/          # Generated vector database
│
├── src/
│   ├── config.py
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── llm.py
│   ├── prompts.py
│   └── utils.py
│
├── tests/
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/HimanshooJ/Modern-RAG.git

cd Modern-RAG
```

Create a virtual environment

```bash
python -m venv rag_env
```

Activate it

**Windows**

```bash
rag_env\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
GOOGLE_API_KEY=YOUR_API_KEY
```

---

## Usage

### Step 1 — Build the Vector Database

```bash
python app.py
```

This loads the PDFs, creates embeddings, and stores them in ChromaDB.

### Step 2 — Start the Chatbot

```bash
python chat.py
```

Example:

```
Question: What is Himanshoo's CGPA?

Answer:
Himanshoo's CGPA is 7.08.
```

If the information isn't available in the document:

```
Question: What is Himanshoo's favorite movie?

Answer:
I couldn't find that information in the provided documents.
```

---

## Why I built this

There are a lot of tutorials that can get a RAG chatbot running in just a few lines of code. While that's great for getting started, I wanted to understand what each component actually does.

So instead of using a complete pipeline, I built each part separately—document loading, chunking, embeddings, vector storage, retrieval, prompt generation, and LLM integration. This also makes the project easier to modify and extend later.

---

## What's Next?

Some improvements I'd like to add:

- Streamlit Web Interface
- Support for multiple PDFs
- Conversation memory
- Source citations
- Better retrieval techniques
- Docker support
- Deploy online

---

## Note

This project is still evolving as I keep learning more about RAG systems. If you have suggestions or spot something that can be improved, feel free to let me know.
