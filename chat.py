import os
import time

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

# =====================================================
# Configuration
# =====================================================

CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.5-flash"

TOP_K = 3
MAX_RETRIES = 3
RETRY_DELAY = 3

# =====================================================
# Load Embedding Model
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# =====================================================
# Load Chroma Vector Database
# =====================================================

vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)

# =====================================================
# Load Gemini
# =====================================================

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# =====================================================
# Chat Loop
# =====================================================

print("=" * 80)
print("                 Modern RAG Chatbot")
print("=" * 80)
print("Type 'exit', 'quit' or 'bye' to close.\n")

while True:

    question = input("Question : ").strip()

    if question.lower() in {"exit", "quit", "bye"}:
        print("\n👋 Thanks for using Modern RAG.")
        break

    # =================================================
    # Retrieve Relevant Documents
    # =================================================

    docs = retriever.invoke(question)

    if not docs:
        print("\nNo relevant documents found.\n")
        continue

    # =================================================
    # Build Context
    # =================================================

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # =================================================
    # Prompt
    # =================================================

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, reply exactly:

"I couldn't find that information in the provided documents."

Do not make up facts.

Context:
{context}

Question:
{question}

Answer:
"""

    # =================================================
    # Generate Response
    # =================================================

    response = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(f"\nGenerating answer... ({attempt}/{MAX_RETRIES})")

            response = llm.invoke(prompt)

            break

        except Exception as e:

            print(f"\nAttempt {attempt} failed.")

            if attempt < MAX_RETRIES:

                print(f"Retrying in {RETRY_DELAY} seconds...\n")

                time.sleep(RETRY_DELAY)

            else:

                print("\nGemini is currently unavailable.")
                print(e)

    if response is None:
        continue

    # =================================================
    # Display Response
    # =================================================

    print("\n" + "=" * 80)
    print("Answer")
    print("=" * 80)

    answer = response.content

    if isinstance(answer, str):

        print(answer)

    elif isinstance(answer, list):

        printed = False

        for item in answer:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    print(item["text"])
                    printed = True
                    break

            elif hasattr(item, "text"):

                print(item.text)
                printed = True
                break

        if not printed:
            print(answer)

    else:

        print(answer)

    print("=" * 80)