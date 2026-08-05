#Modern RAG Chatbot

import time
from src.config import (
    MAX_RETRIES,
    RETRY_DELAY
)
from src.embeddings import get_embeddings
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.llm import get_llm
from src.prompts import build_prompt

def extract_answer(response):
    """
    Extract plain text answer from Gemini response.
    """

    answer = response.content

    if isinstance(answer, str):
        return answer

    if isinstance(answer, list):

        for item in answer:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    return item["text"]

            elif hasattr(item, "text"):
                return item.text

    return str(answer)


def main():

    print("=" * 80)
    print("Modern RAG Chatbot".center(80))
    print("=" * 80)
    print("Type 'exit', 'quit' or 'bye' to close.\n")

    embeddings = get_embeddings()

    vectorstore = load_vectorstore(
        embeddings
    )

    retriever = get_retriever(
        vectorstore
    )

    llm = get_llm()

    while True:

        question = input("Question : ").strip()

        if question.lower() in {
            "exit",
            "quit",
            "bye"
        }:

            print("\n👋 Thanks for using Modern RAG.")
            break

        docs = retriever.invoke(question)

        if not docs:

            print("\nNo relevant documents found.\n")
            continue

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = build_prompt(
            question,
            context
        )

        response = None

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                print(
                    f"\nGenerating answer... ({attempt}/{MAX_RETRIES})"
                )

                response = llm.invoke(prompt)

                break

            except Exception as e:

                print(
                    f"\nAttempt {attempt} failed."
                )

                if attempt < MAX_RETRIES:

                    print(
                        f"Retrying in {RETRY_DELAY} seconds...\n"
                    )

                    time.sleep(RETRY_DELAY)

                else:

                    print(
                        "\nGemini is currently unavailable."
                    )

                    print(e)

        if response is None:
            continue

        print("\n" + "=" * 80)
        print("Answer")
        print("=" * 80)

        print(
            extract_answer(response)
        )

        print("=" * 80)


if __name__ == "__main__":
    main()