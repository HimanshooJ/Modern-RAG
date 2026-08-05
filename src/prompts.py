#Prompt Builder Module

def build_prompt(question, context):
    """
    Build the prompt for Gemini.
    """

    return f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not available in the context, reply exactly:

"I couldn't find that information in the provided documents."

Do not make up facts.

Context:
{context}

Question:
{question}

Answer:
"""