#Gemini LLM Module

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_MODEL
load_dotenv()


def get_llm():
    """
    Load the Gemini LLM.
    """

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )