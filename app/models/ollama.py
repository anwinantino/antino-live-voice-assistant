"""
Ollama / Mistral LLM wrapper using langchain-community ChatOllama.
Supports streaming token generation.
"""
import os
import time
import logging
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "mistral"

_llm = None


def get_llm():
    """Lazy-load ChatOllama singleton."""
    global _llm
    if _llm is None:
        logger.info(f"Initializing Ollama LLM: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        from langchain_community.llms import Ollama
        _llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
    return _llm


SYSTEM_PROMPT = """You are a helpful assistant for Antino, a top software development company.

Answer ONLY using the provided context below.
If the answer is not in the context, say exactly: "I don't have that information."
Do not make up any facts. Keep answers concise and professional.

Context:
{context}

Question:
{query}

Answer:"""


def build_prompt(context: str, query: str) -> str:
    return SYSTEM_PROMPT.format(context=context, query=query)


def stream_response(context: str, query: str):
    """
    Stream the LLM response token-by-token.
    Yields (token_str, is_first_token_flag)
    """
    llm = get_llm()
    prompt = build_prompt(context, query)

    t0 = time.time()
    first = True
    for chunk in llm.stream(prompt):
        if first:
            ttft = time.time() - t0
            logger.info(f"[LLM] Time to first token: {ttft:.2f}s")
            first = False
        yield chunk, first
