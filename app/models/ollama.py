"""
Ollama LLM wrapper — llama3.2:3b for faster CPU inference.

Model choice:
  llama3.2:3b → ~5-8s TTFT on CPU (vs ~20s for mistral 7B)
  
Streaming: True — tokens emitted one-by-one for sentence buffering.
"""
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        logger.info(f"Initializing Ollama: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        from langchain_community.llms import Ollama
        _llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=512,   # cap output length for speed
        )
    return _llm


SYSTEM_PROMPT = """You are a helpful assistant for Antino, a leading software development company.

Answer ONLY using the provided context. Be concise and professional.
If the answer is not in the context, say exactly: "I don't have that information."
Do not invent facts.

Context:
{context}

Question:
{query}

Answer:"""


def build_prompt(context: str, query: str) -> str:
    return SYSTEM_PROMPT.format(context=context[:3000], query=query)


def stream_response(context: str, query: str):
    """
    Stream LLM response token-by-token.
    Yields (token_str, is_first_token: bool)
    """
    llm = get_llm()
    prompt = build_prompt(context, query)

    t0 = time.time()
    first = True
    for chunk in llm.stream(prompt):
        if first:
            ttft = time.time() - t0
            logger.info(f"[LLM] TTFT: {ttft:.2f}s (model={OLLAMA_MODEL})")
        yield chunk, first
        first = False
