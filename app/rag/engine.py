"""
LangGraph 4-node pipeline:
  input_node → retrieval_node → llm_node → tts_node

Conditional: voice → Whisper STT, text → pass-through
"""
import time
import logging
from langgraph.graph import StateGraph, END
from app.utils.helpers import GraphState
from app.models import whisper as whisper_model
from app.ingestion.embedder import query_pinecone
from app.models.ollama import stream_response, build_prompt, get_llm
from app.models.tts import text_to_audio_bytes

logger = logging.getLogger(__name__)

# ── Node 1: Input ──────────────────────────────────────────────────────────

def input_node(state: GraphState) -> GraphState:
    """Run Whisper STT if voice, otherwise pass-through."""
    if state["input_type"] == "voice" and state.get("audio"):
        t0 = time.time()
        text, _ = whisper_model.transcribe(state["audio"])
        logger.info(f"[input_node] STT: '{text}' ({time.time()-t0:.2f}s)")
        return {**state, "query": text}
    # text path — query already set
    return state


# ── Node 2: Retrieval ───────────────────────────────────────────────────────

def retrieval_node(state: GraphState) -> GraphState:
    """Query Pinecone and fill context."""
    query = state.get("query", "").strip()
    if not query:
        return {**state, "context": ""}

    context, elapsed = query_pinecone(query, top_k=4)
    logger.info(f"[retrieval_node] Retrieved context ({elapsed:.2f}s, {len(context)} chars)")
    return {**state, "context": context}


# ── Node 3: LLM ────────────────────────────────────────────────────────────

def llm_node(state: GraphState) -> GraphState:
    """
    Generate full response (non-streaming path used by LangGraph).
    Streaming is handled separately in stream.py for SSE.
    """
    query = state.get("query", "")
    context = state.get("context", "")

    if not context.strip():
        return {**state, "response": "I don't have that information."}

    llm = get_llm()
    prompt = build_prompt(context, query)

    t0 = time.time()
    response = llm.invoke(prompt)
    logger.info(f"[llm_node] Response generated in {time.time()-t0:.2f}s")
    return {**state, "response": response}


# ── Node 4: TTS ────────────────────────────────────────────────────────────

def tts_node(state: GraphState) -> GraphState:
    """Convert final response to audio bytes."""
    response = state.get("response", "")
    if response:
        audio_bytes, elapsed = text_to_audio_bytes(response)
        logger.info(f"[tts_node] TTS done ({elapsed:.2f}s)")
        return {**state, "audio_output": audio_bytes}
    return {**state, "audio_output": None}


# ── Conditional Routing ─────────────────────────────────────────────────────

def route_input(state: GraphState) -> str:
    """Route based on input_type."""
    return "retrieval_node"  # always retrieve after input (STT already handled)


# ── Build Graph ─────────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("input_node", input_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("llm_node", llm_node)
    workflow.add_node("tts_node", tts_node)

    workflow.set_entry_point("input_node")
    workflow.add_edge("input_node", "retrieval_node")
    workflow.add_edge("retrieval_node", "llm_node")
    workflow.add_edge("llm_node", "tts_node")
    workflow.add_edge("tts_node", END)

    return workflow.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
