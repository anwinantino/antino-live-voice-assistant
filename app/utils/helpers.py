"""
GraphState definition and IngestionTracker singleton.
"""
from typing import TypedDict, Optional, Dict
import uuid


class GraphState(TypedDict):
    input_type: str          # "voice" or "text"
    audio: Optional[bytes]   # raw audio bytes (voice only)
    query: str               # transcribed or user-typed query
    context: str             # retrieved Pinecone context
    response: str            # final LLM response text
    audio_output: Optional[bytes]  # TTS output bytes


class IngestionTask:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.progress: int = 0
        self.status: str = "pending"    # pending | running | done | error
        self.message: str = "Queued..."
        self.total_chunks: int = 0
        self.done_chunks: int = 0


class IngestionTracker:
    """Singleton tracker for background ingestion tasks."""
    _instance = None
    _tasks: Dict[str, IngestionTask] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = IngestionTask(task_id)
        return task_id

    def get_task(self, task_id: str) -> Optional[IngestionTask]:
        return self._tasks.get(task_id)

    def update(self, task_id: str, progress: int, status: str, message: str):
        if task_id in self._tasks:
            t = self._tasks[task_id]
            t.progress = min(progress, 100)
            t.status = status
            t.message = message

    def to_dict(self, task_id: str) -> dict:
        t = self._tasks.get(task_id)
        if not t:
            return {"error": "Task not found"}
        return {
            "task_id": t.task_id,
            "progress": t.progress,
            "status": t.status,
            "message": t.message,
        }


tracker = IngestionTracker()
