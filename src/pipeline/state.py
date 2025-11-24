from pydantic import BaseModel
from typing import List

class PipelineState(BaseModel):
    topic: str
    text: str
    k: int = 5
    top_sentences: List[dict] = []
    intents: List[str] = []
    difficulty: List[str] = []
    translations: List[str] = []
