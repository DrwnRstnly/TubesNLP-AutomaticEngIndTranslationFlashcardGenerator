from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.langgraph_pipeline import flashcard_graph, PipelineState

app = FastAPI(
    title="Automatic Flashcard Generator API",
    version="1.0.0",
)

class GenerateRequest(BaseModel):
    topic: str
    text: str
    k: int = 5

class GenerateResponse(BaseModel):
    top_sentences: list
    intents: list
    difficulty: list
    translations: list

@app.post("/generate", response_model=GenerateResponse)
def generate_flashcard(req: GenerateRequest):

    init_state = PipelineState(
        topic=req.topic,
        text=req.text,
        k=req.k
    )

    result = flashcard_graph.invoke(init_state)

    return GenerateResponse(
        top_sentences=result["top_sentences"],
        intents=result.get("intents", []),
        difficulty=result.get("difficulty", []),
        translations=result.get("translations", []),
    )
