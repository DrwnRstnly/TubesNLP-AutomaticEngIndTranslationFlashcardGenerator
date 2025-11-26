from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.pipeline.langgraph_pipeline import flashcard_graph, PipelineState

app = FastAPI(
    title="Automatic Flashcard Generator API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# @app.post("/mock/generate", response_model=GenerateResponse)
# def mock_generate_flashcard(req: GenerateRequest):
#     mock_sentences = [
#         {"sentence": "I really appreciate your help with this project.", "similarity": 0.95},
#         {"sentence": "Could you please submit the report by tomorrow?", "similarity": 0.88},
#         {"sentence": "I promise to call you as soon as I arrive.", "similarity": 0.82}
#     ]

#     mock_intents = ["Inform", "Directive", "Commissive"]

#     mock_difficulty = ["B1", "B2", "A2"]

#     mock_translations = [
#         "Saya sangat menghargai bantuan Anda dalam proyek ini.",
#         "Bisakah Anda mengirimkan laporannya besok?",
#         "Saya berjanji akan menelepon Anda segera setelah saya tiba."
#     ]

#     return GenerateResponse(
#         top_sentences=mock_sentences,
#         intents=mock_intents,
#         difficulty=mock_difficulty,
#         translations=mock_translations
#     )

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
