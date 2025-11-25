from src.pipeline.state import PipelineState
from src.task3_cefr.inference import load_cefr_model, predict_cefr

class CEFRNode:
    def __init__(self, model_path: str):
        self.classifier, self.cefr_order = load_cefr_model(model_path)

    def __call__(self, state: PipelineState) -> PipelineState:
        if not state.top_sentences:
            return state

        sentences = [item["sentence"] for item in state.top_sentences]

        results = predict_cefr(
            classifier=self.classifier,
            sentences=sentences,
            cefr_order=self.cefr_order,
        )

        state.difficulty = [r["pred_label"] for r in results]
        return state
