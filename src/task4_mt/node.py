from src.pipeline.state import PipelineState
from src.task4_mt.inference import load_translation_model, translate_sentences

class TranslationNode:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load_translation_model(model_path)

    def __call__(self, state: PipelineState) -> PipelineState:
        if not state.top_sentences:
            state.translations = []
            return state

        english_sentences = [item["sentence"] for item in state.top_sentences]

        translations = translate_sentences(self.model, self.tokenizer, english_sentences)

        state.translations = translations
        return state