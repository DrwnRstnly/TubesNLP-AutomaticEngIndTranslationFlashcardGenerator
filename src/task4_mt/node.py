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
    
    
# translation_node_qwen.py
# from src.pipeline.state import PipelineState
# from src.task4_mt.inference_qwen import (
#     load_qwen_lora_model,
#     translate_batch
# )

# class TranslationNode:

#     def __init__(self, base_model: str, lora_path: str, domain: str = None):
#         self.domain = domain
#         self.model, self.tokenizer = load_qwen_lora_model(base_model, lora_path)

#     def __call__(self, state: PipelineState) -> PipelineState:
#         if not state.top_sentences:
#             state.translations = []
#             return state

#         english_sentences = [item["sentence"] for item in state.top_sentences]

#         translations = translate_batch(
#             self.model,
#             self.tokenizer,
#             english_sentences,
#             domain=self.domain
#         )

#         state.translations = translations
#         return state


# Usage:
# node = TranslationNode(
#     base_model="Qwen/Qwen2.5-3B-Instruct",
#     lora_path="./QWEN3",
#     domain="audit"
# )

# state = PipelineState()
# state.top_sentences = [
#     {"sentence": "The auditors found inconsistencies..."},
#     {"sentence": "Financial statements were reviewed thoroughly..."}
# ]

# state = node(state)
# print(state.translations)