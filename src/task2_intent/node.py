import torch
from src.task2_intent.inference import load_intent_model, classify_sentences

class IntentNode:
    
    def __init__(self, model_path: str):
        self.classifier = load_intent_model(model_path)

    def __call__(self, state):
        english_sentences = [item["sentence"] for item in state.top_sentences]

        if not english_sentences:
            state.intents = []
            return state

        intents = classify_sentences(self.classifier, english_sentences)

        state.intents = intents
        return state