from src.task1_ranker.inference import (
    load_model,
    split_into_sentences,
    compute_similarity,
    get_top_k,
)

class SSRNode:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def __call__(self, state):
        sentences = split_into_sentences(state.text)

        sims = compute_similarity(
            self.model,
            sentences,
            state.topic
        )

        topk = get_top_k(sentences, sims, state.k)

        state.top_sentences = [
            {
                "sentence": item["sentence"],
                "similarity": item["similarity"]
            }
            for item in topk
        ]
        return state
