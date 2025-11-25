from typing import List, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def load_cefr_model(model_path: str):
    print(f"Loading CEFR model from: {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=True,
    )

    id2label_raw = getattr(model.config, "id2label", None)
    if not id2label_raw:
        raise ValueError("id2label not found in model config.")

    id2label = {int(k): v for k, v in id2label_raw.items()}
    label_ids = sorted(id2label.keys())
    cefr_order = [id2label[i] for i in label_ids]

    return classifier, cefr_order

def predict_cefr(
    classifier,
    sentences: List[str],
    cefr_order: List[str],
) -> List[Dict]:
    raw_outputs = classifier(sentences)

    results = []
    for sent, scores in zip(sentences, raw_outputs):
        best = max(scores, key=lambda x: x["score"])
        prob_map = {s["label"]: float(s["score"]) for s in scores}

        probs_ordered = {lab: prob_map.get(lab, 0.0) for lab in cefr_order}

        results.append(
            {
                "text": sent,
                "pred_label": best["label"],
                "probs": probs_ordered,
            }
        )

    return results