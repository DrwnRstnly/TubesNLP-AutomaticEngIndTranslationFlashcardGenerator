from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT_DIR = os.path.join(ROOT_DIR, "runs_cefr_lora_xlm-roberta-base")
MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "merged")
INPUT_TXT = os.path.join(ROOT_DIR, "test/cefr_test.txt")

print(f"Using merged model dir: {MODEL_DIR}")

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label_raw = getattr(model.config, "id2label", None)
if not id2label_raw:
    raise ValueError("id2label not found in model config.")

id2label = {int(k): v for k, v in id2label_raw.items()}
label_ids = sorted(id2label.keys())
CEFR_ORDER = [id2label[i] for i in label_ids]

print("Label mapping (id2label):", id2label)
print("CEFR_ORDER:", CEFR_ORDER, "\n")

def predict_cefr(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_id = label_ids[pred_idx]
    pred_label = id2label[pred_id]

    prob_per_label = {}
    for i in label_ids:
        idx = label_ids.index(i)
        prob_per_label[id2label[i]] = float(probs[idx])

    return {
        "text": text,
        "pred_label": pred_label,
        "pred_id": int(pred_id),
        "probs": prob_per_label,
    }

def load_sentences(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

if not os.path.exists(INPUT_TXT):
    raise FileNotFoundError(f"Input file not found: {INPUT_TXT}")

sentences = load_sentences(INPUT_TXT)
print(f"Loaded {len(sentences)} sentences from {INPUT_TXT}\n")

results = []
for i, sent in enumerate(sentences, start=1):
    res = predict_cefr(sent)
    results.append(res)

    print(f"[{i}]")
    print(f"Text      : {res['text']}")
    print(f"Pred label: {res['pred_label']}")
    probs_str = " | ".join(
        f"{label}: {res['probs'][label]:.3f}" for label in CEFR_ORDER
    )
    print(f"Probs     : {probs_str}")
    print("\n")

