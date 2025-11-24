import argparse
import os

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
ID2LABEL = {i: lab for i, lab in enumerate(CEFR_ORDER)}
LABEL2ID = {lab: i for i, lab in ID2LABEL.items()}

def load_cefr_and_splits(cefr_csv: str):
    df = pd.read_csv(cefr_csv)
    df = df[["text", "cefr_level"]].dropna().reset_index(drop=True)
    df["cefr_level"] = df["cefr_level"].astype(str).str.strip().str.upper()

    # mapping label
    df["label"] = df["cefr_level"].map(LABEL2ID)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # drop duplicate
    before = len(df)
    df = df.drop_duplicates(subset=["text", "cefr_level"])
    after = len(df)
    print(f"Removed {before - after} duplicates from CEFR data")

    # split 80/10/10
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"],
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def load_finetuned_model(run_dir: str):
    """
    run_dir: path ke folder run, misalnya:
      runs_cefr_lora_xlm-roberta-base
    """
    merged_dir = os.path.join(run_dir, "merged")

    if not os.path.isdir(merged_dir):
        raise FileNotFoundError(
            f"Expected merged model folder at: {merged_dir}\n"
            f"Pastikan di notebook kamu sudah memanggil merge_and_unload() dan save_pretrained() ke subfolder 'merged'."
        )

    print(f"Loading MERGED finetuned model from: {merged_dir}")
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    model = AutoModelForSequenceClassification.from_pretrained(merged_dir)

    return model, tokenizer

def evaluate_model(model, tokenizer, df, text_column="text", label_column="label"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = df[label_column].values

    all_preds = []
    texts = df[text_column].tolist()
    n = len(texts)
    batch_size = 32

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]

            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.append(preds)

    y_pred = np.concatenate(all_preds, axis=0)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    return acc, f1_macro, f1_weighted

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cefr_csv",
    default="cefr.csv",
    help="Path ke file CEFR (seperti di notebook, default: cefr.csv)"
)
parser.add_argument(
    "--models_dir",
    required=True,
    help=(
        "Path ke folder run (mis. ./runs_cefr_lora_xlm-roberta-base) "
        "yang DI DALAMNYA terdapat subfolder 'merged/'."
    )
)
parser.add_argument(
    "--out",
    required=False,
    help="Output CSV path untuk menyimpan hasil evaluasi (opsional)"
)
args = parser.parse_args()

# Load data & split test
_, _, test_df = load_cefr_and_splits(args.cefr_csv)

run_dir = os.path.abspath(args.models_dir)
model_name = os.path.basename(run_dir)

model, tokenizer = load_finetuned_model(run_dir)

acc, f1_macro, f1_weighted = evaluate_model(
    model,
    tokenizer,
    test_df,
    text_column="text",
    label_column="label",
)

results = [{
    "model": model_name,
    "accuracy": acc,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,
}]

df_out = pd.DataFrame(results)

if args.out:
    df_out.to_csv(args.out, index=False)
    print(f"\nSaved results to {args.out}")
else:
    print("\nFinal evaluation results (not saved to file):")
print(df_out.to_string(index=False))

