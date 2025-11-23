import argparse
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer, util

from peft import PeftModel

BASE_MODELS = {
    "minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm_l12": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "e5_base": "intfloat/e5-base",
    "e5_large": "intfloat/e5-large"
}

def load_sts_test():
    ds = load_dataset("mteb/stsbenchmark-sts")
    return pd.DataFrame(ds["test"])

def load_finetuned_model(model_dir, model_type):
    print(f"Loading {model_type}: {model_dir}")

    model = SentenceTransformer(model_dir)

    if model_type == "lora_finetuned":
        encoder = model._first_module().auto_model
        adapter_config = os.path.join(model_dir, "adapter_config.json")

        if not os.path.exists(adapter_config):
            raise ValueError(f"Expected LoRA adapter in: {model_dir}")

        encoder = PeftModel.from_pretrained(encoder, model_dir)
        encoder.eval()
        model._first_module().auto_model = encoder

    return model

def evaluate_model(model, df):
    emb1 = model.encode(df["sentence1"].tolist(), convert_to_tensor=True)
    emb2 = model.encode(df["sentence2"].tolist(), convert_to_tensor=True)

    sim = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
    human = df["score"].values / 5.0

    pearson = pearsonr(sim, human)[0]
    spearman = spearmanr(sim, human)[0]

    return pearson, spearman

def evaluate_base_models(df):
    results = []
    for key, model_path in BASE_MODELS.items():
        print(f"\nEvaluating BASE model: {key}")
        model = SentenceTransformer(model_path)
        p, s = evaluate_model(model, df)

        results.append({
            "model": key,
            "pearson": p,
            "spearman": s,
            "type": "base"
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True, help="Directory containing finetuned full and LoRA models")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    df_test = load_sts_test()
    results = []

    print("Evaluating FINETUNED models...")
    for model_name in os.listdir(args.models_dir):
        model_path = os.path.join(args.models_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        if model_name.lower().endswith("_lora"):
            model_type = "lora_finetuned"
        elif model_name.lower().endswith("_full"):
            model_type = "full_finetuned"
        else:
            print(f"Skipping unknown folder: {model_name}")
            continue

        model = load_finetuned_model(model_path, model_type)
        p, s = evaluate_model(model, df_test)

        results.append({
            "model": model_name,
            "pearson": p,
            "spearman": s,
            "type": model_type
        })

        print(f"{model_name} ({model_type}): Pearson={p:.4f}, Spearman={s:.4f}")

    print("\nEvaluating BASE models...")
    results.extend(evaluate_base_models(df_test))

    df_out = pd.DataFrame(results)
    df_out.sort_values(by=["pearson", "spearman"], ascending=False, inplace=True)
    df_out.to_csv(args.out, index=False)

    print(f"\nSaved results to {args.out}")

if __name__ == "__main__":
    main()
