import argparse
import os
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset
from sentence_transformers.models import Pooling, Transformer
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model
from transformers import AutoModel

MODEL_LIST = {
    "minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm_l12": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "e5_base": "intfloat/e5-base",
    "e5_large": "intfloat/e5-large"
}

def load_sts_dataset(sample_size=None):
    ds = load_dataset("mteb/stsbenchmark-sts")
    train_df = pd.DataFrame(ds["train"])

    if sample_size:
        train_df = train_df.sample(sample_size, random_state=42).reset_index(drop=True)
    return train_df

def prepare_examples(df):
    examples = []
    for _, row in df.iterrows():
        examples.append(
            InputExample(
                texts=[row["sentence1"], row["sentence2"]],
                label=float(row["score"]) / 5.0
            )
        )
    return examples


def add_lora(model_name):

    base = Transformer(model_name)
    encoder = base.auto_model

    if "mpnet" in model_name:
        target_modules = ["q", "k", "v", "out_proj"]
    else:
        target_modules = ["query", "key", "value"]
        
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=target_modules
    )

    encoder = get_peft_model(encoder, config)
    encoder.print_trainable_parameters()

    base.auto_model = encoder

    pooling = Pooling(
        base.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(modules=[base, pooling])
    return model


def finetune_model(model_key, output_dir, batch_size=32, epochs=1, use_lora=False):

    model_name = MODEL_LIST[model_key]
    print(f"Fine tuning model: {model_name}")
    print(f"Training mode: {'LoRA' if use_lora else 'FULL'}")

    train_df = load_sts_dataset()
    examples = prepare_examples(train_df)

    if not use_lora:
        model = SentenceTransformer(model_name)
    else:
        model = add_lora(model_name)

    train_dataset = SentencesDataset(examples, model)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    loss = losses.CosineSimilarityLoss(model)

    os.makedirs(output_dir, exist_ok=True)

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        output_path=output_dir
    )

    print(f"Model saved to: {output_dir}")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_path = "finetune_log.csv"
    run_info = {
        "model_key": model_key,
        "model_name": model_name,
        "finetune_type": "lora" if use_lora else "full",
        "epochs": epochs,
        "trainable_params": total_trainable,
        "output_dir": output_dir
    }

    if not os.path.exists(log_path):
        pd.DataFrame([run_info]).to_csv(log_path, index=False)
    else:
        df_old = pd.read_csv(log_path)
        df_new = pd.concat([df_old, pd.DataFrame([run_info])], ignore_index=True)
        df_new.to_csv(log_path, index=False)

    print(f"Logged run info to {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key: minilm_l6, minilm_l12, mpnet, e5_base, e5_large")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora", action="store_true", help="Use LoRA training")

    args = parser.parse_args()

    finetune_model(
        model_key=args.model,
        output_dir=args.out,
        epochs=args.epochs,
        use_lora=args.lora
    )


if __name__ == "__main__":
    main()
