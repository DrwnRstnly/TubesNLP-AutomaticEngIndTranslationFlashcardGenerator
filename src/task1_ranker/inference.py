import argparse
import os
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import pandas as pd


nltk.download("punkt_tab")

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    return SentenceTransformer(model_path)

def split_into_sentences(text):
    return sent_tokenize(text)

def compute_similarity(model, sentences, topic):
    topic_emb = model.encode(topic, convert_to_tensor=True)

    sent_emb = model.encode(sentences, convert_to_tensor=True)

    sim = util.cos_sim(topic_emb, sent_emb).cpu().numpy()[0]

    return sim

def get_top_k(sentences, similarities, k):
    idx = np.argsort(similarities)[::-1][:k] 
    results = []

    for i in idx:
        results.append({
            "sentence": sentences[i],
            "similarity": float(similarities[i])
        })
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best trained model")
    parser.add_argument("--text", required=True, help="Input text file or raw text")
    parser.add_argument("--topic", required=True, help="Topic or prompt")
    parser.add_argument("--k", type=int, default=5, help="Number of top sentences")

    args = parser.parse_args()

    model = load_model(args.model)

    if os.path.exists(args.text):
        with open(args.text, "r", encoding="utf8") as f:
            text = f.read()
    else:
        text = args.text
    
    sentences = split_into_sentences(text)

    sim = compute_similarity(model, sentences, args.topic)

    topk = get_top_k(sentences, sim, args.k)

    df = pd.DataFrame(topk)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()