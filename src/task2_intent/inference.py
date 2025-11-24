import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

NUM_LABELS = 4 

def load_intent_model(model_path: str):
    """Loads a fine-tuned sequence classification model and creates a pipeline."""
    print(f"Loading Intent Classification model from: {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1, 
        return_all_scores=False
    )
    return classifier

def classify_sentences(classifier, sentences: list) -> list:
    """Classifies a list of sentences using the loaded pipeline."""
    
    results = classifier(sentences)
    
    intents = []
    for res in results:
        label = res['label'].lower()
        
        if label in ['commissive', 'directive', 'inform', 'question']:
            intents.append(label.capitalize())
        else:
            intents.append("Unknown") 
    
    return intents