import torch
from transformers import MarianTokenizer, MarianMTModel

def load_translation_model(model_path: str):
    print(f"Loading Translation model from: {model_path}")
    
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model, tokenizer

def translate_sentences(model, tokenizer, sentences: list, batch_size=16) -> list:
    device = model.device
    translated_sentences = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=128)
        
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        translated_sentences.extend([d.strip() for d in decoded])
        
    return translated_sentences