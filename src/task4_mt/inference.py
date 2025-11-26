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


# import torch
# import re
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# def prompt_role_based(text, domain=None):
#     domain_info = f"({domain})" if domain else ""
#     return f"""
# Translate the English sentence to Indonesian {domain_info}.
# This is a direct mapping task. No reasoning. No explanation.
# Do not modify meaning. Do not invert comparative direction.
# cheaper → lebih murah
# more expensive → lebih mahal

# Return ONLY the Indonesian translation.

# English: {text}
# Indonesian:
# """.strip()


# def extract_translation(full_output, prompt):
#     gen = full_output.replace(prompt, "").strip()
#     if "Indonesian:" in gen:
#         gen = gen.split("Indonesian:")[1].strip()

#     # take until first newline
#     gen = gen.split("\n")[0].strip()
#     # and first period if needed
#     gen = gen.split(".")[0].strip()
#     return gen


# def clean_input_text(text):
#     text = re.sub(r"^(ID|EN|To|From)\s*:\s*", "", text, flags=re.I)
#     return text.strip()


# def load_translation_model(base_model: str, lora_path: str):
#     print(f"Loading Qwen model: {base_model}")
#     print(f"Loading LoRA from: {lora_path}")

#     tokenizer = AutoTokenizer.from_pretrained(base_model)

#     model = AutoModelForCausalLM.from_pretrained(
#         base_model,
#         torch_dtype=torch.float16,
#         device_map="cuda"
#     )

#     model = PeftModel.from_pretrained(
#         model,
#         lora_path,
#         device_map="cuda"
#     )

#     model.eval()
#     return model, tokenizer


# def translate_sentences(model, tokenizer, sentences, domain=None, batch_size=8):
#     device = "cuda"
#     outputs = []

#     for i in range(0, len(sentences), batch_size):
#         batch = sentences[i:i+batch_size]

#         prompts = [
#             prompt_role_based(clean_input_text(s), domain)
#             for s in batch
#         ]

#         inputs = tokenizer(
#             prompts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         ).to(device)

#         with torch.no_grad():
#             generated = model.generate(
#                 **inputs,
#                 max_new_tokens=160,
#                 temperature=0.0,
#                 do_sample=False,
#                 top_p=1.0
#             )

#         decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

#         batch_outputs = [
#             extract_translation(decoded[j], prompts[j])
#             for j in range(len(batch))
#         ]

#         outputs.extend(batch_outputs)

#     return outputs