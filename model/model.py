import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model_and_tokenizer():
    model_dir = "backend/model/pretrained_model"
    
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict_match(job_desc, resume_text, model, tokenizer, device):
    # Safety check: if text is empty or too short
    if not resume_text.strip() or len(resume_text.strip()) < 20:
        print("[⚠️] Resume text is too short or empty.")
        return False, 0.0

    # Tokenize the input
    inputs = tokenizer(
        job_desc,
        resume_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Move inputs to correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        match_prob = probs[0][1].item()

    is_match = match_prob >= 0.5
    return is_match, round(match_prob, 3)

