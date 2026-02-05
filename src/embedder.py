def get_codebert_embedding(code_path, tokenizer, model, device):
    """Estrae l'embedding CodeBERT per un singolo file."""
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except Exception as e:
        return None