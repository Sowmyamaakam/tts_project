import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

# Paths
metadata_path = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\metadata_final.csv"
output_path = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\metadata_with_bert.csv"

# Load metadata
metadata = pd.read_csv(metadata_path)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to get [CLS] embedding
def get_bert_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # (hidden_size,)
    return cls_embedding

# Extract embeddings
bert_embeddings = []
for text in tqdm(metadata['transcript'], desc="Extracting BERT embeddings"):
    emb = get_bert_cls_embedding(str(text))
    # Store as a string for CSV (comma-separated)
    emb_str = ','.join([f"{x:.6f}" for x in emb])
    bert_embeddings.append(emb_str)

# Add to metadata and save
metadata['bert_embedding'] = bert_embeddings
metadata.to_csv(output_path, index=False)
print(f"Saved metadata with BERT embeddings to {output_path}")