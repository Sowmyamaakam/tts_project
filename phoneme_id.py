import pandas as pd

# Load your metadata
metadata = pd.read_csv(r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\metadata_final.csv")

# Build phoneme-to-ID mapping
phonemes = set()
for seq in metadata['phonemes']:
    phonemes.update(seq.split())
phoneme2id = {p: i+1 for i, p in enumerate(sorted(phonemes))}  # 0 for padding
phoneme2id['<pad>'] = 0

# Save mapping for later use
import json
with open(r"C:\Users\Admin\Downloads\tts_project\emovdb\phoneme2id.json", "w") as f:
    json.dump(phoneme2id, f)