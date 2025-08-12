import pandas as pd
from g2p_en import G2p
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('cmudict')

# Path to your metadata_with_mel.csv
metadata_path = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\metadata_with_mel.csv"
output_path = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\metadata_final.csv"

# Load metadata
metadata = pd.read_csv(metadata_path)
g2p = G2p()

phoneme_seqs = []
for text in metadata['transcript']:
    phonemes = g2p(text)
    # Remove spaces and join phonemes with space
    phoneme_seq = ' '.join([p for p in phonemes if p != ' '])
    phoneme_seqs.append(phoneme_seq)

metadata['phonemes'] = phoneme_seqs
metadata.to_csv(output_path, index=False)
print("Phonemization complete! Saved to", output_path)
