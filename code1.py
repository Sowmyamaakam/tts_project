import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json

class FastSpeech2Dataset(Dataset):
    def __init__(self, metadata_path, phoneme2id_path):
        self.data = pd.read_csv(metadata_path)
        with open(phoneme2id_path, "r") as f:
            self.phoneme2id = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Convert phoneme string to list of IDs
        phoneme_ids = [self.phoneme2id[p] for p in row['phonemes'].split()]
        phoneme_ids = torch.LongTensor(phoneme_ids)
        # Speaker and emotion IDs
        speaker_id = torch.LongTensor([row['speaker_id']])
        emotion_id = torch.LongTensor([row['emotion_id']])
        # Load mel-spectrogram
        mel = np.load(row['mel_path'])
        mel = torch.FloatTensor(mel)
        # Parse BERT embedding
        bert_embedding = torch.tensor([float(x) for x in row['bert_embedding'].split(',')], dtype=torch.float32)
        return {
            'phoneme_ids': phoneme_ids,
            'speaker_id': speaker_id,
            'emotion_id': emotion_id,
            'mel': mel,
            'bert_embedding': bert_embedding
        }