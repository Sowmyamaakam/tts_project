from code1 import FastSpeech2Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import numpy as np
import json
import pandas as pd

class PatchedFastSpeech2Dataset(FastSpeech2Dataset):
    def __init__(self, metadata_path, phoneme2id_path, mel_mean=None, mel_std=None):
        self.data = pd.read_csv(metadata_path)
        with open(phoneme2id_path, 'r') as f:
            self.phoneme2id = json.load(f)

        self.mel_mean = mel_mean
        self.mel_std = mel_std

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        phoneme_ids = [self.phoneme2id[p] for p in row['phonemes'].split()]
        phoneme_ids = torch.LongTensor(phoneme_ids)
        speaker_id = torch.LongTensor([row['speaker_id']])
        emotion_id = torch.LongTensor([row['emotion_id']])

        mel_path = row['mel_path']
        if not os.path.isabs(mel_path):
            mel_path = os.path.join('emovdb', mel_path)
        mel = np.load(mel_path)
        if self.mel_mean is not None and self.mel_std is not None:
            mel = (mel - self.mel_mean[:, None]) / self.mel_std[:, None]
        mel = torch.FloatTensor(mel)

        bert_embedding = np.fromstring(row['bert_embedding'], sep=',', dtype=np.float32)
        bert_embedding = torch.FloatTensor(bert_embedding)

        return {
            'phoneme_ids': phoneme_ids,
            'speaker_id': speaker_id,
            'emotion_id': emotion_id,
            'mel': mel,
            'bert_embedding': bert_embedding
        }


def collate_fn(batch):
    phoneme_ids = [item['phoneme_ids'] for item in batch]
    phoneme_ids_padded = pad_sequence(phoneme_ids, batch_first=True, padding_value=0)
    mels = [item['mel'].transpose(0, 1) for item in batch]  # (n_mels, T) -> (T, n_mels)
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=0).transpose(1, 2)  # (B, n_mels, T)
    speaker_ids = torch.cat([item['speaker_id'] for item in batch])
    emotion_ids = torch.cat([item['emotion_id'] for item in batch])
    bert_embeddings = torch.stack([item['bert_embedding'] for item in batch])
    return {
        'phoneme_ids': phoneme_ids_padded,
        'speaker_id': speaker_ids,
        'emotion_id': emotion_ids,
        'mel': mels_padded,
        'bert_embedding': bert_embeddings
    }

# Test
metadata_path = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\metadata_with_bert.csv"
phoneme2id_path = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\phoneme2id.json"

# Load mean/std if available (example from saved .npy)
mel_mean = np.load("emovdb/mel_mean.npy") if os.path.exists("emovdb/mel_mean.npy") else None
mel_std = np.load("emovdb/mel_std.npy") if os.path.exists("emovdb/mel_std.npy") else None

dataset = PatchedFastSpeech2Dataset(metadata_path, phoneme2id_path, mel_mean, mel_std)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

for batch in dataloader:
    print("Phoneme IDs shape:", batch['phoneme_ids'].shape)
    print("Speaker IDs shape:", batch['speaker_id'].shape)
    print("Emotion IDs shape:", batch['emotion_id'].shape)
    print("Mel shape:", batch['mel'].shape)
    print("BERT Embedding shape:", batch['bert_embedding'].shape)
    break
