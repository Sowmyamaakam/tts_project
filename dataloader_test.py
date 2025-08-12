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

        self.base_dir = os.path.dirname(metadata_path)
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
            mel_path = os.path.join(self.base_dir, mel_path)
        mel = np.load(mel_path)
        if self.mel_mean is not None and self.mel_std is not None:
            mel = (mel - self.mel_mean[:, None]) / self.mel_std[:, None]
        mel_length = int(mel.shape[1])  # (n_mels, T) -> T
        mel = torch.FloatTensor(mel)

        bert_embedding = np.fromstring(row['bert_embedding'], sep=',', dtype=np.float32)
        bert_embedding = torch.FloatTensor(bert_embedding)

        # Prefer durations from CSV if provided; otherwise compute heuristic
        durations = None
        if 'durations' in self.data.columns and isinstance(row['durations'], str) and len(row['durations']) > 0:
            try:
                parsed = [int(x) for x in str(row['durations']).split(',') if x.strip() != '']
                durations = torch.LongTensor(parsed)
            except Exception:
                durations = None
        if durations is None or durations.numel() != phoneme_ids.numel():
            num_phonemes = int(phoneme_ids.size(0))
            if num_phonemes > 0:
                base = mel_length // num_phonemes
                remainder = mel_length % num_phonemes
                durations_list = [base + (1 if i < remainder else 0) for i in range(num_phonemes)]
            else:
                durations_list = []
            durations = torch.LongTensor(durations_list)  # (T_phonemes,)

        return {
            'phoneme_ids': phoneme_ids,
            'speaker_id': speaker_id,
            'emotion_id': emotion_id,
            'mel': mel,
            'bert_embedding': bert_embedding,
            'durations': durations,
            'mel_length': torch.LongTensor([mel_length])  # keep as tensor for easy collation
        }


def collate_fn(batch):
    phoneme_ids = [item['phoneme_ids'] for item in batch]
    phoneme_ids_padded = pad_sequence(phoneme_ids, batch_first=True, padding_value=0)

    # Durations padded to phoneme length
    durations = [item['durations'] for item in batch]
    durations_padded = pad_sequence(durations, batch_first=True, padding_value=0)

    mels = [item['mel'].transpose(0, 1) for item in batch]  # (n_mels, T) -> (T, n_mels)
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=0).transpose(1, 2)  # (B, n_mels, T)

    speaker_ids = torch.cat([item['speaker_id'] for item in batch])
    emotion_ids = torch.cat([item['emotion_id'] for item in batch])
    bert_embeddings = torch.stack([item['bert_embedding'] for item in batch])

    mel_lengths = torch.cat([item['mel_length'] for item in batch]).long()  # (B,)

    return {
        'phoneme_ids': phoneme_ids_padded,
        'speaker_id': speaker_ids,
        'emotion_id': emotion_ids,
        'mel': mels_padded,
        'bert_embedding': bert_embeddings,
        'durations': durations_padded,
        'mel_lengths': mel_lengths
    }

# Test
if __name__ == '__main__':
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
        print("Durations shape:", batch['durations'].shape)
        print("Mel lengths shape:", batch['mel_lengths'].shape)
        break
