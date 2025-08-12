import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
DATASET_ROOT = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb"
WAV_ROOT = os.path.join(DATASET_ROOT)
MEL_ROOT = os.path.join(DATASET_ROOT, "mels")
os.makedirs(MEL_ROOT, exist_ok=True)

# Mel-spectrogram parameters
SR = 22050
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024

# Load metadata
metadata = pd.read_csv(os.path.join(DATASET_ROOT, "metadata.csv"))

mel_paths = []

for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    wav_path = os.path.join(WAV_ROOT, row['wav_path'])
    mel_filename = os.path.splitext(os.path.basename(wav_path))[0] + ".npy"
    mel_path = os.path.join("mels", mel_filename)
    try:
        y, sr = librosa.load(wav_path, sr=SR)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        np.save(os.path.join(DATASET_ROOT, mel_path), mel_db)
        mel_paths.append(mel_path)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        mel_paths.append("")  # Or None, if you want to mark missing

# Add mel_path to your metadata
metadata['mel_path'] = mel_paths
metadata.to_csv(os.path.join(DATASET_ROOT, "metadata_with_mel.csv"), index=False)

print("Mel-spectrogram extraction complete!")