import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader_test import PatchedFastSpeech2Dataset, collate_fn
from fastspeech2 import FastSpeech2  # Your model file
import os

# ================== CONFIG ==================
metadata_path = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\metadata_with_bert.csv"
phoneme2id_path = r"C:\Users\Makam Sowmya\Downloads\tts_project_1\emovdb\phoneme2id.json"

batch_size = 8
num_epochs = 30
learning_rate = 1e-4
mel_dim = 80
phoneme_embedding_dim = 256
speaker_embedding_dim = 128
bert_embedding_dim = 768

lambda_mel_l1 = 1.0
lambda_mel_mse = 1.0
lambda_dur = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== DATA ==================
dataset = PatchedFastSpeech2Dataset(metadata_path, phoneme2id_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

vocab_size = len(dataset.phoneme2id)
num_speakers = len(dataset.data['speaker_id'].unique())

# ================== MODEL ==================
model = FastSpeech2(
    vocab_size=vocab_size,
    phoneme_embedding_dim=phoneme_embedding_dim,
    speaker_count=num_speakers,
    speaker_embedding_dim=speaker_embedding_dim,
    bert_embedding_dim=bert_embedding_dim,
    mel_dim=mel_dim
).to(device)

# ================== TRAINING SETUP ==================
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# LR scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# ================== TRAINING LOOP ==================
model.train()
for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    print(f"Starting epoch {epoch}/{num_epochs}...")

    for step, batch in enumerate(dataloader, start=1):
        phoneme_ids = batch['phoneme_ids'].to(device)
        speaker_ids = batch['speaker_id'].to(device)
        mels = batch['mel'].to(device)  # (B, 80, T)
        bert_embeddings = batch['bert_embedding'].to(device)
        durations = batch['durations'].to(device)  # GT durations from aligner or heuristic
        mel_lengths = batch['mel_lengths'].to(device)  # frame lengths
        if step == 1:
            print(f"First batch shapes - phonemes {phoneme_ids.shape}, mel {mels.shape}, durations {durations.shape}")

        # Forward pass with GT durations + target mel_lengths
        mel_outputs, _, predicted_log_durations = model(
            phoneme_ids, speaker_ids, bert_embeddings,
            durations=durations, mel_lengths=mel_lengths
        )

        mel_outputs = mel_outputs.transpose(1, 2)  # (B, 80, T)

        # Masked mel loss (MSE + L1)
        max_len = mel_outputs.size(2)
        mask = (torch.arange(max_len, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1))
        mask = mask.unsqueeze(1).expand(-1, mel_dim, -1)  # (B, 80, T)

        l1 = (mel_outputs - mels).abs()
        mse = (mel_outputs - mels) ** 2
        mel_l1 = l1.masked_select(mask).mean()
        mel_mse = mse.masked_select(mask).mean()

        # Duration prediction loss: supervise on log(dur+1) to stabilize
        # Build padding mask for phoneme positions (0 are pad IDs)
        phoneme_pad_mask = (phoneme_ids != 0).float()  # (B, T_phoneme)
        target_log_dur = torch.log(durations.float() + 1.0)
        dur_diff = (predicted_log_durations - target_log_dur) ** 2
        dur_loss = (dur_diff * phoneme_pad_mask).sum() / (phoneme_pad_mask.sum() + 1e-6)

        loss = lambda_mel_l1 * mel_l1 + lambda_mel_mse * mel_mse + lambda_dur * dur_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch} Step {step}: mel_l1={mel_l1.item():.4f} mel_mse={mel_mse.item():.4f} dur={dur_loss.item():.4f} total={loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"[Epoch {epoch}/{num_epochs}] Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

    if epoch % 5 == 0:
        ckpt_path = f"fastspeech2_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
