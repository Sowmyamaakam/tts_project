import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DurationPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DurationPredictor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.ln1 = nn.LayerNorm(input_dim)

        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.ln2 = nn.LayerNorm(input_dim)

        self.conv_out = nn.Conv1d(input_dim, 1, kernel_size=1)

    def forward(self, x):
        # Input: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.ln1(x)

        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.transpose(1, 2)
        x = self.ln2(x)

        x = x.transpose(1, 2)
        x = self.conv_out(x)  # (B, 1, T)
        return x.squeeze(1)   # (B, T)


class LengthRegulator(nn.Module):
    def forward(self, x, durations, target_len=None):
        # x: (B, T, D)
        # durations: (B, T)
        out = []
        for i in range(x.size(0)):  # batch
            expanded = []
            for j in range(x.size(1)):  # time
                dur = max(int(durations[i, j].item()), 0)
                if dur > 0:
                    expanded.append(x[i, j].unsqueeze(0).repeat(dur, 1))  # (dur, D)

            if expanded:
                expanded = torch.cat(expanded, dim=0)  # (T_expanded, D)
            else:
                expanded = torch.zeros((1, x.size(2)), device=x.device)  # fallback

            # Clip or pad to match target length if provided
            if target_len is not None:
                tgt = int(target_len[i].item())
                if expanded.size(0) > tgt:
                    expanded = expanded[:tgt]
                elif expanded.size(0) < tgt:
                    pad_len = tgt - expanded.size(0)
                    expanded = F.pad(expanded, (0, 0, 0, pad_len))

            out.append(expanded)

        max_len = max(o.size(0) for o in out)
        out_padded = torch.stack([
            F.pad(o, (0, 0, 0, max_len - o.size(0))) for o in out
        ])  # (B, max_len, D)
        return out_padded

    


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        return self.encoder(x)




class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, phoneme_embedding_dim, speaker_count, speaker_embedding_dim, bert_embedding_dim, mel_dim):
        super(FastSpeech2, self).__init__()

        self.phoneme_embedding = nn.Embedding(vocab_size, phoneme_embedding_dim)
        self.speaker_embedding = nn.Embedding(speaker_count, speaker_embedding_dim)
        self.bert_proj = nn.Linear(bert_embedding_dim, bert_embedding_dim)

        self.total_dim = phoneme_embedding_dim + speaker_embedding_dim + bert_embedding_dim

        self.positional_encoding = PositionalEncoding(self.total_dim)
        self.encoder = TransformerEncoder(d_model=self.total_dim, nhead=4, num_layers=4)

        self.duration_predictor = DurationPredictor(self.total_dim)
        self.length_regulator = LengthRegulator()

        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, mel_dim)
        )

    def forward(self, phoneme_ids, speaker_ids, bert_embeddings, durations=None, mel_lengths=None):
        phoneme_embed = self.phoneme_embedding(phoneme_ids)  # (B, T, D1)
        speaker_embed = self.speaker_embedding(speaker_ids).unsqueeze(1).expand(-1, phoneme_embed.size(1), -1)
        bert_embed = self.bert_proj(bert_embeddings).unsqueeze(1).expand(-1, phoneme_embed.size(1), -1)

        x = torch.cat([phoneme_embed, speaker_embed, bert_embed], dim=-1)
        x = self.positional_encoding(x)
        x = self.encoder(x)

        # Always compute predicted log-durations for possible supervision
        predicted_log_durations = self.duration_predictor(x)  # (B, T_phoneme)

        if durations is None:
            # Inference: use predicted durations
            durations = torch.clamp(torch.round(torch.exp(predicted_log_durations) - 1), min=1).long()

        # Pass target mel_lengths during training to clip/pad properly
        x = self.length_regulator(x, durations, target_len=mel_lengths)
        mel_output = self.decoder(x)  # (B, T_mel, mel_dim)

        return mel_output, durations, predicted_log_durations


