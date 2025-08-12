
import os
import csv
import re

# Path to your dataset root (where speaker folders are)
DATASET_ROOT = r"C:\Users\Makam Sowmya\Downloads\emovdb"  # e.g., "emovdb_sorted"
OUTPUT_CSV = os.path.join(DATASET_ROOT, "metadata.csv")

# 1. Parse cmuarctic.data
transcript_path = os.path.join(DATASET_ROOT, "cmuarctic.data")
transcript_dict = {}
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        match = re.match(r'\(\s*(\S+)\s+"(.+)"\s*\)', line)
        if match:
            wav_id, transcript = match.groups()
            transcript_dict[wav_id.lower()] = transcript

# 2. Map speakers and emotions to IDs
speakers = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d != "__MACOSX"])
emotions = set()
for speaker in speakers:
    speaker_path = os.path.join(DATASET_ROOT, speaker)
    for emotion in os.listdir(speaker_path):
        if os.path.isdir(os.path.join(speaker_path, emotion)):
            emotions.add(emotion)
emotions = sorted(list(emotions))

speaker2id = {spk: i for i, spk in enumerate(speakers)}
emotion2id = {emo: i for i, emo in enumerate(emotions)}

# 3. Gather metadata
metadata = []
for speaker in speakers:
    speaker_id = speaker2id[speaker]
    speaker_path = os.path.join(DATASET_ROOT, speaker)
    for emotion in os.listdir(speaker_path):
        emotion_path = os.path.join(speaker_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        emotion_id = emotion2id[emotion]
        for wav_file in os.listdir(emotion_path):
            if not wav_file.lower().endswith(".wav"):
                continue
            # Extract the last 4 digits as the wav_id index
            match = re.search(r'(\d{4})\.wav$', wav_file)
            if not match:
                continue
            wav_index = match.group(1)
            wav_id = f"arctic_a{wav_index}"
            transcript = transcript_dict.get(wav_id, None)
            if transcript is None:
                print(f"Warning: No transcript for {wav_file} (expected id: {wav_id})")
                continue
            wav_path = os.path.join(speaker, emotion, wav_file)
            metadata.append([wav_path, transcript, speaker_id, emotion_id])

# 4. Write metadata.csv
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["wav_path", "transcript", "speaker_id", "emotion_id"])
    for row in metadata:
        writer.writerow(row)

print(f"Metadata written to {OUTPUT_CSV}")
print(f"Speakers: {speaker2id}")
print(f"Emotions: {emotion2id}")
