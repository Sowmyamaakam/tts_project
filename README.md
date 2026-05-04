# 🎙️ Text-to-Speech System (FastSpeech2)

## 📌 Overview

This project implements a **Text-to-Speech (TTS) system** based on the **FastSpeech2 architecture**, designed to generate natural-sounding speech from text input.

It includes support for:

* Multi-speaker voice generation
* Emotion-based speech (using EmoV-DB dataset)
* Duration prediction for better alignment
* Context-aware embeddings using BERT

---

## 🚀 Features

* 🔊 High-quality speech synthesis using FastSpeech2
* 👥 Multi-speaker support
* 😊 Emotion-aware speech generation
* ⏱️ Duration predictor for accurate timing
* 🧠 Optional BERT integration for improved text understanding
* ⚡ Faster inference compared to autoregressive models

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** PyTorch
* **Libraries:** NumPy, Librosa, TensorFlow (optional), Transformers
* **Dataset:** EmoV-DB

---

## 📂 Project Structure

```
├── data/                # Dataset and preprocessing files
├── models/              # Model architecture (FastSpeech2 components)
├── utils/               # Helper functions
├── train.py             # Training script
├── inference.py         # Speech generation script
├── config.py            # Hyperparameters
├── checkpoints/         # Saved models
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tts-project.git
cd tts-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

* Download **EmoV-DB dataset**
* Organize files in the following format:

```
data/
 ├── speaker_1/
 ├── speaker_2/
 └── metadata.csv
```

* Preprocess audio:

```bash
python preprocess.py
```

---

## 🧪 Usage

### ▶️ Train the model

```bash
python train.py
```

### 🔊 Generate speech

```bash
python inference.py --text "Hello, how are you?"
```

---

## 🧠 Model Architecture

FastSpeech2 consists of:

* Encoder → Converts text into embeddings
* Duration Predictor → Predicts phoneme durations
* Length Regulator → Expands sequence
* Decoder → Generates mel-spectrogram
* Vocoder → Converts spectrogram to waveform

---

## 🔍 Key Components

### 📌 Positional Encoding

Adds positional information to input sequences.

### 📌 Duration Predictor

Predicts how long each phoneme should be spoken.

### 📌 Length Regulator

Expands encoded sequence based on predicted durations.

### 📌 Speaker Embedding

Allows multi-speaker voice synthesis.

### 📌 BERT Integration

Improves contextual understanding of text input.

---

## 📈 Results

* Stable training with optimized hyperparameters
* Improved speech clarity with duration modeling
* Faster inference compared to traditional TTS models

---

## ⚠️ Implementation Notes

* Ensure proper audio preprocessing
* Use GPU for faster training
* Tune learning rate for stability
* Normalize dataset for better performance

---

## 🎯 Future Improvements

* Add real-time speech generation
* Improve emotion modeling
* Deploy as web application
* Optimize model size for mobile devices

---

## 📫 Contact

* GitHub: https://github.com/Sowmyamaakam

---

⭐ *This project demonstrates deep learning, speech processing, and real-world AI application skills.*
