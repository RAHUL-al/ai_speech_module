import torch
import torchaudio
import numpy as np

model = torch.jit.load("silero_vad/silero-vad/src/silero_vad/data/silero_vad.jit")
model.eval()

wav, sr = torchaudio.load("harvard.wav")

if wav.shape[0] > 1:
    wav = torch.mean(wav, dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    wav = resampler(wav)

wav = wav / wav.abs().max()
wav = wav.squeeze()

SAMPLE_RATE = 16000
WINDOW_SIZE = 512
HOP_SIZE = 160
THRESHOLD = 0.3

speech_segments = []

for i in range(0, len(wav) - WINDOW_SIZE + 1, HOP_SIZE):
    chunk = wav[i:i + WINDOW_SIZE].unsqueeze(0)
    
    with torch.no_grad():
        try:
            prob = model(chunk, SAMPLE_RATE).item()
        except Exception as e:
            print(f"Skipping chunk due to error: {e}")
            continue

    label = "Speech" if prob > THRESHOLD else "Silence"
    time = i / SAMPLE_RATE
    speech_segments.append((time, label, prob))

print("\nDetected Segments:")
print([seg for seg in speech_segments if seg[1] == "Speech"])
