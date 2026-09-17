import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib
from scipy.io.wavfile import write
import tempfile
import os

SAMPLE_RATE = 16000
DURATION = 3


# ---------------- MODEL DEFINITION (must match train_model.py) ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ---------------- LOAD MODEL ----------------
model = CNN()
model.load_state_dict(torch.load("voice_cnn_best.pt", map_location="cpu"))
model.eval()
print("Loaded voice_cnn_best.pt (CNN model)")

# ---------------- LOAD NORMALIZATION STATS (must match training) ----------------
norm_stats = joblib.load("cnn_norm_stats.pkl")
DATA_MEAN = norm_stats["mean"]
DATA_STD = norm_stats["std"]
print(f"Loaded normalization stats: mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}")


# ---------------- FEATURE EXTRACTION (must match train_model.py) ----------------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 130:
        pad = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :130]
    return mfcc


def detect_voice():
    print("\n🎤 Recording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print("Recording done!")

    tmp = tempfile.mktemp(suffix=".wav")
    write(tmp, SAMPLE_RATE, audio)

    mfcc = extract_features(tmp)

    # Normalize using the EXACT same mean/std as training
    mfcc = (mfcc - DATA_MEAN) / (DATA_STD + 1e-8)

    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).reshape(1, 1, 40, 130)

    with torch.no_grad():
        prediction = model(mfcc_tensor).item()

    os.remove(tmp)

    print(f"Model raw output (0=real, 1=fake): {prediction:.4f}")

    if prediction < 0.5:
        print("✅ REAL VOICE detected!")
    else:
        print("❌ FAKE VOICE detected!")


while True:
    input("\nPress Enter to detect voice (Ctrl+C to stop)...")
    detect_voice()