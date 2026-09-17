import tkinter as tk
import sounddevice as sd
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write
import tempfile
import os
import threading

SAMPLE_RATE = 22050
DURATION = 3
model = joblib.load("voice_model.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def detect_voice():
    btn.config(state="disabled", text="⏳ Recording...")
    result_label.config(text="🎤 Speak Now!", fg="yellow")
    status_label.config(text="Recording for 3 seconds...", fg="gray")
    root.update()

    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    tmp = tempfile.mktemp(suffix=".wav")
    write(tmp, SAMPLE_RATE, audio)
    features = extract_features(tmp)
    prediction = model.predict([features])[0]
    os.remove(tmp)

    if prediction == 0:
        result_label.config(text="✅ REAL VOICE!", fg="#00ff88")
        status_label.config(text="Human voice detected!", fg="#00ff88")
        frame.config(bg="#003300")
    else:
        result_label.config(text="❌ FAKE VOICE!", fg="#ff4444")
        status_label.config(text="AI/Recorded voice detected!", fg="#ff4444")
        frame.config(bg="#330000")

    btn.config(state="normal", text="🎤 Start Recording")

def start_detection():
    threading.Thread(target=detect_voice).start()

# Main window
root = tk.Tk()
root.title("Fake Voice Detector")
root.geometry("500x400")
root.configure(bg="#0a0a1a")
root.resizable(False, False)

# Title
title = tk.Label(root, text="🎙️ Fake Voice Detector",
                 font=("Arial", 22, "bold"),
                 bg="#0a0a1a", fg="#7c3aed")
title.pack(pady=20)

subtitle = tk.Label(root, text="AI-Powered Voice Authentication System",
                    font=("Arial", 10),
                    bg="#0a0a1a", fg="gray")
subtitle.pack()

# Frame
frame = tk.Frame(root, bg="#1a1a2e", width=400, height=180)
frame.pack(pady=20, padx=30, fill="both")

result_label = tk.Label(frame, text="Press button to start",
                        font=("Arial", 20, "bold"),
                        bg="#1a1a2e", fg="white")
result_label.pack(pady=30)

status_label = tk.Label(frame, text="Ready to detect",
                        font=("Arial", 11),
                        bg="#1a1a2e", fg="gray")
status_label.pack()

# Button
btn = tk.Button(root, text="🎤 Start Recording",
                font=("Arial", 14, "bold"),
                bg="#7c3aed", fg="white",
                padx=30, pady=12,
                relief="flat", cursor="hand2",
                command=start_detection)
btn.pack(pady=20)

# Footer
footer = tk.Label(root, text="Powered by ML | MFCC + Random Forest",
                  font=("Arial", 8),
                  bg="#0a0a1a", fg="#444")
footer.pack(side="bottom", pady=10)

root.mainloop()