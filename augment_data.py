import os
import numpy as np
import librosa
import soundfile as sf

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"

# How many augmented copies to create PER ORIGINAL FILE.
# 3 copies per file means: 1150 original -> 1150*3 = 3450 new files (+ 1150 original = 4600 total)
COPIES_PER_FILE = 1

SR = 22050


def load_audio(path):
    y, sr = librosa.load(path, sr=SR)
    return y, sr


def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)


def augment_one(y, sr, variant):
    """Apply a different augmentation depending on variant index."""
    if variant == 0:
        # pitch shift up slightly + light noise
        y_aug = pitch_shift(y, sr, n_steps=2)
        y_aug = add_noise(y_aug, 0.003)
    elif variant == 1:
        # pitch shift down slightly
        y_aug = pitch_shift(y, sr, n_steps=-2)
    elif variant == 2:
        # speed change + noise
        try:
            y_aug = time_stretch(y, rate=1.1)
        except Exception:
            y_aug = y
        y_aug = add_noise(y_aug, 0.004)
    else:
        # fallback: just add noise
        y_aug = add_noise(y, 0.005)
    return y_aug


def process_folder(folder):
    # Support both .wav and .mp3 input files (user's real recordings are .mp3)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".mp3"))]
    # Only look at ORIGINAL files (skip ones we already generated, marked with "_aug")
    original_files = [f for f in files if "_aug" not in f]

    print(f"\nProcessing folder: {folder}")
    print(f"Original files found: {len(original_files)}")

    created = 0
    for filename in original_files:
        filepath = os.path.join(folder, filename)
        try:
            y, sr = load_audio(filepath)
        except Exception as e:
            print(f"Skipping {filename}, could not load: {e}")
            continue

        base_name = os.path.splitext(filename)[0]

        for variant in range(COPIES_PER_FILE):
            try:
                y_aug = augment_one(y, sr, variant)
                # Always save augmented output as .wav (safe, lossless format)
                out_name = f"{base_name}_aug{variant}.wav"
                out_path = os.path.join(folder, out_name)
                sf.write(out_path, y_aug, sr)
                created += 1
            except Exception as e:
                print(f"Failed to augment {filename} variant {variant}: {e}")

    print(f"New augmented files created in {folder}: {created}")


if __name__ == "__main__":
    # Only augment REAL voice folder — fake already has enough samples (~4846).
    # Augmenting fake too would widen the imbalance instead of fixing it.
    process_folder(REAL_DIR)
    # process_folder(FAKE_DIR)   # intentionally skipped

    total_real = len(
        [f for f in os.listdir(REAL_DIR) if f.lower().endswith((".wav", ".mp3"))]
    )
    total_fake = len(
        [f for f in os.listdir(FAKE_DIR) if f.lower().endswith((".wav", ".mp3"))]
    )

    print("\n---- DONE ----")
    print("Total real files now:", total_real)
    print("Total fake files now:", total_fake)