import os
import librosa
import soundfile as sf

SOURCE_DIR = os.path.join(os.environ["USERPROFILE"], "Downloads", "LibriSpeech", "dev-clean")
DEST_DIR = "dataset/real"
TARGET_SR = 22050

# How many files to convert (2703 available, pick a reasonable number)
MAX_FILES = 500

os.makedirs(DEST_DIR, exist_ok=True)


def get_next_index(folder, prefix):
    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".wav")]
    nums = []
    for f in existing:
        try:
            nums.append(int(f.replace(prefix, "").replace(".wav", "")))
        except ValueError:
            continue
    return max(nums) + 1 if nums else 1


start_idx = get_next_index(DEST_DIR, "real_")
print(f"Starting from real_{start_idx}.wav")

count = 0
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(".flac"):
            if count >= MAX_FILES:
                break

            src_path = os.path.join(root, file)
            idx = start_idx + count
            dest_path = os.path.join(DEST_DIR, f"real_{idx}.wav")

            try:
                audio, sr = librosa.load(src_path, sr=TARGET_SR)
                sf.write(dest_path, audio, TARGET_SR)
                count += 1
                if count % 50 == 0:
                    print(f"Converted {count}/{MAX_FILES}")
            except Exception as e:
                print(f"Failed {file}: {e}")

    if count >= MAX_FILES:
        break

print(f"\nDone! Converted {count} files to {DEST_DIR}")