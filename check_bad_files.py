import os
import numpy as np
import librosa

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 130:
        pad = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :130]
    return mfcc


def check_folder(folder):
    bad_files = []
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"\nChecking {len(files)} files in {folder}...")

    for f in files:
        path = os.path.join(folder, f)
        try:
            mfcc = extract_features(path)
            if np.isnan(mfcc).any() or np.isinf(mfcc).any():
                bad_files.append(path)
                print(f"  BAD (NaN/Inf): {f}")
        except Exception as e:
            bad_files.append(path)
            print(f"  BAD (error: {e}): {f}")

    return bad_files


def main():
    bad_real = check_folder(REAL_DIR)
    bad_fake = check_folder(FAKE_DIR)

    all_bad = bad_real + bad_fake

    print(f"\n\nTotal bad files found: {len(all_bad)}")

    if all_bad:
        confirm = input("\nDelete these bad files? (yes/no): ")
        if confirm.strip().lower() == "yes":
            for path in all_bad:
                os.remove(path)
                print(f"Deleted: {path}")
            print("\nAll bad files deleted.")
        else:
            print("No files deleted.")
    else:
        print("No corrupted files found!")


if __name__ == "__main__":
    main()