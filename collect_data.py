import os
import time
import random
import sounddevice as sd
from scipy.io.wavfile import write
from sentences_200 import SENTENCES


# Audio settings
SAMPLE_RATE = 22050
DURATION = 3   # seconds

# Real voice folder
REAL_DIR = "dataset/real"

os.makedirs(REAL_DIR, exist_ok=True)


# Find next file number (avoid overwrite)
def get_next_index(folder, prefix):

    existing_files = [
        f for f in os.listdir(folder)
        if f.startswith(prefix) and f.endswith(".wav")
    ]

    max_num = 0

    for file in existing_files:
        try:
            num = int(
                file.replace(prefix, "")
                    .replace(".wav", "")
            )
            max_num = max(max_num, num)

        except:
            pass

    return max_num + 1



# Record voice function
def record_voice(filename):

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )

    sd.wait()

    filepath = os.path.join(
        REAL_DIR,
        filename
    )

    write(
        filepath,
        SAMPLE_RATE,
        audio
    )

    print("Saved:", filepath)



# Number of real samples to collect
NUM_REAL_TO_RECORD = 200


print("===== REAL VOICE COLLECTION =====")


start_index = get_next_index(
    REAL_DIR,
    "real_"
)

print(
    f"Starting from real_{start_index}.wav"
)


for i in range(NUM_REAL_TO_RECORD):

    current_index = start_index + i

    sentence = random.choice(SENTENCES)

    print(f"\n--- real_{current_index}.wav ---")
    print(f"Speak this sentence: {sentence}")
    print("Recording starts in 2 seconds... get ready...")

    time.sleep(2)

    print("Recording... speak now!")

    record_voice(
        f"real_{current_index}.wav"
    )


print("\n================================")
print("Real voice collection completed!")
print("================================")