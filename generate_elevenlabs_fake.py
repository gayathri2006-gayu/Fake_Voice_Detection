import asyncio
import edge_tts
import os
import random

output_folder = "dataset/fake"
os.makedirs(output_folder, exist_ok=True)


texts = [
    "Hello, this is a generated fake voice sample.",
    "Artificial intelligence is changing the world.",
    "Voice detection system testing sample.",
    "This audio is created using text to speech technology.",
    "Machine learning helps to identify fake voices.",
    "Cyber security is important in today's digital world.",
    "Deep learning models are used for voice analysis.",
    "Fake voice detection improves online safety.",
    "Technology is developing every day.",
    "Voice authentication needs strong security.",
    "Artificial intelligence can generate human like voices.",
    "This sample is created for testing purposes.",
    "Machine learning helps computers learn from data.",
    "Audio processing is an important field.",
    "Speech recognition systems use artificial intelligence."
]

voices = [
     # en-US
    "en-US-AriaNeural",
    "en-US-AnaNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-MichelleNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
    "en-US-BrianNeural",
    "en-US-EmmaNeural",
    "en-US-AvaNeural",
    "en-US-AndrewNeural",
    # en-GB
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-GB-LibbyNeural",
    "en-GB-ThomasNeural",
    "en-GB-MaisieNeural",
    # en-AU
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    # en-IN
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    # en-CA
    "en-CA-ClaraNeural",
    "en-CA-LiamNeural",
    # en-IE / en-ZA / en-NZ / en-PH etc
    "en-IE-ConnorNeural",
    "en-IE-EmilyNeural",
    "en-ZA-LeahNeural",
    "en-ZA-LukeNeural",
    "en-NZ-MitchellNeural",
    "en-NZ-MollyNeural",
    "en-PH-JamesNeural",
    "en-PH-RosaNeural",

]


# 10 sentences × 100 = 1000 audio files
texts = texts * 200


async def generate_voice(text, filename, voice):

    for attempt in range(3):  # Retry 3 times
        try:
            communicate = edge_tts.Communicate(
                text,
                voice
            )

            await communicate.save(filename)
            return True

        except Exception as e:
            print(f"Retry {attempt+1}/3 failed:", e)
            await asyncio.sleep(3)

    return False


async def main():

    # Continue from existing files
    existing_files = os.listdir(output_folder)

    start_number = 1

    if existing_files:
        numbers = []

        for file in existing_files:
            if file.startswith("fake_") and file.endswith(".mp3"):
                num = file.replace("fake_", "").replace(".mp3", "")
                if num.isdigit():
                    numbers.append(int(num))

        if numbers:
            start_number = max(numbers) + 1


    for index, text in enumerate(texts, start=start_number):

        filename = f"{output_folder}/fake_{index}.mp3"

        # Skip if already exists
        if os.path.exists(filename):
            continue

        voice = random.choice(voices)

        success = await generate_voice(
            text,
            filename,
            voice
        )

        if success:
            print(
                f"Generated {filename} using {voice}"
            )
        else:
            print(
                f"Failed {filename}"
            )


asyncio.run(main())