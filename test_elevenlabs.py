from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")

if api_key:
    print("API Key loaded successfully ✅")
else:
    print("API Key not found ❌")