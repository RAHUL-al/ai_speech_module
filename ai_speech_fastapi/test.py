import os
import requests
from dotenv import load_dotenv

load_dotenv()

def speech_to_text() -> str:
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        headers = {
            "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
            "Content-Type": "audio/wav"
        }

        if not os.path.exists("harvard.wav"):
            print(f"[Error] Audio file not found: ")
            return ""

        try:
            with open("harvard.wav", "rb") as f:
                data = f.read()

            response = requests.post(API_URL, headers=headers, data=data)
            response.raise_for_status()

            result = response.json()
            text = result.get("text", "").strip()
            print(f"Transcribed [{os.path.basename("harvard.wav")}]: {text}")
            return text

        except Exception as e:
            print(f"[Error] Failed to transcribe harvard.wav: {e}")
            return ""
        
speech_to_text()