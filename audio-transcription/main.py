from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

def transcribe(audio_file):
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file
    )
    return transcription.text

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    audio_file = open(os.path.join(dir, "test-audio.mp3"), "rb")
    print(transcribe(audio_file))