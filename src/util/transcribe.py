import os

from src.util.openai import OpenAIClient

def transcribe(audio_file):
    client = OpenAIClient().client
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file
    )
    return transcription.text

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    audio_file = open(os.path.join(dir, "test-audio.mp3"), "rb")
    print(transcribe(audio_file))