import os

from src.util.openai import OpenAIClient
from src.util.token_usage import record_chat_usage

TRANSCRIBE_MODEL = "gpt-transcribe"


def transcribe(audio_file):
    client = OpenAIClient().client
    transcription = client.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=audio_file,
    )
    record_chat_usage(transcription, model=TRANSCRIBE_MODEL)
    return transcription.text

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    audio_file = open(os.path.join(dir, "test-audio.mp3"), "rb")
    print(transcribe(audio_file))
