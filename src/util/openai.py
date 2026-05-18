from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# TODO: Define the models for each task

class OpenAIClient:
    def __init__(self):
        self.client = client