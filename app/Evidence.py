# app/evidence.py

from openai import OpenAI
from pathlib import Path
import os

from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
api_key=os.getenv("OPENROUTER_API_KEY")

# Load OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
    #"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"
)

def generate_evidence(statement: str, classification: str) -> str:
    """
    Uses LLM to generate supporting evidence or source for the statement.
    """
    prompt = (
        f"Statement: \"{statement}\"\n"
        f"Classification: {classification}\n\n"
        f"Provide a short piece of factual evidence or a reliable source to support this classification. "
        f"If no strong evidence exists, say 'No direct evidence available'. Keep it under 50 words."
    )

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528:free",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    evidence = response.choices[0].message.content
    return evidence
