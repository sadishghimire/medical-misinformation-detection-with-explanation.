# app/explain.py

from openai import OpenAI
from pathlib import Path
import os
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

load_dotenv()
api_key=os.getenv("OPENROUTER_API_KEY")

# Load OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
    #"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"
   
)
def generate_explanation(statement: str, classification: str) -> str:
    """
    Uses LLM to generate a reason for the classification.
    """
    prompt = (
        f"Statement: \"{statement}\"\n"
        f"Classification: {classification}\n\n"
        f"Explain briefly why this statement is classified as {classification}. "
        f"Give clear reasoning in 2-3 sentences.If the statement is other than health domain do not provide answer rather give answer by requesting to give input related to health domain.If the statement is related to health but not related to COVID then give a explanation but request saying that the classification model was trained on covid dataset and other than covid related statements the classification may not be accurate"
    )

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528:free",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    explanation = response.choices[0].message.content
    return explanation
