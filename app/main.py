from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.predict import predict
from app.Explain import generate_explanation
from app.Evidence import generate_evidence
from pathlib import Path
import os
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Debug: Print current working directory
print(f"Current directory: {os.getcwd()}")

# 1. Verify .env location
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
print(f"Looking for .env at: {env_path}")


if not env_path.exists():
    raise FileNotFoundError(f"No .env file found at {env_path}")

load_dotenv(env_path)
# 3. Debug print all variables
print("Loaded environment variables:", os.environ)

# 4. Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
print(f"API Key: {'****' + api_key[-4:] if api_key else 'NOT FOUND'}")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not loaded from .env")


# ---- Load model and tokenizer ----
# MODEL_PATH = "../saved_model"  # adjust this path if your model is elsewhere

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model")

print(f"Resolved model path: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# ---- FastAPI app ----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for dev http://127.0.0.1:5500/UI/index.html
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Pydantic input ----
class Item(BaseModel):
    text: str

# GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
# ---- Label mapping ----
id2label = {0: "False", 1: "True", 2: "Unverified"}

@app.post("/predict")
async def predict(item: Item):
    print(f"Received: {item.text}")
    inputs = tokenizer(item.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    label = id2label[pred.item()]
    confidence = round(conf.item() * 100, 2)

    return {"label": label, "confidence": confidence}



# 2. Verify key is loaded
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("Missing OPENROUTER_API_KEY in .env file")


#comment Init your OpenRouter client (DeepSeek)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
    #"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"
   
)

class ExplainInput(BaseModel):
    text: str
    label: str
    confidence: int

# @app.post("/explain")
# async def explain_claim(item: ExplainInput):
#     prompt = (
#         f"Statement: \"{item.text}\"\n"
#         f"Classification: {item.label}\n"
#         f"Confidence: {item.confidence}%\n\n"
#         "You are a professional health person.Provide a clear, detailed justification for why this statement was classified this way. The justification should be short and sweet no more than 4 lines.If the statement is out of medical context,provide a justification in a way so that next time user gives medical justification only.If possible provide the research paper source that was studied.the explanation should be structured and static for same statement"
#     )

#     response = client.chat.completions.create(
#         model="deepseek/deepseek-r1-0528:free",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an expert fact-checking assistant."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )

#     explanation = response.choices[0].message.content
#     return {"explanation": explanation}

@app.post("/explain")
async def explain_route(request: Request):
    body = await request.json()
    statement = body['text']
    classification = body['label']
    explanation = generate_explanation(statement, classification)
    return {"explanation": explanation}

@app.post("/evidence")
async def evidence_route(request: Request):
    body = await request.json()
    statement = body['text']
    classification = body['label']
    evidence = generate_evidence(statement, classification)
    return {"evidence": evidence}