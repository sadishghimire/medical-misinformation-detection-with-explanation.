from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# from predict import predict
from app.predict import predict
import os

from openai import OpenAI

from fastapi.middleware.cors import CORSMiddleware



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



#comment Init your OpenRouter client (DeepSeek)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

class ExplainInput(BaseModel):
    text: str
    label: str
    confidence: int

@app.post("/explain")
async def explain_claim(item: ExplainInput):
    prompt = (
        f"Statement: \"{item.text}\"\n"
        f"Classification: {item.label}\n"
        f"Confidence: {item.confidence}%\n\n"
        "You are a professional health person.Provide a clear, detailed justification for why this statement was classified this way. The justification should be short and sweet no more than 4 lines.If the statement is out of medical context,provide a justification in a way so that next time user gives medical justification only.If possible provide the research paper source that was studied.the explanation should be structured and static for same statement"
    )

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528:free",
        messages=[
            {
                "role": "system",
                "content": "You are an expert fact-checking assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    explanation = response.choices[0].message.content
    return {"explanation": explanation}