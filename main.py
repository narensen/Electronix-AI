from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os, json

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Predict(BaseModel):
    text: str

MODEL_PATH = "./model"
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

if len(os.listdir(MODEL_PATH)) == 0:
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
else:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    label_path = os.path.join(MODEL_PATH, "labels.json")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            label_map = json.load(f)
    else:
        label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.post("/predict")
def predict(request: Predict):
    try:
        result = sentiment_pipeline(request.text)[0]
        label = label_map.get(result["label"], result["label"])
        return {"label": label, "score": round(result["score"], 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
