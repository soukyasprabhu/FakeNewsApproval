from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the backend directory to the path so ml module can be imported correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.predict import predict_news

app = FastAPI(title="Fake News Detection API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Fake News Detection API"}

@app.post("/predict")
def predict(news: NewsInput):
    if not news.text or len(news.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="News text cannot be empty.")
    
    try:
        result = predict_news(news.text)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model files not found. The backend needs to be trained first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
