from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import pickle
import numpy as np
from urllib.parse import urlparse
import uvicorn
from functools import lru_cache
import time

class URLRequest(BaseModel):
    url: str = Field(..., description="URL to check for phishing", min_length=1)
    
    @validator('url')
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        return v.strip()

class BatchURLRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to check", min_items=1, max_items=100)
    
    @validator('urls')
    def validate_urls(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 URLs per batch request')
        return [url.strip() for url in v if url.strip()]

class CleanURLResponse(BaseModel):
    success: bool
    original_url: str
    cleaned_url: str

class PredictionResponse(BaseModel):
    success: bool
    original_url: str
    cleaned_url: str
    prediction: str
    confidence: float
    probability_malicious: float
    probability_benign: float
    is_safe: bool
    risk_level: str
    recommendation: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    success: bool
    count: int
    total_processing_time_ms: float
    results: List[PredictionResponse]

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    original_url: Optional[str] = None

app = FastAPI(
    title="Phishing Detection API",
    description="High-performance API for detecting phishing URLs using machine learning",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = r'D:\GradProject\url-nlp\artifacts\phishing.pkl'
_model_cache = None

@lru_cache(maxsize=1)
def load_model():
    global _model_cache
    if _model_cache is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                _model_cache = pickle.load(f)
        except FileNotFoundError:
            print(f"Model file not found: {MODEL_PATH}")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return _model_cache

@lru_cache(maxsize=10000)
def clean_url(url: str) -> str:
    url = url.strip()
    
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    domain = parsed.netloc.split(':')[0]
    
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain

def get_risk_level(confidence: float, prediction: str) -> str:
    if prediction == 'benign':
        if confidence >= 90:
            return 'Very Low Risk'
        elif confidence >= 70:
            return 'Low Risk'
        else:
            return 'Moderate Risk'
    else:
        if confidence >= 90:
            return 'Critical Risk'
        elif confidence >= 70:
            return 'High Risk'
        else:
            return 'Moderate Risk'

def get_recommendation(prediction: str, confidence: float) -> str:
    if prediction == 'benign':
        if confidence >= 90:
            return 'This URL appears to be legitimate and safe to visit.'
        elif confidence >= 70:
            return 'This URL appears legitimate, but exercise normal caution.'
        else:
            return 'This URL may be legitimate, but verify before entering sensitive information.'
    else:
        if confidence >= 90:
            return 'DO NOT VISIT! This URL is highly likely to be a phishing attempt.'
        elif confidence >= 70:
            return 'WARNING: This URL is likely a phishing attempt. Avoid visiting.'
        else:
            return 'CAUTION: This URL may be a phishing attempt. Verify before visiting.'

def predict_url_fast(url: str, model) -> dict:
    start_time = time.perf_counter()
    
    original_url = url
    cleaned_url = clean_url(url)
    
    prediction = model.predict([cleaned_url])[0]
    probabilities = model.predict_proba([cleaned_url])[0]
    classes = model.classes_
    
    pred_idx = np.where(classes == prediction)[0][0]
    confidence = probabilities[pred_idx] * 100
    
    malicious_idx = np.where(classes == 'malicious')[0][0]
    benign_idx = np.where(classes == 'benign')[0][0]
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    return {
        'success': True,
        'original_url': original_url,
        'cleaned_url': cleaned_url,
        'prediction': prediction,
        'confidence': round(confidence, 2),
        'probability_malicious': round(probabilities[malicious_idx] * 100, 2),
        'probability_benign': round(probabilities[benign_idx] * 100, 2),
        'is_safe': prediction == 'benign',
        'risk_level': get_risk_level(confidence, prediction),
        'recommendation': get_recommendation(prediction, confidence),
        'processing_time_ms': round(processing_time, 2)
    }

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print("API ready to accept requests")
    except Exception as e:
        print(f"Failed to start API: {e}")
        raise
    print("="*60 + "\n")

@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "Phishing Detection API",
        "version": "2.0",
        "performance": "High-performance with FastAPI + Uvicorn",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "POST /predict": "Predict single URL",
            "POST /predict_batch": "Predict multiple URLs (max 100)",
            "POST /clean": "Clean URL only",
            "GET /health": "Health check"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    try:
        model = load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "cache_info": clean_url.cache_info()._asdict()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: URLRequest):
    try:
        model = load_model()
        result = predict_url_fast(request.url, model)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4,
        log_level="info"
    )