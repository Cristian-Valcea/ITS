#!/usr/bin/env python3
"""
AI Inference Service for Live Trading
Lightweight FastAPI microservice that loads the 201K dual-ticker model
and exposes /predict endpoint for real-time trading decisions
"""

import os
import logging
import threading
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model loading
try:
    from sb3_contrib import RecurrentPPO
    SB3_AVAILABLE = True
except ImportError:
    try:
        from stable_baselines3 import PPO as RecurrentPPO  # Fallback to regular PPO
        SB3_AVAILABLE = True
        logger.warning("Using regular PPO instead of RecurrentPPO")
    except ImportError:
        SB3_AVAILABLE = False
        logger.error("stable_baselines3 not available - inference service cannot start")

# FastAPI app
app = FastAPI(
    title="IntradayJules AI Inference Service", 
    description="Real-time trading decision API using 201K dual-ticker model",
    version="1.0.0"
)

# Global model instance
model = None
model_loaded = False
model_lock = threading.Lock()  # Thread safety for model predictions

class ObservationRequest(BaseModel):
    """Request format for model prediction"""
    observation: List[float]
    timestamp: str = None
    symbols: List[str] = ["NVDA", "MSFT"]
    
    @validator('observation')
    def validate_observation_length(cls, v):
        if len(v) != 26:
            raise ValueError(f'Observation must have exactly 26 features, got {len(v)}')
        return v
    
    @validator('observation', each_item=True)
    def validate_observation_values(cls, v):
        if not isinstance(v, (int, float)) or not (-1000 <= v <= 1000):
            raise ValueError(f'Invalid observation value: {v}. Must be numeric and reasonable.')
        return float(v)

class PredictionResponse(BaseModel):
    """Response format with trading action"""
    action: int
    confidence: float = 0.0
    action_name: str = ""
    model_version: str = "201K_dual_ticker"
    timestamp: str = ""

def load_model():
    """Load the 201K dual-ticker model"""
    global model, model_loaded
    
    # Model path
    model_path = Path("deploy_models/dual_ticker_prod_20250731_step201k_stable.zip")
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        logger.info(f"Loading model from {model_path}...")
        model = RecurrentPPO.load(str(model_path))
        model_loaded = True
        logger.info("✅ 201K dual-ticker model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def action_to_name(action: int) -> str:
    """Convert numeric action to human-readable name"""
    action_map = {
        0: "HOLD_HOLD",     # Hold both
        1: "HOLD_BUY",      # Hold NVDA, Buy MSFT  
        2: "HOLD_SELL",     # Hold NVDA, Sell MSFT
        3: "BUY_HOLD",      # Buy NVDA, Hold MSFT
        4: "BUY_BUY",       # Buy both
        5: "BUY_SELL",      # Buy NVDA, Sell MSFT
        6: "SELL_HOLD",     # Sell NVDA, Hold MSFT
        7: "SELL_BUY",      # Sell NVDA, Buy MSFT
        8: "SELL_SELL"      # Sell both
    }
    return action_map.get(action, f"UNKNOWN_{action}")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    if not SB3_AVAILABLE:
        logger.error("stable_baselines3 not available - service disabled")
        return
        
    success = load_model()
    if not success:
        logger.error("Failed to load model - service will return errors")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "sb3_available": SB3_AVAILABLE
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ObservationRequest):
    """Make trading prediction from observation"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not SB3_AVAILABLE:
        raise HTTPException(status_code=503, detail="stable_baselines3 not available")
    
    try:
        # Convert to numpy array (validation already done by pydantic)
        obs_array = np.array(request.observation, dtype=np.float32)
        
        # Get model prediction (thread-safe)
        with model_lock:
            action, _states = model.predict(obs_array, deterministic=True)
            action_int = int(action)
        
        # Create response
        response = PredictionResponse(
            action=action_int,
            action_name=action_to_name(action_int),
            confidence=1.0,  # PPO doesn't provide confidence, assume deterministic
            model_version="201K_dual_ticker",
            timestamp=request.timestamp or ""
        )
        
        # Structured logging for monitoring
        logger.info(f"🤖 Prediction: {response.action_name} (action={action_int}) - latency_ms=<calculated_later>")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": "RecurrentPPO",
        "version": "201K_dual_ticker", 
        "observation_space": 26,
        "action_space": 9,
        "symbols": ["NVDA", "MSFT"],
        "loaded": model_loaded
    }

if __name__ == "__main__":
    # Security: bind to localhost by default, configurable via env
    host = os.getenv('AI_SERVICE_HOST', '127.0.0.1')  # Secure default
    port = int(os.getenv('AI_SERVICE_PORT', '8000'))
    workers = int(os.getenv('AI_SERVICE_WORKERS', '1'))  # Single worker for thread safety
    
    logger.info(f"Starting AI Inference Service on {host}:{port} with {workers} worker(s)")
    
    uvicorn.run(
        "ai_inference_service:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        reload=False
    )