#!/usr/bin/env python3
"""
🤖 SIMPLE INFERENCE API - MOCK VERSION FOR TESTING
Lightweight API that returns mock trading decisions for paper trading testing
"""

import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple Inference API", version="1.0.0")

# Mock model state
model_loaded = True
inference_count = 0

class InferenceRequest(BaseModel):
    features: Dict[str, float]
    symbol: str
    timestamp: Optional[float] = None

class InferenceResponse(BaseModel):
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    symbol: str
    timestamp: float
    model_version: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_inferences: int

# Startup time
startup_time = time.time()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        uptime_seconds=time.time() - startup_time,
        total_inferences=inference_count
    )

@app.get("/status")
async def get_status():
    """Status endpoint"""
    return {
        "status": "operational",
        "model": "mock_201k_stable",
        "version": "1.0.0",
        "uptime": time.time() - startup_time,
        "inferences": inference_count
    }

@app.post("/inference", response_model=InferenceResponse)
async def get_inference(request: InferenceRequest):
    """Get trading inference - MOCK VERSION"""
    global inference_count
    
    start_time = time.time()
    inference_count += 1
    
    # Mock decision logic based on simple rules
    symbol = request.symbol
    features = request.features
    
    # Simple mock logic
    if "rsi" in features:
        rsi = features["rsi"]
        if rsi < 30:
            action = "BUY"
            confidence = 0.8
        elif rsi > 70:
            action = "SELL" 
            confidence = 0.8
        else:
            action = "HOLD"
            confidence = 0.6
    else:
        # Random decision for testing
        actions = ["BUY", "SELL", "HOLD", "HOLD", "HOLD"]  # Bias toward HOLD
        action = random.choice(actions)
        confidence = random.uniform(0.5, 0.9)
    
    latency_ms = (time.time() - start_time) * 1000
    
    logger.info(f"🤖 Inference #{inference_count}: {symbol} -> {action} (conf: {confidence:.2f})")
    
    return InferenceResponse(
        action=action,
        confidence=confidence,
        symbol=symbol,
        timestamp=request.timestamp or time.time(),
        model_version="mock_201k_stable",
        latency_ms=latency_ms
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple Inference API - Mock Version",
        "status": "operational",
        "endpoints": ["/health", "/status", "/inference"]
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Simple Inference API (Mock Version)")
    logger.info("   This is a lightweight mock API for testing paper trading")
    logger.info("   Port: 8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)