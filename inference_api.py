#!/usr/bin/env python3
"""
🤖 INFERENCE API SERVICE
FastAPI service that loads production model and provides real-time trading recommendations
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import redis
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Model loading
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize

# Local imports
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Trading Inference API", version="1.0.0")

# Global model state
model_state = {
    "model": None,
    "vec_normalize": None,
    "env": None,
    "loaded_at": None,
    "model_path": None
}

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class MarketData(BaseModel):
    timestamp: float
    nvda_price: float
    nvda_volume: int
    msft_price: float 
    msft_volume: int

class TradingRecommendation(BaseModel):
    timestamp: float
    action: int
    action_name: str
    confidence: float
    portfolio_value: float
    positions: Dict[str, float]
    risk_metrics: Dict[str, float]

class ModelStatus(BaseModel):
    loaded: bool
    model_path: Optional[str]
    loaded_at: Optional[str]
    last_inference: Optional[str]
    total_inferences: int

# Model loading
def load_production_model(model_path: str = "deploy_models/dual_ticker_prod_20250731_step201k_stable.zip"):
    """Load the production model for inference"""
    
    logger.info(f"🤖 Loading production model: {model_path}")
    
    try:
        # Load the model
        model = RecurrentPPO.load(model_path)
        logger.info("✅ Model loaded successfully")
        
        # Create environment for normalization (simplified for inference)
        env = None  # Skip environment creation for inference-only mode
        
        # Try to load VecNormalize if available
        vec_normalize_path = model_path.replace('.zip', '_vecnorm.pkl')
        vec_normalize = None
        if Path(vec_normalize_path).exists():
            try:
                vec_normalize = VecNormalize.load(vec_normalize_path)
                logger.info("✅ VecNormalize loaded")
            except Exception as e:
                logger.warning(f"VecNormalize not loaded: {e}")
        
        # Update global state
        model_state.update({
            "model": model,
            "vec_normalize": vec_normalize,
            "env": env,
            "loaded_at": datetime.now().isoformat(),
            "model_path": model_path
        })
        
        logger.info("🎉 Production model ready for inference")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# Inference counters
inference_stats = {
    "total_inferences": 0,
    "last_inference": None,
    "error_count": 0
}

def get_action_name(action: int) -> str:
    """Convert action index to human-readable name"""
    action_names = {
        0: "HOLD_BOTH",
        1: "BUY_NVDA_HOLD_MSFT", 
        2: "SELL_NVDA_HOLD_MSFT",
        3: "HOLD_NVDA_BUY_MSFT",
        4: "BUY_BOTH",
        5: "SELL_NVDA_BUY_MSFT", 
        6: "HOLD_NVDA_SELL_MSFT",
        7: "BUY_NVDA_SELL_MSFT",
        8: "SELL_BOTH"
    }
    return action_names.get(action, f"UNKNOWN_{action}")

async def process_market_data():
    """Background task to process Redis stream data"""
    
    logger.info("🔄 Starting market data processing loop")
    
    while True:
        try:
            if not model_state["model"]:
                await asyncio.sleep(5)
                continue
                
            # Read from Redis stream
            messages = redis_client.xread({'polygon:ticks': '$'}, count=1, block=1000)
            
            if not messages:
                continue
                
            for stream_name, stream_messages in messages:
                for message_id, fields in stream_messages:
                    await process_single_tick(fields)
                    
        except Exception as e:
            logger.error(f"❌ Error in market data processing: {e}")
            await asyncio.sleep(5)

async def process_single_tick(tick_data: Dict[str, str]):
    """Process a single market tick and generate recommendation"""
    
    try:
        # Parse tick data
        timestamp = float(tick_data.get('ts', time.time()))
        symbol = tick_data.get('sym', '')
        price = float(tick_data.get('p', 0))
        volume = int(tick_data.get('s', 0))
        
        # We need both NVDA and MSFT data for inference
        # For now, we'll simulate having both (production would buffer)
        if symbol not in ['NVDA', 'MSFT']:
            return
            
        # Create observation (simplified - production would use proper feature engineering)
        observation = create_observation_from_tick(tick_data)
        
        if observation is None:
            return
            
        # Get model prediction
        model = model_state["model"]
        action, _states = model.predict(observation, deterministic=True)
        
        # Calculate confidence (simplified)
        confidence = 0.75  # In production, use model uncertainty
        
        # Create recommendation
        recommendation = TradingRecommendation(
            timestamp=timestamp,
            action=int(action),
            action_name=get_action_name(int(action)),
            confidence=confidence,
            portfolio_value=100000.0,  # Would be calculated from current positions
            positions={"NVDA": 0.0, "MSFT": 0.0},  # Would be actual positions
            risk_metrics={"drawdown": 0.0, "volatility": 0.0}
        )
        
        # Publish to Redis
        redis_client.xadd("trading:recommendations", recommendation.dict())
        
        # Update stats
        inference_stats["total_inferences"] += 1
        inference_stats["last_inference"] = datetime.now().isoformat()
        
        logger.debug(f"📊 Generated recommendation: {recommendation.action_name}")
        
    except Exception as e:
        logger.error(f"❌ Error processing tick: {e}")
        inference_stats["error_count"] += 1

def create_observation_from_tick(tick_data: Dict[str, str]) -> Optional[np.ndarray]:
    """Create observation array from tick data (simplified)"""
    
    try:
        # In production, this would use proper feature engineering
        # For now, create a dummy observation matching the model's expected input
        
        # Dual ticker model expects 26-dimensional observation
        observation = np.zeros(26, dtype=np.float32)
        
        # Fill with some basic features (simplified)
        price = float(tick_data.get('p', 100.0))
        volume = float(tick_data.get('s', 1000))
        
        # Normalize and fill observation
        observation[0] = price / 1000.0  # Normalized price
        observation[1] = volume / 10000.0  # Normalized volume
        # ... rest would be filled with technical indicators
        
        return observation.reshape(1, -1)
        
    except Exception as e:
        logger.error(f"❌ Error creating observation: {e}")
        return None

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    
    logger.info("🚀 Starting Inference API service")
    
    # Load production model
    success = load_production_model()
    if not success:
        logger.error("❌ Failed to load production model on startup")
    
    # Start background processing
    asyncio.create_task(process_market_data())

@app.get("/")
async def root():
    return {"message": "Trading Inference API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    model_loaded = model_state["model"] is not None
    redis_connected = True
    
    try:
        redis_client.ping()
    except:
        redis_connected = False
    
    return {
        "status": "healthy" if model_loaded and redis_connected else "unhealthy",
        "model_loaded": model_loaded,
        "redis_connected": redis_connected,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """Get model loading status"""
    
    return ModelStatus(
        loaded=model_state["model"] is not None,
        model_path=model_state["model_path"],
        loaded_at=model_state["loaded_at"],
        last_inference=inference_stats["last_inference"],
        total_inferences=inference_stats["total_inferences"]
    )

@app.post("/model/reload")
async def reload_model(model_path: str = "deploy_models/dual_ticker_prod_20250731_step201k_stable.zip"):
    """Reload the production model"""
    
    logger.info(f"🔄 Reloading model: {model_path}")
    
    success = load_production_model(model_path)
    
    if success:
        return {"status": "success", "message": f"Model reloaded: {model_path}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.post("/inference", response_model=TradingRecommendation)
async def get_trading_recommendation(market_data: MarketData):
    """Get trading recommendation for given market data"""
    
    if not model_state["model"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create observation from market data
        observation = np.array([
            market_data.nvda_price / 1000.0,
            market_data.nvda_volume / 10000.0,
            market_data.msft_price / 1000.0,
            market_data.msft_volume / 10000.0,
            # ... fill remaining 22 features with zeros for now
            *[0.0] * 22
        ], dtype=np.float32).reshape(1, -1)
        
        # Get prediction
        model = model_state["model"]
        action, _states = model.predict(observation, deterministic=True)
        
        # Create recommendation
        recommendation = TradingRecommendation(
            timestamp=market_data.timestamp,
            action=int(action),
            action_name=get_action_name(int(action)),
            confidence=0.75,
            portfolio_value=100000.0,
            positions={"NVDA": 0.0, "MSFT": 0.0},
            risk_metrics={"drawdown": 0.0, "volatility": 0.0}
        )
        
        # Update stats
        inference_stats["total_inferences"] += 1
        inference_stats["last_inference"] = datetime.now().isoformat()
        
        return recommendation
        
    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/stats")
async def get_inference_stats():
    """Get inference statistics"""
    
    return {
        "total_inferences": inference_stats["total_inferences"],
        "last_inference": inference_stats["last_inference"],
        "error_count": inference_stats["error_count"],
        "model_loaded": model_state["model"] is not None,
        "uptime": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )