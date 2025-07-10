"""
Data Transfer Objects (DTOs) for the IntradayJules trading system.

This module contains Pydantic models for data validation and serialization
across different components of the system.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class TradingSignal(BaseModel):
    """DTO for trading signals."""
    
    symbol: str = Field(..., description="Trading symbol")
    action: int = Field(..., ge=0, le=2, description="Trading action (0=Sell, 1=Hold, 2=Buy)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    timestamp: datetime = Field(default_factory=datetime.now, description="Signal timestamp")
    features: Optional[Dict[str, float]] = Field(None, description="Features used for signal")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OrderRequest(BaseModel):
    """DTO for order placement requests."""
    
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., regex="^(BUY|SELL)$", description="Order action")
    quantity: float = Field(..., gt=0, description="Order quantity")
    order_type: str = Field(default="MKT", description="Order type")
    limit_price: Optional[float] = Field(None, gt=0, description="Limit price for limit orders")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price for stop orders")
    time_in_force: str = Field(default="DAY", description="Time in force")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional order metadata")
    
    @validator('limit_price')
    def validate_limit_price(cls, v, values):
        if values.get('order_type') == 'LMT' and v is None:
            raise ValueError('Limit price required for limit orders')
        return v


class OrderStatus(BaseModel):
    """DTO for order status updates."""
    
    order_id: int = Field(..., description="Order ID")
    symbol: str = Field(..., description="Trading symbol")
    status: str = Field(..., description="Order status")
    filled_quantity: float = Field(default=0.0, ge=0, description="Filled quantity")
    remaining_quantity: float = Field(default=0.0, ge=0, description="Remaining quantity")
    avg_fill_price: Optional[float] = Field(None, gt=0, description="Average fill price")
    commission: Optional[float] = Field(None, description="Commission paid")
    timestamp: datetime = Field(default_factory=datetime.now, description="Status timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PositionInfo(BaseModel):
    """DTO for position information."""
    
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., description="Position quantity (positive=long, negative=short)")
    avg_cost: float = Field(..., description="Average cost basis")
    market_price: float = Field(..., gt=0, description="Current market price")
    market_value: float = Field(..., description="Current market value")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    realized_pnl: float = Field(default=0.0, description="Realized P&L")
    timestamp: datetime = Field(default_factory=datetime.now, description="Position timestamp")
    
    @validator('market_value', always=True)
    def calculate_market_value(cls, v, values):
        return values.get('quantity', 0) * values.get('market_price', 0)
    
    @validator('unrealized_pnl', always=True)
    def calculate_unrealized_pnl(cls, v, values):
        quantity = values.get('quantity', 0)
        market_price = values.get('market_price', 0)
        avg_cost = values.get('avg_cost', 0)
        return quantity * (market_price - avg_cost)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RiskMetrics(BaseModel):
    """DTO for risk metrics."""
    
    symbol: Optional[str] = Field(None, description="Symbol (None for portfolio-level)")
    var_95: Optional[float] = Field(None, description="95% Value at Risk")
    var_99: Optional[float] = Field(None, description="99% Value at Risk")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    volatility: Optional[float] = Field(None, ge=0, description="Volatility")
    beta: Optional[float] = Field(None, description="Beta (if applicable)")
    position_concentration: Optional[float] = Field(None, ge=0, le=1, description="Position concentration")
    leverage: Optional[float] = Field(None, ge=0, description="Leverage ratio")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketData(BaseModel):
    """DTO for market data."""
    
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    open_price: float = Field(..., gt=0, description="Open price")
    high_price: float = Field(..., gt=0, description="High price")
    low_price: float = Field(..., gt=0, description="Low price")
    close_price: float = Field(..., gt=0, description="Close price")
    volume: int = Field(..., ge=0, description="Volume")
    vwap: Optional[float] = Field(None, gt=0, description="Volume-weighted average price")
    
    @validator('high_price')
    def validate_high_price(cls, v, values):
        low = values.get('low_price')
        if low is not None and v < low:
            raise ValueError('High price cannot be less than low price')
        return v
    
    @validator('low_price')
    def validate_low_price(cls, v, values):
        high = values.get('high_price')
        if high is not None and v > high:
            raise ValueError('Low price cannot be greater than high price')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingMetrics(BaseModel):
    """DTO for training metrics."""
    
    episode: int = Field(..., ge=0, description="Training episode")
    total_reward: float = Field(..., description="Total episode reward")
    episode_length: int = Field(..., ge=0, description="Episode length")
    mean_reward: Optional[float] = Field(None, description="Mean reward over recent episodes")
    std_reward: Optional[float] = Field(None, ge=0, description="Standard deviation of rewards")
    exploration_rate: Optional[float] = Field(None, ge=0, le=1, description="Exploration rate")
    learning_rate: Optional[float] = Field(None, gt=0, description="Current learning rate")
    loss: Optional[float] = Field(None, ge=0, description="Training loss")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStatus(BaseModel):
    """DTO for system status information."""
    
    component: str = Field(..., description="Component name")
    status: str = Field(..., regex="^(RUNNING|STOPPED|ERROR|STARTING|STOPPING)$", description="Status")
    uptime_seconds: Optional[float] = Field(None, ge=0, description="Uptime in seconds")
    last_heartbeat: datetime = Field(default_factory=datetime.now, description="Last heartbeat")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional status metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConfigurationUpdate(BaseModel):
    """DTO for configuration updates."""
    
    component: str = Field(..., description="Component to update")
    config_path: str = Field(..., description="Configuration path (dot notation)")
    new_value: Union[str, int, float, bool, Dict, List] = Field(..., description="New configuration value")
    old_value: Optional[Union[str, int, float, bool, Dict, List]] = Field(None, description="Previous value")
    timestamp: datetime = Field(default_factory=datetime.now, description="Update timestamp")
    user: Optional[str] = Field(None, description="User making the change")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }