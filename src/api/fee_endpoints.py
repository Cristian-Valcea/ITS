"""
Fee Schedule API Endpoints

Provides REST API access to fee schedules and calculations.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

try:
    from ..shared.fee_schedule import get_cme_fee_schedule, calculate_cme_fee
    FEE_ENGINE_AVAILABLE = True
except ImportError:
    FEE_ENGINE_AVAILABLE = False

router = APIRouter(prefix="/api/v1/fees", tags=["fees"])


class FeeCalculationRequest(BaseModel):
    """Request model for fee calculation."""
    symbol: str
    quantity: int
    volume_ytd: Optional[int] = 0


class FeeCalculationResponse(BaseModel):
    """Response model for fee calculation."""
    symbol: str
    quantity: int
    volume_ytd: int
    fee_per_contract: float
    total_fee: float
    currency: str
    venue: str


class TradingDayRequest(BaseModel):
    """Request model for trading day fee analysis."""
    trades: List[Dict[str, Any]]  # List of {symbol, quantity} dicts


class TradingDayResponse(BaseModel):
    """Response model for trading day analysis."""
    total_volume: int
    total_fees: float
    average_fee_per_contract: float
    fee_breakdown: Dict[str, float]
    fee_drag_bps: float  # Fee drag in basis points on $100k account


@router.get("/health")
async def fee_engine_health():
    """Check fee engine health status."""
    if not FEE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fee engine not available")
    
    try:
        schedule = get_cme_fee_schedule()
        return {
            "status": "healthy",
            "venue": schedule.venue,
            "symbols_available": len(schedule.list_symbols()),
            "last_updated": schedule.last_updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fee engine error: {str(e)}")


@router.get("/venues")
async def list_venues():
    """List available fee venues."""
    if not FEE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fee engine not available")
    
    return {
        "venues": ["CME"],
        "default_venue": "CME"
    }


@router.get("/symbols")
async def list_symbols(venue: str = Query("CME", description="Venue name")):
    """List available symbols for a venue."""
    if not FEE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fee engine not available")
    
    if venue.upper() != "CME":
        raise HTTPException(status_code=404, detail=f"Venue {venue} not supported")
    
    try:
        schedule = get_cme_fee_schedule()
        symbols = schedule.list_symbols()
        
        # Get fee info for each symbol
        symbol_info = {}
        for symbol in symbols:
            info = schedule.get_fee_info(symbol)
            symbol_info[symbol] = {
                "fee_per_contract": info.get('base_fee'),
                "currency": info.get('currency'),
                "description": info.get('description'),
                "is_tiered": info.get('is_tiered', False)
            }
        
        return {
            "venue": venue.upper(),
            "symbols": symbol_info,
            "total_symbols": len(symbols)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing symbols: {str(e)}")


@router.get("/calculate/{symbol}")
async def calculate_fee(
    symbol: str,
    quantity: int = Query(..., description="Number of contracts"),
    volume_ytd: int = Query(0, description="Year-to-date volume for tiered pricing"),
    venue: str = Query("CME", description="Venue name")
):
    """Calculate fee for a specific trade."""
    if not FEE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fee engine not available")
    
    if venue.upper() != "CME":
        raise HTTPException(status_code=404, detail=f"Venue {venue} not supported")
    
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    
    try:
        schedule = get_cme_fee_schedule()
        
        # Get fee info
        fee_obj = schedule.lookup(symbol, volume_ytd)
        total_fee = schedule.calculate_total_fee(symbol, quantity, volume_ytd)
        
        return FeeCalculationResponse(
            symbol=symbol,
            quantity=quantity,
            volume_ytd=volume_ytd,
            fee_per_contract=fee_obj.fee_per_side,
            total_fee=total_fee,
            currency=fee_obj.currency,
            venue=schedule.venue
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating fee: {str(e)}")


@router.post("/calculate/batch")
async def calculate_batch_fees(request: List[FeeCalculationRequest]):
    """Calculate fees for multiple trades."""
    if not FEE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fee engine not available")
    
    if len(request) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 calculations per batch")
    
    try:
        schedule = get_cme_fee_schedule()
        results = []
        
        for calc_request in request:
            fee_obj = schedule.lookup(calc_request.symbol, calc_request.volume_ytd)
            total_fee = schedule.calculate_total_fee(
                calc_request.symbol, 
                calc_request.quantity, 
                calc_request.volume_ytd
            )
            
            results.append(FeeCalculationResponse(
                symbol=calc_request.symbol,
                quantity=calc_request.quantity,
                volume_ytd=calc_request.volume_ytd,
                fee_per_contract=fee_obj.fee_per_side,
                total_fee=total_fee,
                currency=fee_obj.currency,
                venue=schedule.venue
            ))
        
        return {
            "calculations": results,
            "total_calculations": len(results),
            "total_fees": sum(r.total_fee for r in results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch calculation: {str(e)}")


@router.post("/analyze/trading-day")
async def analyze_trading_day(request: TradingDayRequest):
    """Analyze fees for a full trading day."""
    if not FEE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fee engine not available")
    
    if len(request.trades) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 trades per analysis")
    
    try:
        schedule = get_cme_fee_schedule()
        
        total_volume = 0
        total_fees = 0.0
        fee_breakdown = {}
        
        for trade in request.trades:
            symbol = trade.get('symbol')
            quantity = trade.get('quantity', 0)
            
            if not symbol or quantity <= 0:
                continue
            
            fee = calculate_cme_fee(symbol, quantity)
            total_fees += fee
            total_volume += quantity
            
            fee_breakdown[symbol] = fee_breakdown.get(symbol, 0) + fee
        
        average_fee = total_fees / total_volume if total_volume > 0 else 0
        fee_drag_bps = (total_fees / 100000) * 10000  # Basis points on $100k account
        
        return TradingDayResponse(
            total_volume=total_volume,
            total_fees=total_fees,
            average_fee_per_contract=average_fee,
            fee_breakdown=fee_breakdown,
            fee_drag_bps=fee_drag_bps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing trading day: {str(e)}")


@router.get("/info/{symbol}")
async def get_symbol_info(
    symbol: str,
    venue: str = Query("CME", description="Venue name")
):
    """Get detailed information about a symbol's fee structure."""
    if not FEE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fee engine not available")
    
    if venue.upper() != "CME":
        raise HTTPException(status_code=404, detail=f"Venue {venue} not supported")
    
    try:
        schedule = get_cme_fee_schedule()
        info = schedule.get_fee_info(symbol)
        
        return {
            "symbol": symbol,
            "venue": venue.upper(),
            **info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting symbol info: {str(e)}")


@router.get("/venue/{venue}/info")
async def get_venue_info(venue: str):
    """Get information about a venue."""
    if not FEE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fee engine not available")
    
    if venue.upper() != "CME":
        raise HTTPException(status_code=404, detail=f"Venue {venue} not supported")
    
    try:
        schedule = get_cme_fee_schedule()
        info = schedule.get_venue_info()
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting venue info: {str(e)}")


# Example usage endpoints for testing
@router.get("/examples/micro-futures")
async def micro_futures_example():
    """Example fee calculations for CME micro futures."""
    if not FEE_ENGINE_AVAILABLE:
        return {"error": "Fee engine not available"}
    
    examples = [
        {"symbol": "MES", "quantity": 10, "description": "10 Micro E-mini S&P 500 contracts"},
        {"symbol": "MNQ", "quantity": 5, "description": "5 Micro E-mini NASDAQ-100 contracts"},
        {"symbol": "M2K", "quantity": 20, "description": "20 Micro E-mini Russell 2000 contracts"},
        {"symbol": "MCL", "quantity": 3, "description": "3 Micro WTI Crude Oil contracts"}
    ]
    
    results = []
    total_fees = 0
    
    for example in examples:
        try:
            fee = calculate_cme_fee(example["symbol"], example["quantity"])
            total_fees += fee
            results.append({
                **example,
                "fee": fee,
                "fee_per_contract": fee / example["quantity"]
            })
        except Exception as e:
            results.append({
                **example,
                "error": str(e)
            })
    
    return {
        "examples": results,
        "total_fees": total_fees,
        "note": "These are example calculations for CME micro futures"
    }