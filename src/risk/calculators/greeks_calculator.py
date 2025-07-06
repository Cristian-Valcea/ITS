# src/risk/calculators/greeks_calculator.py
"""
Options Greeks Calculator
Calculates option sensitivities (Delta, Gamma, Vega, Theta, Rho) using Black-Scholes model
with microsecond-level latency for real-time risk management.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Any, List, Optional, Union
import logging
from .base_calculator import BaseRiskCalculator, RiskCalculationResult, RiskMetricType


class GreeksCalculator(BaseRiskCalculator):
    """
    Options Greeks calculator using Black-Scholes model.
    
    Calculates:
    - Delta: Price sensitivity to underlying price changes
    - Gamma: Delta sensitivity to underlying price changes  
    - Vega: Price sensitivity to volatility changes
    - Theta: Price sensitivity to time decay
    - Rho: Price sensitivity to interest rate changes
    
    Supports both individual options and portfolio-level aggregation.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize Greeks calculator.
        
        Config options:
        - risk_free_rate: Risk-free interest rate (default: 0.05)
        - dividend_yield: Dividend yield (default: 0.0)
        - time_convention: Days in year for time calculations (default: 365)
        - min_time_to_expiry: Minimum time to expiry in days (default: 1)
        - volatility_floor: Minimum volatility (default: 0.01)
        - volatility_cap: Maximum volatility (default: 5.0)
        """
        super().__init__(config, logger)
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.GREEKS
    
    def _validate_config(self) -> None:
        """Validate Greeks calculator configuration."""
        self.risk_free_rate = self.config.get('risk_free_rate', 0.05)
        self.dividend_yield = self.config.get('dividend_yield', 0.0)
        self.time_convention = self.config.get('time_convention', 365)
        self.min_time_to_expiry = self.config.get('min_time_to_expiry', 1)
        self.volatility_floor = self.config.get('volatility_floor', 0.01)
        self.volatility_cap = self.config.get('volatility_cap', 5.0)
        
        # Validate parameters
        if self.risk_free_rate < 0:
            raise ValueError(f"Risk-free rate cannot be negative: {self.risk_free_rate}")
        
        if self.dividend_yield < 0:
            raise ValueError(f"Dividend yield cannot be negative: {self.dividend_yield}")
        
        if self.time_convention <= 0:
            raise ValueError(f"Time convention must be positive: {self.time_convention}")
        
        if self.volatility_floor <= 0 or self.volatility_floor >= self.volatility_cap:
            raise ValueError(f"Invalid volatility bounds: floor={self.volatility_floor}, cap={self.volatility_cap}")
    
    def get_required_inputs(self) -> List[str]:
        """Return required input data keys."""
        return [
            'spot_price',      # Current underlying price
            'strike_price',    # Option strike price
            'time_to_expiry',  # Time to expiry in days
            'volatility',      # Implied volatility
            'option_type',     # 'call' or 'put'
            'position_size'    # Number of contracts (can be negative for short)
        ]
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate option Greeks.
        
        Args:
            data: Dictionary containing option parameters:
                - spot_price: Current underlying price (float or array)
                - strike_price: Strike price (float or array)
                - time_to_expiry: Time to expiry in days (float or array)
                - volatility: Implied volatility (float or array)
                - option_type: 'call' or 'put' (str or array)
                - position_size: Position size in contracts (float or array)
                - risk_free_rate: Override default risk-free rate (optional)
                - dividend_yield: Override default dividend yield (optional)
        
        Returns:
            RiskCalculationResult with Greeks metrics
        """
        # Extract parameters
        spot_price = np.atleast_1d(data['spot_price'])
        strike_price = np.atleast_1d(data['strike_price'])
        time_to_expiry = np.atleast_1d(data['time_to_expiry'])
        volatility = np.atleast_1d(data['volatility'])
        option_type = np.atleast_1d(data['option_type'])
        position_size = np.atleast_1d(data['position_size'])
        
        # Override rates if provided
        risk_free_rate = data.get('risk_free_rate', self.risk_free_rate)
        dividend_yield = data.get('dividend_yield', self.dividend_yield)
        
        # Validate array shapes
        max_len = max(len(spot_price), len(strike_price), len(time_to_expiry), 
                     len(volatility), len(option_type), len(position_size))
        
        # Broadcast arrays to same length
        spot_price = self._broadcast_array(spot_price, max_len)
        strike_price = self._broadcast_array(strike_price, max_len)
        time_to_expiry = self._broadcast_array(time_to_expiry, max_len)
        volatility = self._broadcast_array(volatility, max_len)
        option_type = self._broadcast_array(option_type, max_len)
        position_size = self._broadcast_array(position_size, max_len)
        
        # Validate and clean inputs
        time_to_expiry = np.maximum(time_to_expiry, self.min_time_to_expiry)
        volatility = np.clip(volatility, self.volatility_floor, self.volatility_cap)
        
        # Convert time to years
        time_to_expiry_years = time_to_expiry / self.time_convention
        
        # Calculate Greeks for each option
        results = {
            'delta': np.zeros(max_len),
            'gamma': np.zeros(max_len),
            'vega': np.zeros(max_len),
            'theta': np.zeros(max_len),
            'rho': np.zeros(max_len),
            'option_value': np.zeros(max_len)
        }
        
        for i in range(max_len):
            greeks = self._calculate_single_option_greeks(
                spot_price[i], strike_price[i], time_to_expiry_years[i],
                volatility[i], risk_free_rate, dividend_yield, option_type[i]
            )
            
            # Scale by position size
            scale = position_size[i]
            results['delta'][i] = greeks['delta'] * scale
            results['gamma'][i] = greeks['gamma'] * scale
            results['vega'][i] = greeks['vega'] * scale
            results['theta'][i] = greeks['theta'] * scale
            results['rho'][i] = greeks['rho'] * scale
            results['option_value'][i] = greeks['option_value'] * scale
        
        # Portfolio-level aggregation
        portfolio_results = {
            'portfolio_delta': float(np.sum(results['delta'])),
            'portfolio_gamma': float(np.sum(results['gamma'])),
            'portfolio_vega': float(np.sum(results['vega'])),
            'portfolio_theta': float(np.sum(results['theta'])),
            'portfolio_rho': float(np.sum(results['rho'])),
            'portfolio_value': float(np.sum(results['option_value'])),
            'net_delta_exposure': float(np.sum(results['delta'] * spot_price)),
            'gamma_risk': float(np.sum(np.abs(results['gamma']) * spot_price**2 * 0.01**2)),  # 1% move
            'vega_risk': float(np.sum(np.abs(results['vega']) * 0.01)),  # 1% vol move
        }
        
        # Individual option results (if single option)
        if max_len == 1:
            portfolio_results.update({
                'delta': float(results['delta'][0]),
                'gamma': float(results['gamma'][0]),
                'vega': float(results['vega'][0]),
                'theta': float(results['theta'][0]),
                'rho': float(results['rho'][0]),
                'option_value': float(results['option_value'][0])
            })
        else:
            # Array results for multiple options
            portfolio_results.update({
                'individual_deltas': results['delta'].tolist(),
                'individual_gammas': results['gamma'].tolist(),
                'individual_vegas': results['vega'].tolist(),
                'individual_thetas': results['theta'].tolist(),
                'individual_rhos': results['rho'].tolist(),
                'individual_values': results['option_value'].tolist()
            })
        
        # Risk metrics
        portfolio_results.update({
            'delta_neutral': abs(portfolio_results['portfolio_delta']) < 0.01,
            'gamma_exposure_pct': portfolio_results['gamma_risk'] / max(portfolio_results['portfolio_value'], 1) * 100,
            'vega_exposure_pct': portfolio_results['vega_risk'] / max(portfolio_results['portfolio_value'], 1) * 100,
            'options_count': max_len
        })
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values=portfolio_results,
            metadata={
                'risk_free_rate': risk_free_rate,
                'dividend_yield': dividend_yield,
                'time_convention': self.time_convention,
                'options_count': max_len,
                'vectorized': True
            }
        )
    
    def _broadcast_array(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Broadcast array to target length."""
        if len(arr) == 1:
            return np.full(target_len, arr[0])
        elif len(arr) == target_len:
            return arr
        else:
            raise ValueError(f"Array length {len(arr)} cannot be broadcast to {target_len}")
    
    def _calculate_single_option_greeks(self, S: float, K: float, T: float, 
                                      sigma: float, r: float, q: float, 
                                      option_type: str) -> Dict[str, float]:
        """
        Calculate Greeks for a single option using Black-Scholes model.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            sigma: Volatility
            r: Risk-free rate
            q: Dividend yield
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with Greeks values
        """
        if T <= 0:
            # Handle expired options
            if option_type.lower() == 'call':
                intrinsic = max(S - K, 0)
            else:
                intrinsic = max(K - S, 0)
            
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0,
                'option_value': intrinsic
            }
        
        # Black-Scholes parameters
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)  # PDF
        
        # Discount factors
        exp_rT = np.exp(-r * T)
        exp_qT = np.exp(-q * T)
        
        if option_type.lower() == 'call':
            # Call option
            option_value = S * exp_qT * N_d1 - K * exp_rT * N_d2
            delta = exp_qT * N_d1
            rho = K * T * exp_rT * N_d2
        else:
            # Put option
            option_value = K * exp_rT * norm.cdf(-d2) - S * exp_qT * norm.cdf(-d1)
            delta = exp_qT * (N_d1 - 1)
            rho = -K * T * exp_rT * norm.cdf(-d2)
        
        # Greeks (same for calls and puts)
        gamma = exp_qT * n_d1 / (S * sigma * sqrt_T)
        vega = S * exp_qT * n_d1 * sqrt_T / 100  # Per 1% vol change
        theta = (
            -S * exp_qT * n_d1 * sigma / (2 * sqrt_T) 
            - r * K * exp_rT * N_d2 
            + q * S * exp_qT * N_d1
        ) / 365  # Per day
        
        if option_type.lower() == 'put':
            theta += r * K * exp_rT - q * S * exp_qT
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho / 100,  # Per 1% rate change
            'option_value': option_value
        }