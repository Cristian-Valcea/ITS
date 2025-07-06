# src/risk/sensors/tail_regime.py
"""
Tail & Regime-Shift Sensors - "What if tomorrow is nothing like today?"

These sensors detect when market conditions are shifting beyond normal parameters,
indicating potential regime changes or tail risk events.

Sensors:
1. ExpectedShortfallSensor - CVaR on intraday P/L, captures severity beyond VaR
2. VolOfVolSensor - Volatility of volatility, early warning of vol surface changes
3. RegimeSwitchSensor - HMM log-likelihood ratio, detects distribution changes
"""

import numpy as np
from typing import Dict, Any, List
from scipy import stats
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

from .base_sensor import BaseSensor, FailureMode, SensorPriority


class ExpectedShortfallSensor(BaseSensor):
    """
    Expected Shortfall (CVaR) Sensor - Captures tail risk severity beyond VaR.
    
    Expected Shortfall measures the average loss in the worst-case scenarios,
    providing a more comprehensive view of tail risk than VaR alone.
    
    Formula: ES_α = E[L | L > VaR_α] where L is loss
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.TAIL_REGIME_SHIFT
    
    def _get_data_requirements(self) -> List[str]:
        return ['returns', 'timestamp']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute Expected Shortfall (CVaR) from returns."""
        returns = np.array(data['returns'])
        
        if len(returns) < 10:
            return 0.0
        
        # Convert returns to losses (negative returns)
        losses = -returns
        
        # Calculate VaR at specified confidence level
        confidence_level = float(self.config.get('confidence_level', 0.95))
        var_quantile = np.quantile(losses, confidence_level)
        
        # Calculate Expected Shortfall (average of losses beyond VaR)
        tail_losses = losses[losses >= var_quantile]
        
        if len(tail_losses) == 0:
            return var_quantile  # If no tail losses, return VaR
        
        expected_shortfall = np.mean(tail_losses)
        
        return expected_shortfall
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on sample size and distribution stability."""
        base_confidence = super()._compute_confidence(value, data)
        
        returns = np.array(data['returns'])
        
        # Higher confidence with more data points
        if len(returns) >= 100:
            sample_bonus = 0.2
        elif len(returns) >= 50:
            sample_bonus = 0.1
        else:
            sample_bonus = 0.0
        
        # Check for distribution stability using Kolmogorov-Smirnov test
        if len(returns) >= 20:
            mid_point = len(returns) // 2
            first_half = returns[:mid_point]
            second_half = returns[mid_point:]
            
            try:
                ks_stat, p_value = stats.ks_2samp(first_half, second_half)
                if p_value < 0.05:  # Distributions are significantly different
                    stability_penalty = 0.3
                else:
                    stability_penalty = 0.0
            except:
                stability_penalty = 0.1
        else:
            stability_penalty = 0.0
        
        confidence = base_confidence + sample_bonus - stability_penalty
        return max(0.0, min(1.0, confidence))


class VolOfVolSensor(BaseSensor):
    """
    Volatility of Volatility Sensor - Early warning of vol surface changes.
    
    This sensor monitors the volatility of volatility, which tends to spike
    before major market dislocations. It uses realized volatility and
    implied volatility ratios when available.
    
    Formula: VolOfVol = std(realized_vol_series) / mean(realized_vol_series)
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.TAIL_REGIME_SHIFT
    
    def _get_data_requirements(self) -> List[str]:
        return ['returns', 'timestamp']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute volatility of volatility."""
        returns = np.array(data['returns'])
        
        if len(returns) < 20:
            return 0.0
        
        # Calculate rolling realized volatility
        vol_window = int(self.config.get('vol_window', 10))
        vol_series = []
        
        for i in range(vol_window, len(returns)):
            window_returns = returns[i-vol_window:i]
            realized_vol = np.std(window_returns) * np.sqrt(252)  # Annualized
            vol_series.append(realized_vol)
        
        vol_series = np.array(vol_series)
        
        if len(vol_series) < 5:
            return 0.0
        
        # Calculate volatility of volatility
        mean_vol = np.mean(vol_series)
        if mean_vol == 0:
            return 0.0
        
        vol_of_vol = np.std(vol_series) / mean_vol
        
        # If implied volatility is available, incorporate IV/RV ratio
        implied_vol = data.get('implied_volatility')
        if implied_vol is not None and len(implied_vol) >= len(vol_series):
            iv_series = np.array(implied_vol[-len(vol_series):])
            iv_rv_ratio = iv_series / vol_series
            iv_rv_vol = np.std(iv_rv_ratio)
            
            # Combine realized vol-of-vol with IV/RV volatility
            vol_of_vol = 0.7 * vol_of_vol + 0.3 * iv_rv_vol
        
        return vol_of_vol
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data quality and market conditions."""
        base_confidence = super()._compute_confidence(value, data)
        
        returns = np.array(data['returns'])
        
        # Higher confidence with more data
        if len(returns) >= 100:
            data_bonus = 0.2
        elif len(returns) >= 50:
            data_bonus = 0.1
        else:
            data_bonus = 0.0
        
        # Bonus if implied volatility data is available
        implied_vol = data.get('implied_volatility')
        if implied_vol is not None:
            iv_bonus = 0.1
        else:
            iv_bonus = 0.0
        
        confidence = base_confidence + data_bonus + iv_bonus
        return max(0.0, min(1.0, confidence))


class RegimeSwitchSensor(BaseSensor):
    """
    Regime Switch Sensor - Detects when return distribution changes.
    
    This sensor uses a Hidden Markov Model approach to detect when the
    current return distribution no longer fits the "normal" regime,
    indicating a potential regime shift.
    
    Formula: log_likelihood_ratio = log(P(data|current_regime)) - log(P(data|normal_regime))
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.TAIL_REGIME_SHIFT
    
    def _get_data_requirements(self) -> List[str]:
        return ['returns', 'timestamp']
    
    def __init__(self, sensor_id: str, sensor_name: str, config: Dict[str, Any]):
        super().__init__(sensor_id, sensor_name, config)
        
        # Initialize regime models
        self.normal_regime_model = None
        self.current_regime_model = None
        self.calibration_window = int(config.get('calibration_window', 100))
        self.detection_window = int(config.get('detection_window', 20))
        
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute regime switch likelihood ratio."""
        returns = np.array(data['returns'])
        
        if len(returns) < self.calibration_window + self.detection_window:
            return 0.0
        
        # Split data into calibration and detection windows
        calibration_data = returns[:-self.detection_window]
        detection_data = returns[-self.detection_window:]
        
        try:
            # Fit normal regime model on historical data
            if self.normal_regime_model is None or len(calibration_data) > len(self._calibration_data):
                self.normal_regime_model = self._fit_regime_model(calibration_data)
            
            # Fit current regime model on recent data
            self.current_regime_model = self._fit_regime_model(detection_data)
            
            # Calculate log-likelihood for detection data under both models
            normal_ll = self._calculate_log_likelihood(detection_data, self.normal_regime_model)
            current_ll = self._calculate_log_likelihood(detection_data, self.current_regime_model)
            
            # Log-likelihood ratio (higher values indicate regime shift)
            ll_ratio = current_ll - normal_ll
            
            # Normalize by detection window size
            normalized_ll_ratio = ll_ratio / len(detection_data)
            
            return abs(normalized_ll_ratio)
            
        except Exception as e:
            self.logger.debug(f"Regime detection failed: {e}")
            return 0.0
    
    def _fit_regime_model(self, returns: np.ndarray) -> Dict[str, float]:
        """Fit a simple Gaussian model to returns."""
        if len(returns) < 5:
            return {'mean': 0.0, 'std': 1.0}
        
        # Try to fit a 2-component Gaussian mixture model
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(returns.reshape(-1, 1))
            
            # Use the component with higher weight
            dominant_component = np.argmax(gmm.weights_)
            mean = gmm.means_[dominant_component][0]
            std = np.sqrt(gmm.covariances_[dominant_component][0][0])
            
            return {'mean': mean, 'std': std}
            
        except:
            # Fallback to simple Gaussian
            return {'mean': np.mean(returns), 'std': np.std(returns)}
    
    def _calculate_log_likelihood(self, returns: np.ndarray, model: Dict[str, float]) -> float:
        """Calculate log-likelihood of returns under a Gaussian model."""
        mean = model['mean']
        std = max(model['std'], 1e-6)  # Avoid division by zero
        
        # Log-likelihood of Gaussian distribution
        ll = -0.5 * len(returns) * np.log(2 * np.pi * std**2)
        ll -= 0.5 * np.sum((returns - mean)**2) / (std**2)
        
        return ll
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on model fit quality."""
        base_confidence = super()._compute_confidence(value, data)
        
        returns = np.array(data['returns'])
        
        # Higher confidence with more calibration data
        if len(returns) >= 200:
            calibration_bonus = 0.2
        elif len(returns) >= 100:
            calibration_bonus = 0.1
        else:
            calibration_bonus = 0.0
        
        # Check model stability
        if self.normal_regime_model is not None:
            model_std = self.normal_regime_model['std']
            if model_std > 0 and model_std < 1.0:  # Reasonable volatility
                stability_bonus = 0.1
            else:
                stability_bonus = 0.0
        else:
            stability_bonus = 0.0
        
        confidence = base_confidence + calibration_bonus + stability_bonus
        return max(0.0, min(1.0, confidence))
    
    def _format_message(self, value: float, threshold: float, action) -> str:
        """Format message with regime context."""
        if self.normal_regime_model and self.current_regime_model:
            normal_vol = self.normal_regime_model['std']
            current_vol = self.current_regime_model['std']
            vol_ratio = current_vol / normal_vol if normal_vol > 0 else 1.0
            
            return (f"{self.sensor_name}: {value:.4f} (threshold: {threshold:.4f}) "
                    f"[Vol Ratio: {vol_ratio:.2f}] → {action.value}")
        else:
            return super()._format_message(value, threshold, action)