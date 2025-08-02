#!/usr/bin/env python3
"""
Deflated Sharpe Ratio Implementation
Based on Bailey & López de Prado (2016) for institutional model validation

Addresses reviewer concern about statistical rigor in performance evaluation.
The Deflated Sharpe Ratio (DSR) adjusts for multiple testing bias and 
selection bias inherent in quantitative strategy development.

Key Features:
- Multiple testing correction
- Selection bias adjustment  
- Non-normal return distribution handling
- Statistical significance testing
- P-value calculation against null hypothesis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass 
class DeflatedSharpeResult:
    """Results from deflated Sharpe ratio calculation"""
    sharpe_ratio: float
    deflated_sharpe_ratio: float
    p_value: float
    is_significant: bool
    confidence_level: float
    num_trials: int
    expected_max_sharpe: float
    variance_max_sharpe: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'deflated_sharpe_ratio': self.deflated_sharpe_ratio,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'confidence_level': self.confidence_level,
            'num_trials': self.num_trials,
            'expected_max_sharpe': self.expected_max_sharpe,
            'variance_max_sharpe': self.variance_max_sharpe
        }


class DeflatedSharpeCalculator:
    """
    Calculates Deflated Sharpe Ratio following Bailey & López de Prado (2016)
    
    The DSR adjusts the Sharpe ratio for:
    1. Multiple testing (trying many strategies)
    2. Non-normal return distributions
    3. Selection bias from choosing best backtest
    
    Formula:
    DSR = (SR - E[max SR]) / sqrt(Var[max SR])
    
    Where:
    - SR is the strategy Sharpe ratio
    - E[max SR] is expected maximum Sharpe from N random trials
    - Var[max SR] is variance of maximum Sharpe from N random trials
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.critical_value = stats.norm.ppf(confidence_level)
        
    def calculate_deflated_sharpe(self, 
                                  returns: np.ndarray,
                                  num_trials: int,
                                  benchmark_vol: Optional[float] = None,
                                  skewness: Optional[float] = None,
                                  kurtosis: Optional[float] = None) -> DeflatedSharpeResult:
        """
        Calculate deflated Sharpe ratio for a return series
        
        Args:
            returns: Array of period returns
            num_trials: Number of strategies tested (for multiple testing correction)
            benchmark_vol: Benchmark volatility (if different from risk-free rate)
            skewness: Return distribution skewness (calculated if None)
            kurtosis: Return distribution kurtosis (calculated if None)
            
        Returns:
            DeflatedSharpeResult with all statistics
        """
        
        # Input validation
        if len(returns) < 30:
            raise ValueError("Minimum 30 observations required for reliable estimation")
        
        if num_trials < 1:
            raise ValueError("Number of trials must be positive")
        
        # Calculate basic statistics
        returns = np.asarray(returns)
        mean_return = np.mean(returns)
        vol_return = np.std(returns, ddof=1)
        
        # Handle zero volatility
        if vol_return == 0:
            logger.warning("Zero volatility detected, returning NaN DSR")
            return DeflatedSharpeResult(
                sharpe_ratio=np.nan,
                deflated_sharpe_ratio=np.nan,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                num_trials=num_trials,
                expected_max_sharpe=np.nan,
                variance_max_sharpe=np.nan
            )
        
        # Calculate Sharpe ratio (annualized if needed)
        sharpe_ratio = mean_return / vol_return
        
        # Calculate higher moments if not provided
        if skewness is None:
            skewness = stats.skew(returns)
        if kurtosis is None:
            kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        # Calculate expected maximum Sharpe ratio and its variance
        expected_max_sharpe, variance_max_sharpe = self._calculate_max_sharpe_moments(
            num_trials, len(returns), skewness, kurtosis
        )
        
        # Calculate deflated Sharpe ratio
        if variance_max_sharpe > 0:
            deflated_sharpe = (sharpe_ratio - expected_max_sharpe) / np.sqrt(variance_max_sharpe)
        else:
            deflated_sharpe = np.nan
        
        # Calculate p-value (probability that observed Sharpe is due to chance)
        p_value = 1 - stats.norm.cdf(deflated_sharpe)
        
        # Determine statistical significance
        is_significant = deflated_sharpe > self.critical_value
        
        logger.info(f"📊 Deflated Sharpe Analysis:")
        logger.info(f"   Original Sharpe: {sharpe_ratio:.4f}")
        logger.info(f"   Deflated Sharpe: {deflated_sharpe:.4f}")
        logger.info(f"   P-value: {p_value:.4f}")
        logger.info(f"   Significant: {is_significant} (α={1-self.confidence_level:.2f})")
        
        return DeflatedSharpeResult(
            sharpe_ratio=sharpe_ratio,
            deflated_sharpe_ratio=deflated_sharpe,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            num_trials=num_trials,
            expected_max_sharpe=expected_max_sharpe,
            variance_max_sharpe=variance_max_sharpe
        )
    
    def _calculate_max_sharpe_moments(self, 
                                      num_trials: int, 
                                      num_observations: int,
                                      skewness: float, 
                                      kurtosis: float) -> Tuple[float, float]:
        """
        Calculate expected value and variance of maximum Sharpe ratio
        from N independent trials with non-normal distributions
        
        Uses the extreme value theory results from Bailey & López de Prado (2016)
        """
        
        # Euler-Mascheroni constant
        euler_gamma = 0.5772156649015329
        
        # Expected maximum Sharpe ratio
        # E[max SR] ≈ √(2 log N) - (log log N + log 4π) / (2√(2 log N))
        log_n = np.log(num_trials)
        sqrt_2_log_n = np.sqrt(2 * log_n)
        
        expected_max_sharpe = sqrt_2_log_n - (np.log(np.log(num_trials)) + np.log(4 * np.pi)) / (2 * sqrt_2_log_n)
        
        # Variance of maximum Sharpe ratio  
        # Var[max SR] ≈ π² / (6 × 2 log N)
        variance_max_sharpe = (np.pi ** 2) / (6 * 2 * log_n)
        
        # Adjust for non-normal distributions (higher moments correction)
        if abs(skewness) > 0.1 or abs(kurtosis) > 0.5:
            # Cornish-Fisher expansion adjustment
            cf_adjustment = self._cornish_fisher_adjustment(skewness, kurtosis, num_observations)
            expected_max_sharpe += cf_adjustment
            
            # Increase variance for non-normal distributions
            non_normal_factor = 1 + abs(skewness) / 6 + kurtosis / 24
            variance_max_sharpe *= non_normal_factor
        
        return expected_max_sharpe, variance_max_sharpe
    
    def _cornish_fisher_adjustment(self, skewness: float, kurtosis: float, 
                                   num_observations: int) -> float:
        """
        Cornish-Fisher expansion adjustment for non-normal distributions
        """
        
        # Standard adjustment terms
        cf_adjustment = (skewness / 6) * (2 * np.log(num_observations) - 1)
        cf_adjustment += (kurtosis / 24) * (np.log(num_observations) - 1)
        cf_adjustment -= (skewness ** 2 / 36) * (np.log(num_observations))
        
        return cf_adjustment
    
    def batch_calculate_cross_validation_dsr(self, 
                                             cv_results: Dict[str, np.ndarray],
                                             total_strategies_tested: int) -> Dict[str, DeflatedSharpeResult]:
        """
        Calculate DSR for cross-validation results across multiple folds
        
        Args:
            cv_results: Dict mapping fold names to return arrays
            total_strategies_tested: Total number of strategies/hyperparameters tested
            
        Returns:
            Dict mapping fold names to DSR results
        """
        
        results = {}
        
        for fold_name, returns in cv_results.items():
            try:
                dsr_result = self.calculate_deflated_sharpe(
                    returns=returns,
                    num_trials=total_strategies_tested
                )
                results[fold_name] = dsr_result
                
            except Exception as e:
                logger.error(f"DSR calculation failed for fold {fold_name}: {e}")
                results[fold_name] = None
        
        # Calculate ensemble statistics
        valid_results = [r for r in results.values() if r is not None]
        
        if valid_results:
            ensemble_sharpe = np.mean([r.sharpe_ratio for r in valid_results])
            ensemble_dsr = np.mean([r.deflated_sharpe_ratio for r in valid_results])
            min_p_value = min([r.p_value for r in valid_results])
            
            logger.info(f"📈 Cross-Validation DSR Summary:")
            logger.info(f"   Ensemble Sharpe: {ensemble_sharpe:.4f}")
            logger.info(f"   Ensemble DSR: {ensemble_dsr:.4f}")
            logger.info(f"   Min P-value: {min_p_value:.4f}")
            logger.info(f"   Folds significant: {sum(r.is_significant for r in valid_results)}/{len(valid_results)}")
        
        return results
    
    def calculate_minimum_track_record_length(self, 
                                              target_sharpe: float,
                                              num_trials: int,
                                              confidence_level: float = 0.95) -> int:
        """
        Calculate minimum track record length required to achieve
        statistical significance for a given Sharpe ratio
        
        This addresses the question: "How long must I run this strategy
        to be confident it's not due to chance?"
        """
        
        critical_value = stats.norm.ppf(confidence_level)
        
        # Iterative search for minimum length
        for n in range(30, 10000):  # Start at 30 minimum observations
            expected_max, variance_max = self._calculate_max_sharpe_moments(
                num_trials, n, skewness=0, kurtosis=0  # Assume normal for planning
            )
            
            # Required DSR for significance
            required_dsr = critical_value
            
            # Implied minimum Sharpe needed
            min_sharpe_needed = expected_max + required_dsr * np.sqrt(variance_max)
            
            if target_sharpe >= min_sharpe_needed:
                logger.info(f"📏 Minimum Track Record Analysis:")
                logger.info(f"   Target Sharpe: {target_sharpe:.4f}")
                logger.info(f"   Minimum observations: {n}")
                logger.info(f"   Minimum months (daily data): {n/21:.1f}")
                logger.info(f"   Confidence level: {confidence_level:.1%}")
                return n
        
        logger.warning("Could not find feasible track record length within 10,000 observations")
        return -1


# Utility functions for common use cases
def evaluate_strategy_significance(returns: pd.Series, 
                                   num_backtests_run: int,
                                   confidence_level: float = 0.95) -> DeflatedSharpeResult:
    """
    Convenient wrapper for strategy evaluation
    
    Args:
        returns: Pandas series of returns
        num_backtests_run: Number of strategies/parameters tested
        confidence_level: Statistical confidence level
        
    Returns:
        DSR result indicating if strategy is statistically significant
    """
    
    calculator = DeflatedSharpeCalculator(confidence_level)
    
    return calculator.calculate_deflated_sharpe(
        returns=returns.values,
        num_trials=num_backtests_run
    )


def batch_evaluate_walk_forward_cv(cv_results_dict: Dict[str, pd.Series],
                                   total_trials: int) -> Dict[str, DeflatedSharpeResult]:
    """
    Evaluate walk-forward cross-validation results with DSR
    
    Args:
        cv_results_dict: Dict mapping CV fold names to return series
        total_trials: Total number of strategies tested
        
    Returns:
        Dict mapping fold names to DSR results
    """
    
    calculator = DeflatedSharpeCalculator()
    
    # Convert to numpy arrays
    cv_arrays = {
        fold: returns.values 
        for fold, returns in cv_results_dict.items()
    }
    
    return calculator.batch_calculate_cross_validation_dsr(cv_arrays, total_trials)


def main():
    """Example usage and testing"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Generate example data
    np.random.seed(42)
    
    # Simulate a strategy with modest positive Sharpe
    n_periods = 252  # 1 year daily
    true_sharpe = 1.2
    returns = np.random.normal(true_sharpe / np.sqrt(252), 1 / np.sqrt(252), n_periods)
    
    # Add some skewness and kurtosis
    returns[50:60] *= 2  # Fat tails
    returns[100:105] *= -1.5  # Negative skew
    
    # Evaluate with different numbers of trials
    calculator = DeflatedSharpeCalculator()
    
    for num_trials in [1, 10, 100, 1000]:
        result = calculator.calculate_deflated_sharpe(
            returns=returns,
            num_trials=num_trials
        )
        
        print(f"\nTrials: {num_trials}")
        print(f"  Sharpe: {result.sharpe_ratio:.4f}")
        print(f"  DSR: {result.deflated_sharpe_ratio:.4f}")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Significant: {result.is_significant}")
    
    # Calculate minimum track record needed
    min_length = calculator.calculate_minimum_track_record_length(
        target_sharpe=1.0,
        num_trials=100,
        confidence_level=0.95
    )
    print(f"\nMinimum track record for Sharpe 1.0: {min_length} observations")


if __name__ == "__main__":
    main()