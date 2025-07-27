# src/risk/calculators/stress_test_calculator.py
"""
Stress Test Calculator - Nightly stress testing with scenario analysis.

Implements multiple stress testing methodologies:
- Historical scenario replay (2008, 2020, etc.)
- Monte Carlo stress scenarios
- Factor shock tests
- Tail risk scenarios

Designed for nightly batch processing with comprehensive reporting.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from .base_calculator import BaseRiskCalculator, RiskCalculationResult, RiskMetricType


class StressTestCalculator(BaseRiskCalculator):
    """
    Comprehensive stress test calculator for nightly risk assessment.
    
    Features:
    - Historical scenario replay
    - Monte Carlo stress scenarios
    - Factor shock tests
    - Tail risk analysis
    - Portfolio stress P&L calculation
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize stress test calculator.
        
        Config options:
        - scenarios: List of stress scenarios to run
        - confidence_levels: Confidence levels for stress VaR (default: [0.95, 0.99, 0.999])
        - monte_carlo_runs: Number of MC simulations (default: 10000)
        - factor_shocks: Factor shock magnitudes in standard deviations
        - historical_scenarios: Historical crisis periods to replay
        - tail_threshold: Threshold for tail risk analysis (default: 0.01)
        """
        super().__init__(config, logger)
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.STRESS_TEST
    
    def _validate_config(self) -> None:
        """Validate stress test calculator configuration."""
        # Set defaults
        self.scenarios = self.config.get('scenarios', ['historical', 'monte_carlo', 'factor_shock'])
        self.confidence_levels = self.config.get('confidence_levels', [0.95, 0.99, 0.999])
        self.monte_carlo_runs = self.config.get('monte_carlo_runs', 10000)
        self.factor_shocks = self.config.get('factor_shocks', [1, 2, 3, 5])  # Standard deviations
        self.tail_threshold = self.config.get('tail_threshold', 0.01)
        
        # Historical scenarios configuration
        self.historical_scenarios = self.config.get('historical_scenarios', {
            'financial_crisis_2008': {
                'start_date': '2008-09-01',
                'end_date': '2008-12-31',
                'description': 'Financial Crisis 2008'
            },
            'covid_crash_2020': {
                'start_date': '2020-02-20',
                'end_date': '2020-04-30',
                'description': 'COVID-19 Market Crash 2020'
            },
            'dot_com_crash_2000': {
                'start_date': '2000-03-01',
                'end_date': '2000-12-31',
                'description': 'Dot-com Crash 2000'
            }
        })
        
        # Validate confidence levels
        for level in self.confidence_levels:
            if not (0 < level < 1):
                raise ValueError(f"Confidence level must be between 0 and 1, got {level}")
        
        # Validate monte carlo runs
        if self.monte_carlo_runs <= 0:
            raise ValueError(f"Monte Carlo runs must be positive, got {self.monte_carlo_runs}")
    
    def get_required_inputs(self) -> List[str]:
        """Return required input data keys."""
        return [
            'positions',           # Current portfolio positions
            'returns_history',     # Historical returns data
            'factor_exposures',    # Factor exposures (optional)
            'correlation_matrix',  # Asset correlation matrix (optional)
            'volatilities'         # Asset volatilities
        ]
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Run comprehensive stress tests.
        
        Args:
            data: Dictionary containing:
                - positions: Dict of {symbol: quantity}
                - returns_history: DataFrame with historical returns
                - factor_exposures: Dict of factor exposures (optional)
                - correlation_matrix: Asset correlation matrix (optional)
                - volatilities: Dict of asset volatilities
        
        Returns:
            RiskCalculationResult with stress test results
        """
        positions = data.get('positions', {})
        returns_history = data.get('returns_history', pd.DataFrame())
        factor_exposures = data.get('factor_exposures', {})
        correlation_matrix = data.get('correlation_matrix', None)
        volatilities = data.get('volatilities', {})
        
        if not positions:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={},
                is_valid=False,
                error_message="No positions provided"
            )
        
        stress_results = {}
        
        # Run different stress test scenarios
        if 'historical' in self.scenarios:
            historical_results = self._run_historical_scenarios(
                positions, returns_history
            )
            stress_results.update(historical_results)
        
        if 'monte_carlo' in self.scenarios:
            mc_results = self._run_monte_carlo_stress(
                positions, volatilities, correlation_matrix
            )
            stress_results.update(mc_results)
        
        if 'factor_shock' in self.scenarios:
            factor_results = self._run_factor_shock_tests(
                positions, factor_exposures, volatilities
            )
            stress_results.update(factor_results)
        
        # Calculate stress VaR and Expected Shortfall
        stress_var_results = self._calculate_stress_var(stress_results)
        stress_results.update(stress_var_results)
        
        # Add summary statistics
        stress_results.update({
            'total_scenarios_run': len([k for k in stress_results.keys() if k.endswith('_pnl')]),
            'worst_case_scenario': self._find_worst_case_scenario(stress_results),
            'stress_test_timestamp': datetime.now().isoformat(),
            'portfolio_size': len(positions),
            'total_notional': sum(abs(qty) for qty in positions.values())
        })
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values=stress_results,
            metadata={
                'scenarios_run': self.scenarios,
                'confidence_levels': self.confidence_levels,
                'monte_carlo_runs': self.monte_carlo_runs,
                'historical_scenarios': list(self.historical_scenarios.keys())
            }
        )
    
    def _run_historical_scenarios(self, positions: Dict[str, float], 
                                 returns_history: pd.DataFrame) -> Dict[str, Any]:
        """Run historical scenario stress tests."""
        results = {}
        
        if returns_history.empty:
            self.logger.warning("No historical returns data for historical scenarios")
            return results
        
        for scenario_id, scenario_config in self.historical_scenarios.items():
            try:
                # Filter returns for scenario period
                start_date = pd.to_datetime(scenario_config['start_date'])
                end_date = pd.to_datetime(scenario_config['end_date'])
                
                if hasattr(returns_history, 'index') and hasattr(returns_history.index, 'to_pydatetime'):
                    scenario_returns = returns_history[
                        (returns_history.index >= start_date) & 
                        (returns_history.index <= end_date)
                    ]
                else:
                    # Fallback if index is not datetime
                    scenario_returns = returns_history
                
                if scenario_returns.empty:
                    self.logger.warning(f"No data for scenario {scenario_id}")
                    continue
                
                # Calculate portfolio P&L for each day in scenario
                scenario_pnl = []
                for date, returns_row in scenario_returns.iterrows():
                    daily_pnl = 0.0
                    for symbol, quantity in positions.items():
                        if symbol in returns_row:
                            daily_pnl += quantity * returns_row[symbol]
                    scenario_pnl.append(daily_pnl)
                
                if scenario_pnl:
                    results[f'{scenario_id}_pnl'] = scenario_pnl
                    results[f'{scenario_id}_total_pnl'] = sum(scenario_pnl)
                    results[f'{scenario_id}_worst_day'] = min(scenario_pnl)
                    results[f'{scenario_id}_best_day'] = max(scenario_pnl)
                    results[f'{scenario_id}_volatility'] = np.std(scenario_pnl)
                    results[f'{scenario_id}_description'] = scenario_config['description']
                
            except Exception as e:
                self.logger.error(f"Error in historical scenario {scenario_id}: {e}")
                continue
        
        return results
    
    def _run_monte_carlo_stress(self, positions: Dict[str, float],
                               volatilities: Dict[str, float],
                               correlation_matrix: Optional[np.ndarray]) -> Dict[str, Any]:
        """Run Monte Carlo stress test scenarios."""
        results = {}
        
        if not volatilities:
            self.logger.warning("No volatilities provided for Monte Carlo stress")
            return results
        
        symbols = list(positions.keys())
        position_values = np.array([positions[symbol] for symbol in symbols])
        vols = np.array([volatilities.get(symbol, 0.2) for symbol in symbols])  # Default 20% vol
        
        # Use correlation matrix if provided, otherwise assume independence
        if correlation_matrix is not None and correlation_matrix.shape[0] == len(symbols):
            cov_matrix = np.outer(vols, vols) * correlation_matrix
        else:
            cov_matrix = np.diag(vols ** 2)
        
        # Generate Monte Carlo scenarios
        try:
            # Generate random returns
            random_returns = multivariate_normal.rvs(
                mean=np.zeros(len(symbols)),
                cov=cov_matrix,
                size=self.monte_carlo_runs
            )
            
            # Calculate portfolio P&L for each scenario
            portfolio_pnl = np.dot(random_returns, position_values)
            
            results['monte_carlo_pnl'] = portfolio_pnl.tolist()
            results['monte_carlo_mean'] = float(np.mean(portfolio_pnl))
            results['monte_carlo_std'] = float(np.std(portfolio_pnl))
            results['monte_carlo_min'] = float(np.min(portfolio_pnl))
            results['monte_carlo_max'] = float(np.max(portfolio_pnl))
            
            # Percentile analysis
            for percentile in [1, 5, 10, 90, 95, 99]:
                results[f'monte_carlo_p{percentile}'] = float(np.percentile(portfolio_pnl, percentile))
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo stress test: {e}")
        
        return results
    
    def _run_factor_shock_tests(self, positions: Dict[str, float],
                               factor_exposures: Dict[str, Dict[str, float]],
                               volatilities: Dict[str, float]) -> Dict[str, Any]:
        """Run factor shock stress tests."""
        results = {}
        
        if not factor_exposures:
            self.logger.warning("No factor exposures provided for factor shock tests")
            return results
        
        # Common factors to shock
        factors = ['market', 'size', 'value', 'momentum', 'quality', 'volatility']
        
        for factor in factors:
            for shock_magnitude in self.factor_shocks:
                scenario_name = f'factor_shock_{factor}_{shock_magnitude}std'
                
                total_pnl = 0.0
                for symbol, quantity in positions.items():
                    if symbol in factor_exposures:
                        exposure = factor_exposures[symbol].get(factor, 0.0)
                        vol = volatilities.get(symbol, 0.2)
                        
                        # Calculate P&L from factor shock
                        factor_pnl = quantity * exposure * shock_magnitude * vol
                        total_pnl += factor_pnl
                
                results[scenario_name] = float(total_pnl)
        
        return results
    
    def _calculate_stress_var(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate stress VaR from all scenario results."""
        var_results = {}
        
        # Collect all P&L scenarios
        all_pnl = []
        
        # Add historical scenario P&Ls
        for key, value in stress_results.items():
            if key.endswith('_pnl') and isinstance(value, list):
                all_pnl.extend(value)
            elif key.endswith('_total_pnl') and isinstance(value, (int, float)):
                all_pnl.append(value)
            elif key.startswith('factor_shock_') and isinstance(value, (int, float)):
                all_pnl.append(value)
        
        if not all_pnl:
            return var_results
        
        all_pnl = np.array(all_pnl)
        
        # Calculate stress VaR at different confidence levels
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            var_percentile = np.percentile(all_pnl, alpha * 100)
            
            confidence_pct = int(confidence_level * 1000) / 10  # Handle 99.9% format
            var_results[f'stress_var_{confidence_pct}'] = float(-var_percentile)  # Convert to loss
        
        # Calculate Expected Shortfall (Conditional VaR)
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            var_threshold = np.percentile(all_pnl, alpha * 100)
            tail_losses = all_pnl[all_pnl <= var_threshold]
            
            if len(tail_losses) > 0:
                expected_shortfall = -np.mean(tail_losses)  # Convert to loss
                confidence_pct = int(confidence_level * 1000) / 10
                var_results[f'stress_es_{confidence_pct}'] = float(expected_shortfall)
        
        # Additional stress statistics
        var_results.update({
            'stress_scenarios_count': len(all_pnl),
            'stress_worst_case': float(-np.min(all_pnl)),
            'stress_best_case': float(-np.max(all_pnl)),
            'stress_mean': float(-np.mean(all_pnl)),
            'stress_std': float(np.std(all_pnl))
        })
        
        return var_results
    
    def _find_worst_case_scenario(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the worst-case scenario from all stress tests."""
        worst_case = {
            'scenario_name': 'none',
            'pnl': 0.0,
            'description': 'No scenarios run'
        }
        
        worst_pnl = float('inf')
        
        # Check historical scenarios
        for key, value in stress_results.items():
            if key.endswith('_total_pnl') and isinstance(value, (int, float)):
                if value < worst_pnl:
                    worst_pnl = value
                    scenario_name = key.replace('_total_pnl', '')
                    worst_case = {
                        'scenario_name': scenario_name,
                        'pnl': float(value),
                        'description': stress_results.get(f'{scenario_name}_description', scenario_name)
                    }
            
            # Check factor shocks
            elif key.startswith('factor_shock_') and isinstance(value, (int, float)):
                if value < worst_pnl:
                    worst_pnl = value
                    worst_case = {
                        'scenario_name': key,
                        'pnl': float(value),
                        'description': f'Factor shock scenario: {key}'
                    }
        
        # Check Monte Carlo worst case
        mc_min = stress_results.get('monte_carlo_min')
        if mc_min is not None and mc_min < worst_pnl:
            worst_case = {
                'scenario_name': 'monte_carlo_worst',
                'pnl': float(mc_min),
                'description': 'Monte Carlo worst-case scenario'
            }
        
        return worst_case


__all__ = ['StressTestCalculator']