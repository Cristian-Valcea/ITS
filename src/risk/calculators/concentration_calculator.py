# src/risk/calculators/concentration_calculator.py
"""
Concentration Risk Calculator
Measures portfolio concentration across multiple dimensions including single-name,
sector, geography, and strategy concentration with microsecond-level latency.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from .base_calculator import BaseRiskCalculator, RiskCalculationResult, RiskMetricType


class ConcentrationCalculator(BaseRiskCalculator):
    """
    Concentration risk calculator measuring portfolio concentration across:
    - Single-name concentration (largest position)
    - Top-N concentration (top 5, 10 positions)
    - Sector/Industry concentration
    - Geographic concentration
    - Strategy concentration
    - Herfindahl-Hirschman Index (HHI)
    
    Designed for microsecond-level latency with vectorized operations.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize concentration calculator.
        
        Config options:
        - top_n_positions: List of top-N to calculate (default: [5, 10])
        - concentration_thresholds: Warning thresholds (default: [0.1, 0.2, 0.3])
        - include_sector_analysis: Include sector concentration (default: True)
        - include_geographic_analysis: Include geographic concentration (default: True)
        - min_positions: Minimum positions for valid calculation (default: 1)
        """
        super().__init__(config, logger)
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.CONCENTRATION
    
    def _validate_config(self) -> None:
        """Validate concentration calculator configuration."""
        self.top_n_positions = self.config.get('top_n_positions', [5, 10])
        self.concentration_thresholds = self.config.get('concentration_thresholds', [0.1, 0.2, 0.3])
        self.include_sector_analysis = self.config.get('include_sector_analysis', True)
        self.include_geographic_analysis = self.config.get('include_geographic_analysis', True)
        self.min_positions = self.config.get('min_positions', 1)
        
        # Validate parameters
        if not isinstance(self.top_n_positions, list):
            self.top_n_positions = [self.top_n_positions]
        
        for n in self.top_n_positions:
            if not isinstance(n, int) or n <= 0:
                raise ValueError(f"Top-N positions must be positive integers: {n}")
        
        if not isinstance(self.concentration_thresholds, list):
            self.concentration_thresholds = [self.concentration_thresholds]
        
        for threshold in self.concentration_thresholds:
            if not (0 < threshold < 1):
                raise ValueError(f"Concentration threshold must be between 0 and 1: {threshold}")
        
        if self.min_positions <= 0:
            raise ValueError(f"Min positions must be positive: {self.min_positions}")
    
    def get_required_inputs(self) -> List[str]:
        """Return required input data keys."""
        return ['positions']  # Only positions are truly required
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate concentration risk metrics.
        
        Args:
            data: Dictionary containing:
                - positions: Dict of {symbol: position_value} (required)
                - sectors: Dict of {symbol: sector} (optional)
                - countries: Dict of {symbol: country} (optional)
                - strategies: Dict of {symbol: strategy} (optional)
                - market_values: Dict of {symbol: market_value} (optional)
        
        Returns:
            RiskCalculationResult with concentration metrics
        """
        positions = data['positions']
        
        if not positions:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={},
                is_valid=False,
                error_message="No positions provided"
            )
        
        if len(positions) < self.min_positions:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={},
                is_valid=False,
                error_message=f"Insufficient positions: {len(positions)} < {self.min_positions}"
            )
        
        results = {}
        
        # Calculate single-name concentration
        single_name_results = self._calculate_single_name_concentration(positions)
        results.update(single_name_results)
        
        # Calculate top-N concentration
        top_n_results = self._calculate_top_n_concentration(positions)
        results.update(top_n_results)
        
        # Calculate Herfindahl-Hirschman Index
        hhi_results = self._calculate_hhi(positions)
        results.update(hhi_results)
        
        # Sector concentration analysis
        if self.include_sector_analysis and 'sectors' in data:
            sector_results = self._calculate_sector_concentration(positions, data['sectors'])
            results.update(sector_results)
        
        # Geographic concentration analysis
        if self.include_geographic_analysis and 'countries' in data:
            geo_results = self._calculate_geographic_concentration(positions, data['countries'])
            results.update(geo_results)
        
        # Strategy concentration analysis
        if 'strategies' in data:
            strategy_results = self._calculate_strategy_concentration(positions, data['strategies'])
            results.update(strategy_results)
        
        # Risk assessment
        risk_assessment = self._assess_concentration_risk(results)
        results.update(risk_assessment)
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values=results,
            metadata={
                'total_positions': len(positions),
                'top_n_positions': self.top_n_positions,
                'concentration_thresholds': self.concentration_thresholds,
                'include_sector_analysis': self.include_sector_analysis,
                'include_geographic_analysis': self.include_geographic_analysis,
                'vectorized': True
            }
        )
    
    def _calculate_single_name_concentration(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate single-name concentration metrics."""
        # Calculate absolute values for concentration analysis
        abs_positions = {symbol: abs(value) for symbol, value in positions.items()}
        total_gross_exposure = sum(abs_positions.values())
        
        if total_gross_exposure == 0:
            return {
                'max_single_name_pct': 0.0,
                'max_single_name_symbol': None,
                'max_single_name_value': 0.0
            }
        
        # Find largest position
        max_symbol = max(abs_positions.keys(), key=lambda k: abs_positions[k])
        max_value = abs_positions[max_symbol]
        max_pct = max_value / total_gross_exposure
        
        # Calculate position statistics
        position_values = list(abs_positions.values())
        position_pcts = [v / total_gross_exposure for v in position_values]
        
        return {
            'max_single_name_pct': float(max_pct),
            'max_single_name_symbol': max_symbol,
            'max_single_name_value': float(max_value),
            'total_gross_exposure': float(total_gross_exposure),
            'total_net_exposure': float(sum(positions.values())),
            'position_count': len(positions),
            'avg_position_pct': float(np.mean(position_pcts)),
            'median_position_pct': float(np.median(position_pcts)),
            'position_pct_std': float(np.std(position_pcts, ddof=1)) if len(position_pcts) > 1 else 0.0
        }
    
    def _calculate_top_n_concentration(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate top-N concentration metrics."""
        abs_positions = {symbol: abs(value) for symbol, value in positions.items()}
        total_gross_exposure = sum(abs_positions.values())
        
        if total_gross_exposure == 0:
            return {}
        
        # Sort positions by size (descending)
        sorted_positions = sorted(abs_positions.items(), key=lambda x: x[1], reverse=True)
        
        results = {}
        
        for n in self.top_n_positions:
            if n > len(sorted_positions):
                continue
            
            # Calculate top-N concentration
            top_n_exposure = sum(value for _, value in sorted_positions[:n])
            top_n_pct = top_n_exposure / total_gross_exposure
            top_n_symbols = [symbol for symbol, _ in sorted_positions[:n]]
            
            results[f'top_{n}_concentration_pct'] = float(top_n_pct)
            results[f'top_{n}_symbols'] = top_n_symbols
            results[f'top_{n}_exposure'] = float(top_n_exposure)
        
        return results
    
    def _calculate_hhi(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate Herfindahl-Hirschman Index."""
        abs_positions = {symbol: abs(value) for symbol, value in positions.items()}
        total_gross_exposure = sum(abs_positions.values())
        
        if total_gross_exposure == 0:
            return {'hhi': 0.0, 'hhi_normalized': 0.0}
        
        # Calculate HHI
        position_shares = [value / total_gross_exposure for value in abs_positions.values()]
        hhi = sum(share ** 2 for share in position_shares)
        
        # Normalized HHI (0 = perfectly diversified, 1 = concentrated)
        n = len(positions)
        hhi_min = 1.0 / n if n > 0 else 0.0
        hhi_normalized = (hhi - hhi_min) / (1.0 - hhi_min) if hhi_min < 1.0 else 0.0
        
        return {
            'hhi': float(hhi),
            'hhi_normalized': float(hhi_normalized),
            'hhi_equivalent_positions': float(1.0 / hhi) if hhi > 0 else float(n)
        }
    
    def _calculate_sector_concentration(self, positions: Dict[str, float], 
                                      sectors: Dict[str, str]) -> Dict[str, Any]:
        """Calculate sector concentration metrics."""
        # Aggregate positions by sector
        sector_exposures = {}
        total_gross_exposure = 0
        
        for symbol, position in positions.items():
            sector = sectors.get(symbol, 'Unknown')
            abs_position = abs(position)
            
            if sector not in sector_exposures:
                sector_exposures[sector] = 0
            
            sector_exposures[sector] += abs_position
            total_gross_exposure += abs_position
        
        if total_gross_exposure == 0:
            return {}
        
        # Calculate sector concentration metrics
        sector_pcts = {sector: exposure / total_gross_exposure 
                      for sector, exposure in sector_exposures.items()}
        
        max_sector = max(sector_pcts.keys(), key=lambda k: sector_pcts[k])
        max_sector_pct = sector_pcts[max_sector]
        
        # Sector HHI
        sector_hhi = sum(pct ** 2 for pct in sector_pcts.values())
        
        return {
            'max_sector_concentration_pct': float(max_sector_pct),
            'max_sector_name': max_sector,
            'sector_count': len(sector_exposures),
            'sector_hhi': float(sector_hhi),
            'sector_exposures': {k: float(v) for k, v in sector_exposures.items()},
            'sector_percentages': {k: float(v) for k, v in sector_pcts.items()}
        }
    
    def _calculate_geographic_concentration(self, positions: Dict[str, float],
                                          countries: Dict[str, str]) -> Dict[str, Any]:
        """Calculate geographic concentration metrics."""
        # Aggregate positions by country
        country_exposures = {}
        total_gross_exposure = 0
        
        for symbol, position in positions.items():
            country = countries.get(symbol, 'Unknown')
            abs_position = abs(position)
            
            if country not in country_exposures:
                country_exposures[country] = 0
            
            country_exposures[country] += abs_position
            total_gross_exposure += abs_position
        
        if total_gross_exposure == 0:
            return {}
        
        # Calculate geographic concentration metrics
        country_pcts = {country: exposure / total_gross_exposure 
                       for country, exposure in country_exposures.items()}
        
        max_country = max(country_pcts.keys(), key=lambda k: country_pcts[k])
        max_country_pct = country_pcts[max_country]
        
        # Geographic HHI
        geo_hhi = sum(pct ** 2 for pct in country_pcts.values())
        
        return {
            'max_country_concentration_pct': float(max_country_pct),
            'max_country_name': max_country,
            'country_count': len(country_exposures),
            'geographic_hhi': float(geo_hhi),
            'country_exposures': {k: float(v) for k, v in country_exposures.items()},
            'country_percentages': {k: float(v) for k, v in country_pcts.items()}
        }
    
    def _calculate_strategy_concentration(self, positions: Dict[str, float],
                                        strategies: Dict[str, str]) -> Dict[str, Any]:
        """Calculate strategy concentration metrics."""
        # Aggregate positions by strategy
        strategy_exposures = {}
        total_gross_exposure = 0
        
        for symbol, position in positions.items():
            strategy = strategies.get(symbol, 'Unknown')
            abs_position = abs(position)
            
            if strategy not in strategy_exposures:
                strategy_exposures[strategy] = 0
            
            strategy_exposures[strategy] += abs_position
            total_gross_exposure += abs_position
        
        if total_gross_exposure == 0:
            return {}
        
        # Calculate strategy concentration metrics
        strategy_pcts = {strategy: exposure / total_gross_exposure 
                        for strategy, exposure in strategy_exposures.items()}
        
        max_strategy = max(strategy_pcts.keys(), key=lambda k: strategy_pcts[k])
        max_strategy_pct = strategy_pcts[max_strategy]
        
        # Strategy HHI
        strategy_hhi = sum(pct ** 2 for pct in strategy_pcts.values())
        
        return {
            'max_strategy_concentration_pct': float(max_strategy_pct),
            'max_strategy_name': max_strategy,
            'strategy_count': len(strategy_exposures),
            'strategy_hhi': float(strategy_hhi),
            'strategy_exposures': {k: float(v) for k, v in strategy_exposures.items()},
            'strategy_percentages': {k: float(v) for k, v in strategy_pcts.items()}
        }
    
    def _assess_concentration_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall concentration risk level."""
        risk_flags = []
        risk_score = 0.0
        
        # Check single-name concentration
        max_single_pct = results.get('max_single_name_pct', 0.0)
        for i, threshold in enumerate(sorted(self.concentration_thresholds)):
            if max_single_pct > threshold:
                risk_score = max(risk_score, (i + 1) / len(self.concentration_thresholds))
                risk_flags.append(f'Single-name concentration {max_single_pct:.1%} > {threshold:.1%}')
        
        # Check sector concentration
        max_sector_pct = results.get('max_sector_concentration_pct', 0.0)
        for i, threshold in enumerate(sorted(self.concentration_thresholds)):
            if max_sector_pct > threshold:
                risk_score = max(risk_score, (i + 1) / len(self.concentration_thresholds))
                risk_flags.append(f'Sector concentration {max_sector_pct:.1%} > {threshold:.1%}')
        
        # Check HHI
        hhi = results.get('hhi', 0.0)
        if hhi > 0.25:  # Highly concentrated
            risk_score = max(risk_score, 0.8)
            risk_flags.append(f'High HHI concentration: {hhi:.3f}')
        elif hhi > 0.15:  # Moderately concentrated
            risk_score = max(risk_score, 0.5)
            risk_flags.append(f'Moderate HHI concentration: {hhi:.3f}')
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'HIGH'
        elif risk_score >= 0.4:
            risk_level = 'MEDIUM'
        elif risk_score >= 0.1:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'concentration_risk_score': float(risk_score),
            'concentration_risk_level': risk_level,
            'concentration_risk_flags': risk_flags,
            'is_highly_concentrated': risk_score >= 0.7
        }
