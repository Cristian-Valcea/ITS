# src/risk/calculators/concentration_calculator.py
"""
Placeholder for concentration calculator - to be implemented in future phases.
"""

from typing import Dict, Any, List
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class ConcentrationCalculator(VectorizedCalculator):
    """Placeholder concentration calculator."""
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.CONCENTRATION
    
    def _validate_config(self) -> None:
        pass
    
    def get_required_inputs(self) -> List[str]:
        return ['positions']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={'concentration': 0.0},
            is_valid=False,
            error_message="Concentration calculator not implemented yet"
        )