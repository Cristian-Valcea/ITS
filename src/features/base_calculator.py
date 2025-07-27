# src/features/base_calculator.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional


class BaseFeatureCalculator(ABC):
    """
    Abstract base class for all feature calculators.
    Each feature calculator is responsible for computing specific technical indicators
    or features from market data.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the feature calculator.
        
        Args:
            config: Configuration dictionary containing parameters for the calculator
            logger: Logger instance for logging messages
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._max_lookback = 0
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features from the input data.
        
        Args:
            data: DataFrame containing OHLCV data with DatetimeIndex
            
        Returns:
            DataFrame with original data plus calculated features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features that this calculator produces.
        
        Returns:
            List of feature column names
        """
        pass
    
    @abstractmethod
    def get_max_lookback(self) -> int:
        """
        Get the maximum lookback period required by this calculator.
        
        Returns:
            Maximum number of periods needed for stable calculation
        """
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that the input data has required columns and proper index.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.error("Input data is None or empty")
            return False
            
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error("Data must have a DatetimeIndex")
            return False
            
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        return True
    
    def handle_calculation_error(self, error: Exception, feature_name: str, 
                                data_index: pd.Index) -> pd.Series:
        """
        Handle calculation errors by returning a Series of NaNs.
        
        Args:
            error: The exception that occurred
            feature_name: Name of the feature being calculated
            data_index: Index for the output Series
            
        Returns:
            Series filled with NaNs
        """
        self.logger.error(f"Error calculating {feature_name}: {error}", exc_info=True)
        return pd.Series(np.nan, index=data_index, name=feature_name)