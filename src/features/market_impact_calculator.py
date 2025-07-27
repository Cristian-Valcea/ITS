"""
Market Impact Feature Calculator

Calculates market microstructure features including spread, queue imbalance,
market impact, and Kyle's lambda for training observations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_calculator import BaseFeatureCalculator
from ..shared.market_impact import calc_market_impact_features, ImpactFeatures


class MarketImpactCalculator(BaseFeatureCalculator):
    """
    Calculator for market impact and microstructure features.
    
    Features computed:
    - spread_bps: Bid-ask spread in basis points
    - queue_imbalance: Order book imbalance [-1, +1]
    - impact_10k: Market impact for 10k USD notional
    - kyle_lambda: Kyle's lambda (price impact per unit volume)
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize MarketImpactCalculator.
        
        Args:
            config: Configuration dictionary with keys:
                - notional_amount: Amount for impact calculation (default: 10000)
                - enable_kyle_lambda: Whether to calculate Kyle's lambda (default: True)
                - min_spread_bps: Minimum spread threshold (default: 0.1)
            logger: Logger instance
        """
        super().__init__(config=config, logger=logger)
        self.name = "market_impact"
        self.required_columns = ["bid_px1", "bid_sz1", "ask_px1", "ask_sz1"]
        
        # Configuration parameters
        self.notional_amount = self.config.get('notional_amount', 10_000)
        self.enable_kyle_lambda = self.config.get('enable_kyle_lambda', True)
        self.min_spread_bps = self.config.get('min_spread_bps', 0.1)
        
        # Feature columns that will be added
        self.feature_columns = [
            'spread_bps',
            'queue_imbalance', 
            'impact_10k',
            'kyle_lambda'
        ]
        
        # State for Kyle's lambda calculation
        self._last_mid = None
        self._volume_history = []
        self._price_history = []
        
        self.logger.info(f"MarketImpactCalculator initialized with notional_amount={self.notional_amount}")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market impact features for the given DataFrame.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            DataFrame with added market impact features
        """
        if df.empty:
            return df
        
        try:
            # Validate required columns
            if not self._validate_required_columns(df):
                self.logger.warning("Missing required order book columns, using default values")
                return self._add_default_features(df)
            
            # Calculate mid price if not present
            if 'mid' not in df.columns:
                df = df.copy()
                df['mid'] = (df['bid_px1'] + df['ask_px1']) / 2
            
            # Initialize feature columns
            result_df = df.copy()
            for col in self.feature_columns:
                result_df[col] = np.nan
            
            # Calculate features for each row
            for i in range(len(df)):
                try:
                    features = self._calculate_row_features(df.iloc[i], i)
                    
                    # Assign features to result DataFrame
                    for feature_name, value in features.items():
                        if feature_name in self.feature_columns:
                            result_df.loc[df.index[i], feature_name] = value
                            
                except Exception as e:
                    self.logger.warning(f"Error calculating features for row {i}: {e}")
                    # Keep NaN values for this row
                    continue
            
            # Fill NaN values with defaults
            result_df = self._fill_nan_values(result_df)
            
            # Validate output
            if not self._validate_output(result_df):
                self.logger.error("Output validation failed")
                return df  # Return original DataFrame if validation fails
            
            self.logger.debug(f"Calculated market impact features for {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in MarketImpactCalculator.calculate: {e}")
            return self._add_default_features(df)
    
    def _calculate_row_features(self, row: pd.Series, row_index: int) -> ImpactFeatures:
        """Calculate features for a single row."""
        try:
            mid = row['mid']
            
            # Get previous mid for Kyle's lambda
            last_mid = self._last_mid if self.enable_kyle_lambda else None
            
            # For signed volume, we'd need trade data - using None for now
            # In a real implementation, this would come from trade ticks
            signed_vol = None
            
            # Calculate features using the shared module
            features = calc_market_impact_features(
                book=row,
                mid=mid,
                last_mid=last_mid,
                signed_vol=signed_vol,
                notional=self.notional_amount
            )
            
            # Update state for next iteration
            self._last_mid = mid
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error calculating row features: {e}")
            return ImpactFeatures(
                spread_bps=0.0,
                queue_imbalance=0.0,
                impact_10k=0.0,
                kyle_lambda=np.nan
            )
    
    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """Validate that required order book columns are present."""
        required_cols = ["bid_px1", "bid_sz1", "ask_px1", "ask_sz1"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")
            return False
        
        return True
    
    def _add_default_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default feature values when calculation fails."""
        result_df = df.copy()
        
        result_df['spread_bps'] = 0.0
        result_df['queue_imbalance'] = 0.0
        result_df['impact_10k'] = 0.0
        result_df['kyle_lambda'] = np.nan
        
        return result_df
    
    def _fill_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with appropriate defaults."""
        result_df = df.copy()
        
        # Fill numeric features with 0.0
        numeric_features = ['spread_bps', 'queue_imbalance', 'impact_10k']
        for feature in numeric_features:
            if feature in result_df.columns:
                result_df[feature] = result_df[feature].fillna(0.0)
        
        # Kyle's lambda can remain NaN (it's expected when no historical data)
        
        return result_df
    
    def _validate_output(self, df: pd.DataFrame) -> bool:
        """Validate the output DataFrame."""
        try:
            # Check that all feature columns are present
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                self.logger.error(f"Missing output features: {missing_features}")
                return False
            
            # Check for reasonable ranges
            if 'spread_bps' in df.columns:
                spread_values = df['spread_bps'].dropna()
                if len(spread_values) > 0:
                    if (spread_values < 0).any():
                        self.logger.warning("Negative spread values detected")
                    if (spread_values > 1000).any():  # 10% spread seems excessive
                        self.logger.warning("Very large spread values detected")
            
            if 'queue_imbalance' in df.columns:
                imbalance_values = df['queue_imbalance'].dropna()
                if len(imbalance_values) > 0:
                    if (imbalance_values < -1.1).any() or (imbalance_values > 1.1).any():
                        self.logger.warning("Queue imbalance values outside expected range [-1, 1]")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating output: {e}")
            return False
    
    def get_lookback_window(self) -> int:
        """
        Return the lookback window required for this calculator.
        
        Market impact features are calculated per-row, so no lookback needed.
        However, Kyle's lambda benefits from historical data.
        """
        return 1  # Minimal lookback for current row calculation
    
    def get_max_lookback(self) -> int:
        """
        Get the maximum lookback period required by this calculator.
        
        Returns:
            Maximum number of periods needed for stable calculation
        """
        return 1  # Market impact features are calculated per-row
    
    def get_feature_names(self) -> list:
        """Return list of feature names this calculator produces."""
        return self.feature_columns.copy()
    
    def reset_state(self):
        """Reset internal state (useful for live trading sessions)."""
        self._last_mid = None
        self._volume_history.clear()
        self._price_history.clear()
        self.logger.debug("MarketImpactCalculator state reset")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for this calculator."""
        return {
            'notional_amount': {
                'type': 'number',
                'default': 10_000,
                'description': 'Notional amount for market impact calculation (USD)'
            },
            'enable_kyle_lambda': {
                'type': 'boolean', 
                'default': True,
                'description': 'Whether to calculate Kyle\'s lambda'
            },
            'min_spread_bps': {
                'type': 'number',
                'default': 0.1,
                'description': 'Minimum spread threshold in basis points'
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return performance statistics for this calculator."""
        return {
            'calculator_name': self.name,
            'feature_count': len(self.feature_columns),
            'lookback_window': self.get_lookback_window(),
            'notional_amount': self.notional_amount,
            'kyle_lambda_enabled': self.enable_kyle_lambda
        }