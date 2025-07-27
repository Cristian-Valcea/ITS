# src/agents/enhanced_data_agent.py
"""
Enhanced DataAgent with Survivorship Bias Awareness

This module extends the existing DataAgent with comprehensive survivorship bias elimination.
It serves as a drop-in replacement that provides bias-free data fetching while maintaining
full compatibility with existing workflows.

Key Enhancements:
- Point-in-time universe filtering at data-join level
- CRSP delisting data integration
- Corporate action adjustments
- Bias impact monitoring and reporting
- Production-ready survivorship bias pipeline integration

Usage:
    # Replace existing DataAgent with EnhancedDataAgent
    from src.agents.enhanced_data_agent import EnhancedDataAgent
    
    config = {
        'data_dir_raw': 'data/raw',
        'survivorship_bias_db': 'data/survivorship_bias.db',
        'enable_bias_correction': True,
        'crsp_data_path': 'data/crsp/',
        'ibkr_conn': {...}
    }
    
    data_agent = EnhancedDataAgent(config)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union, Tuple
from pathlib import Path
import logging

# Import base DataAgent
from .data_agent import DataAgent

# Import survivorship bias components
from ..data.survivorship_bias_handler import SurvivorshipBiasHandler, DelistingEvent
from ..data.bias_aware_data_agent import BiasAwareDataAgent
from ..data.crsp_delisting_integration import CRSPDelistingIntegrator
from ..data.production_survivorship_pipeline import ProductionSurvivorshipPipeline, PipelineConfig


class EnhancedDataAgent(DataAgent):
    """
    Enhanced DataAgent with comprehensive survivorship bias elimination.
    
    This class extends the base DataAgent with:
    1. Survivorship bias awareness and correction
    2. CRSP delisting data integration
    3. Point-in-time universe construction
    4. Corporate action handling
    5. Bias impact monitoring and reporting
    
    Maintains full backward compatibility with existing DataAgent interface.
    """
    
    def __init__(self, config: dict, ib_client=None):
        """
        Initialize Enhanced DataAgent with survivorship bias capabilities.
        
        Args:
            config (dict): Configuration dictionary with additional bias-related settings:
                - survivorship_bias_db: Path to survivorship bias database
                - enable_bias_correction: Whether to apply bias correction (default: True)
                - crsp_data_path: Path to CRSP delisting data files
                - auto_load_crsp: Whether to automatically load CRSP data (default: True)
                - bias_monitoring: Whether to enable bias monitoring (default: True)
                - production_pipeline: Whether to start production pipeline (default: False)
            ib_client: Optional IB client instance
        """
        # Initialize base DataAgent
        super().__init__(config, ib_client)
        
        # Survivorship bias configuration
        self.bias_config = {
            'survivorship_bias_db': config.get('survivorship_bias_db', 'data/survivorship_bias.db'),
            'enable_bias_correction': config.get('enable_bias_correction', True),
            'crsp_data_path': config.get('crsp_data_path', 'data/crsp/'),
            'auto_load_crsp': config.get('auto_load_crsp', True),
            'bias_monitoring': config.get('bias_monitoring', True),
            'production_pipeline': config.get('production_pipeline', False),
            'universe_lookback_days': config.get('universe_lookback_days', 252)
        }
        
        # Initialize survivorship bias components
        self._initialize_bias_components()
        
        # Performance tracking
        self.bias_metrics = {
            'queries_with_bias_correction': 0,
            'symbols_filtered_out': 0,
            'total_bias_impact_queries': 0,
            'avg_survival_rate': 0.0
        }
        
        self.logger.info(f"EnhancedDataAgent initialized with bias correction: {self.bias_config['enable_bias_correction']}")
    
    def _initialize_bias_components(self):
        """Initialize survivorship bias handling components."""
        try:
            # Initialize survivorship bias handler
            self.survivorship_handler = SurvivorshipBiasHandler(
                db_path=self.bias_config['survivorship_bias_db'],
                logger=self.logger
            )
            
            # Initialize CRSP integrator
            self.crsp_integrator = CRSPDelistingIntegrator(
                survivorship_handler=self.survivorship_handler,
                logger=self.logger
            )
            
            # Initialize bias-aware data agent
            bias_aware_config = self.config.copy()
            bias_aware_config.update(self.bias_config)
            self.bias_aware_agent = BiasAwareDataAgent(bias_aware_config, self.ib)
            
            # Auto-load CRSP data if configured
            if self.bias_config['auto_load_crsp']:
                self._auto_load_crsp_data()
            
            # Start production pipeline if configured
            if self.bias_config['production_pipeline']:
                self._start_production_pipeline()
            
            self.logger.info("Survivorship bias components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing bias components: {e}")
            # Fallback to non-bias-aware mode
            self.bias_config['enable_bias_correction'] = False
            self.survivorship_handler = None
            self.crsp_integrator = None
            self.bias_aware_agent = None
    
    def _auto_load_crsp_data(self):
        """Automatically load CRSP data if available."""
        try:
            crsp_path = Path(self.bias_config['crsp_data_path'])
            if crsp_path.exists():
                crsp_files = list(crsp_path.glob("*.csv"))
                if crsp_files:
                    self.logger.info(f"Auto-loading CRSP data from {len(crsp_files)} files...")
                    
                    total_events = 0
                    for crsp_file in crsp_files[:5]:  # Limit to first 5 files for auto-load
                        try:
                            load_stats = self.crsp_integrator.load_crsp_delisting_file(str(crsp_file))
                            total_events += load_stats.get('events_created', 0)
                        except Exception as e:
                            self.logger.warning(f"Error loading CRSP file {crsp_file}: {e}")
                    
                    self.logger.info(f"Auto-loaded {total_events} CRSP delisting events")
                else:
                    self.logger.info("No CRSP files found for auto-loading")
            else:
                self.logger.info(f"CRSP data path {crsp_path} does not exist")
                
        except Exception as e:
            self.logger.error(f"Error in auto-loading CRSP data: {e}")
    
    def _start_production_pipeline(self):
        """Start production survivorship bias pipeline."""
        try:
            pipeline_config = PipelineConfig(
                crsp_data_path=self.bias_config['crsp_data_path'],
                database_path=self.bias_config['survivorship_bias_db'],
                crsp_update_schedule="daily",
                realtime_monitoring=True
            )
            
            self.production_pipeline = ProductionSurvivorshipPipeline(pipeline_config)
            self.production_pipeline.start_pipeline()
            
            self.logger.info("Production survivorship bias pipeline started")
            
        except Exception as e:
            self.logger.error(f"Error starting production pipeline: {e}")
            self.production_pipeline = None
    
    def fetch_universe_data(self,
                          symbols: List[str],
                          start_date: str,
                          end_date: str,
                          as_of_date: Optional[str] = None,
                          include_delisted: bool = True,
                          **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for a universe of symbols with survivorship bias correction.
        
        This method extends the base DataAgent functionality with bias-aware data fetching.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data (YYYY-MM-DD format)
            end_date: End date for data (YYYY-MM-DD format)
            as_of_date: Point-in-time date for universe construction (default: end_date)
            include_delisted: Whether to include delisted securities
            **kwargs: Additional arguments passed to data fetching methods
            
        Returns:
            Dictionary mapping symbols to their data DataFrames
        """
        if not self.bias_config['enable_bias_correction'] or not self.bias_aware_agent:
            # Fallback to traditional data fetching
            return self._fetch_universe_data_traditional(symbols, start_date, end_date, **kwargs)
        
        try:
            # Use bias-aware data fetching
            universe_data = self.bias_aware_agent.fetch_universe_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
                include_delisted=include_delisted,
                **kwargs
            )
            
            # Update metrics
            self.bias_metrics['queries_with_bias_correction'] += 1
            self.bias_metrics['symbols_filtered_out'] += len(symbols) - len(universe_data)
            
            # Calculate survival rate for this query
            if len(symbols) > 0:
                survival_rate = len(universe_data) / len(symbols)
                self.bias_metrics['avg_survival_rate'] = (
                    (self.bias_metrics['avg_survival_rate'] * (self.bias_metrics['queries_with_bias_correction'] - 1) + 
                     survival_rate) / self.bias_metrics['queries_with_bias_correction']
                )
            
            self.logger.info(f"Bias-aware data fetch: {len(universe_data)}/{len(symbols)} symbols "
                           f"(survival rate: {len(universe_data)/len(symbols):.1%})")
            
            return universe_data
            
        except Exception as e:
            self.logger.error(f"Error in bias-aware data fetching: {e}")
            # Fallback to traditional method
            return self._fetch_universe_data_traditional(symbols, start_date, end_date, **kwargs)
    
    def _fetch_universe_data_traditional(self,
                                       symbols: List[str],
                                       start_date: str,
                                       end_date: str,
                                       **kwargs) -> Dict[str, pd.DataFrame]:
        """Traditional data fetching without bias correction (fallback method)."""
        universe_data = {}
        
        for symbol in symbols:
            try:
                # Use existing IBKR data fetching
                data = self.fetch_ibkr_bars(
                    symbol=symbol,
                    end_datetime_str=end_date,
                    duration_str=self._calculate_duration_str(start_date, end_date),
                    **kwargs
                )
                
                if data is not None and not data.empty:
                    universe_data[symbol] = data
                    
            except Exception as e:
                self.logger.warning(f"Error fetching data for {symbol}: {e}")
                continue
        
        return universe_data
    
    def _calculate_duration_str(self, start_date: str, end_date: str) -> str:
        """Calculate duration string for IBKR API."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days = (end_dt - start_dt).days
        
        if days <= 30:
            return f"{days} D"
        elif days <= 365:
            return f"{days // 7} W"
        else:
            return f"{days // 365} Y"
    
    def add_delisting_event(self, event: DelistingEvent) -> bool:
        """
        Add a delisting event to the survivorship bias handler.
        
        Args:
            event: DelistingEvent to add
            
        Returns:
            True if added successfully
        """
        if self.survivorship_handler:
            return self.survivorship_handler.add_delisting_event(event)
        else:
            self.logger.warning("Survivorship handler not available")
            return False
    
    def load_crsp_data(self, file_path: str, file_format: str = "csv") -> int:
        """
        Load CRSP delisting data from file.
        
        Args:
            file_path: Path to CRSP data file
            file_format: File format ("csv", "sas", "stata")
            
        Returns:
            Number of delisting events loaded
        """
        if self.crsp_integrator:
            return self.crsp_integrator.load_crsp_delisting_file(file_path, file_format)
        else:
            self.logger.warning("CRSP integrator not available")
            return 0
    
    def get_point_in_time_universe(self,
                                 symbols: Set[str],
                                 as_of_date: datetime,
                                 include_delisted: bool = False) -> Set[str]:
        """
        Get point-in-time universe of active symbols.
        
        Args:
            symbols: Base universe of symbols
            as_of_date: Point-in-time date
            include_delisted: Whether to include recently delisted symbols
            
        Returns:
            Set of active symbols as of the specified date
        """
        if self.survivorship_handler:
            snapshot = self.survivorship_handler.get_point_in_time_universe(
                as_of_date=as_of_date,
                base_universe=symbols,
                lookback_days=self.bias_config['universe_lookback_days']
            )
            
            active_symbols = snapshot.active_symbols
            if include_delisted:
                active_symbols = active_symbols.union(snapshot.recently_delisted)
            
            return active_symbols
        else:
            self.logger.warning("Survivorship handler not available, returning original universe")
            return symbols
    
    def analyze_survivorship_bias(self,
                                symbols: List[str],
                                start_date: str,
                                end_date: str,
                                strategy_func: callable = None) -> Dict:
        """
        Analyze survivorship bias impact for a given universe and strategy.
        
        Args:
            symbols: Universe of symbols to analyze
            start_date: Analysis start date
            end_date: Analysis end date
            strategy_func: Optional strategy function for return calculation
            
        Returns:
            Dictionary with bias analysis results
        """
        if not self.bias_aware_agent:
            return {'error': 'Bias analysis not available'}
        
        try:
            if strategy_func:
                # Compare biased vs unbiased strategy returns
                comparison = self.bias_aware_agent.compare_biased_vs_unbiased_returns(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_func=strategy_func
                )
                return comparison
            else:
                # Generate bias report
                report = self.bias_aware_agent.generate_bias_report(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                return report
                
        except Exception as e:
            self.logger.error(f"Error in bias analysis: {e}")
            return {'error': str(e)}
    
    def get_survivorship_statistics(self) -> Dict:
        """Get comprehensive survivorship bias statistics."""
        stats = {}
        
        # Handler statistics
        if self.survivorship_handler:
            stats['handler_stats'] = self.survivorship_handler.get_statistics()
        
        # CRSP quality report
        if self.crsp_integrator:
            stats['crsp_quality'] = self.crsp_integrator.generate_crsp_quality_report()
        
        # Agent metrics
        stats['agent_metrics'] = self.bias_metrics
        
        # Production pipeline status
        if hasattr(self, 'production_pipeline') and self.production_pipeline:
            stats['pipeline_status'] = self.production_pipeline.get_pipeline_status()
        
        return stats
    
    def generate_bias_impact_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive bias impact report.
        
        Args:
            output_path: Optional path to save report as JSON
            
        Returns:
            Dictionary with comprehensive bias impact analysis
        """
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'configuration': self.bias_config,
                'statistics': self.get_survivorship_statistics(),
                'recommendations': []
            }
            
            # Add recommendations based on statistics
            if 'handler_stats' in report['statistics']:
                total_events = report['statistics']['handler_stats'].get('total_delisting_events', 0)
                if total_events == 0:
                    report['recommendations'].append("No delisting events found. Consider loading CRSP data.")
                elif total_events < 100:
                    report['recommendations'].append("Limited delisting data. Consider loading comprehensive CRSP dataset.")
            
            if self.bias_metrics['avg_survival_rate'] < 0.85:
                report['recommendations'].append("Low average survival rate detected. Bias correction is critical.")
            
            # Save report if path provided
            if output_path:
                import json
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                self.logger.info(f"Bias impact report saved to {output_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating bias impact report: {e}")
            return {'error': str(e)}
    
    def enable_bias_correction(self, enable: bool = True):
        """Enable or disable survivorship bias correction."""
        self.bias_config['enable_bias_correction'] = enable
        self.logger.info(f"Survivorship bias correction {'enabled' if enable else 'disabled'}")
    
    def is_bias_correction_enabled(self) -> bool:
        """Check if survivorship bias correction is enabled."""
        return self.bias_config['enable_bias_correction']
    
    def get_delisting_events(self,
                           symbol: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[DelistingEvent]:
        """
        Get delisting events from the handler.
        
        Args:
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of DelistingEvent objects
        """
        if self.survivorship_handler:
            return self.survivorship_handler.get_delisting_events(symbol, start_date, end_date)
        else:
            return []
    
    def close(self):
        """Close the enhanced data agent and clean up resources."""
        try:
            # Stop production pipeline if running
            if hasattr(self, 'production_pipeline') and self.production_pipeline:
                self.production_pipeline.stop_pipeline()
            
            # Close survivorship handler
            if self.survivorship_handler:
                self.survivorship_handler.close()
            
            # Call parent close method
            super().close()
            
            self.logger.info("EnhancedDataAgent closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing EnhancedDataAgent: {e}")


# Convenience function for easy migration
def create_enhanced_data_agent(config: dict, ib_client=None) -> EnhancedDataAgent:
    """
    Create an EnhancedDataAgent with recommended survivorship bias settings.
    
    Args:
        config: Base configuration dictionary
        ib_client: Optional IB client instance
        
    Returns:
        Configured EnhancedDataAgent instance
    """
    # Add recommended bias correction settings
    enhanced_config = config.copy()
    enhanced_config.setdefault('enable_bias_correction', True)
    enhanced_config.setdefault('auto_load_crsp', True)
    enhanced_config.setdefault('bias_monitoring', True)
    enhanced_config.setdefault('survivorship_bias_db', 'data/survivorship_bias.db')
    enhanced_config.setdefault('crsp_data_path', 'data/crsp/')
    
    return EnhancedDataAgent(enhanced_config, ib_client)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'data_dir_raw': 'data/raw',
        'survivorship_bias_db': 'data/test_survivorship.db',
        'enable_bias_correction': True,
        'auto_load_crsp': True,
        'crsp_data_path': 'data/crsp/',
        'ibkr_conn': {
            'host': '127.0.0.1',
            'port': 7497,
            'clientId': 1
        }
    }
    
    # Create enhanced data agent
    data_agent = create_enhanced_data_agent(config)
    
    # Test bias-aware data fetching
    test_symbols = ["AAPL", "GOOGL", "MSFT", "ENRN", "WCOM"]  # Mix of active and delisted
    
    universe_data = data_agent.fetch_universe_data(
        symbols=test_symbols,
        start_date="2020-01-01",
        end_date="2023-01-01",
        as_of_date="2023-01-01"
    )
    
    print(f"Fetched data for {len(universe_data)}/{len(test_symbols)} symbols")
    
    # Generate bias impact report
    report = data_agent.generate_bias_impact_report("bias_impact_report.json")
    print(f"Bias impact report generated: {len(report)} sections")
    
    # Get statistics
    stats = data_agent.get_survivorship_statistics()
    print(f"Survivorship statistics: {stats}")
    
    # Close agent
    data_agent.close()