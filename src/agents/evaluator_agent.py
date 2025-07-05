import os
import logging
from typing import Optional, Dict, Any, Tuple, Union
import pandas as pd

from .base_agent import BaseAgent
from src.gym_env.intraday_trading_env import IntradayTradingEnv
from src.evaluation import MetricsCalculator, ModelLoader, BacktestRunner, ReportGenerator

class EvaluatorAgent(BaseAgent):
    """
    EvaluatorAgent is responsible for:
    1. Running backtests of a trained model on a held-out (evaluation) dataset.
    2. Computing performance metrics: Sharpe ratio, max drawdown, total return, turnover.
    3. Generating evaluation reports and logging full trade history from the backtest.
    
    This agent now uses specialized components for better maintainability:
    - MetricsCalculator: Handles all performance metric calculations
    - ModelLoader: Manages model loading and validation
    - BacktestRunner: Executes backtests and manages environment interaction
    - ReportGenerator: Creates and saves evaluation reports
    """
    
    def __init__(self, config: dict, eval_env: Optional[IntradayTradingEnv] = None):
        """
        Initialize the EvaluatorAgent with specialized components.
        
        Args:
            config: Configuration dictionary
            eval_env: Optional evaluation environment
        """
        super().__init__(agent_name="EvaluatorAgent", config=config)
        
        # Initialize specialized components
        self.metrics_calculator = MetricsCalculator(config)
        self.model_loader = ModelLoader(config)
        self.backtest_runner = BacktestRunner(config)
        self.report_generator = ReportGenerator(config)
        
        # Environment management
        self.evaluation_env = eval_env
        
        # Configuration
        self.default_model_load_path = config.get('model_load_path', None)
        
        self.logger.info("EvaluatorAgent initialized with specialized components.")
        self.logger.info(f"Evaluation reports will be saved to: {self.report_generator.reports_dir}")
        self.logger.info(f"Metrics to calculate: {self.metrics_calculator.eval_metrics_config}")

    def set_env(self, env: IntradayTradingEnv) -> None:
        """
        Set the evaluation environment.
        
        Args:
            env: IntradayTradingEnv instance for evaluation
            
        Raises:
            ValueError: If env is not an IntradayTradingEnv instance
        """
        if not isinstance(env, IntradayTradingEnv):
            self.logger.error("Invalid environment type provided to EvaluatorAgent.")
            raise ValueError("Environment must be an instance of IntradayTradingEnv.")
        
        self.evaluation_env = env
        self.logger.info("Evaluation environment set for EvaluatorAgent.")

    def load_model(self, model_path: str, algorithm_name: str = "DQN") -> bool:
        """
        Load a model for evaluation using the ModelLoader component.
        
        Args:
            model_path: Path to the model file
            algorithm_name: Algorithm name (e.g., "DQN")
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        return self.model_loader.load_model(model_path, algorithm_name, env_context=self)

    def run_backtest(self, deterministic: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Run a backtest using the BacktestRunner component.
        
        Args:
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (trade_log_df, portfolio_history_series) or (None, None) if failed
        """
        if not self.model_loader.is_model_loaded():
            self.logger.error("No model loaded. Cannot run backtest.")
            return None, None
            
        if self.evaluation_env is None:
            self.logger.error("Evaluation environment not set. Cannot run backtest.")
            return None, None

        model = self.model_loader.get_loaded_model()
        return self.backtest_runner.run_backtest(model, self.evaluation_env, deterministic)

    def calculate_metrics(
        self, 
        trade_log_df: pd.DataFrame, 
        portfolio_history: pd.Series, 
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics using the MetricsCalculator component.
        
        Args:
            trade_log_df: DataFrame containing trade log
            portfolio_history: Series of portfolio values over time
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary containing calculated metrics
        """
        return self.metrics_calculator.calculate_metrics(trade_log_df, portfolio_history, initial_capital)

    def generate_report(
        self, 
        metrics: Dict[str, Any], 
        trade_log_df: Optional[pd.DataFrame], 
        model_name: str
    ) -> str:
        """
        Generate evaluation report using the ReportGenerator component.
        
        Args:
            metrics: Dictionary of calculated performance metrics
            trade_log_df: DataFrame containing trade log
            model_name: Name/identifier for the model
            
        Returns:
            Path to the saved text report
        """
        # Add model information to the report
        model_info = self.model_loader.get_model_info()
        additional_info = {
            'model_type': model_info.get('type', 'Unknown'),
            'model_class': model_info.get('class', 'Unknown'),
            'backtest_steps': len(trade_log_df) if trade_log_df is not None else 0
        }
        
        return self.report_generator.generate_report(metrics, trade_log_df, model_name, additional_info)

    def run(
        self,
        eval_env: IntradayTradingEnv,
        model_path: Optional[str] = None,
        algorithm_name: str = "DQN",
        model_name_tag: str = "model"
    ) -> Optional[Dict[str, Any]]:
        """
        Main method for EvaluatorAgent: orchestrates the complete evaluation process.
        
        Args:
            eval_env: Environment for evaluation
            model_path: Path to model file (optional, uses default if not provided)
            algorithm_name: Algorithm name for model loading
            model_name_tag: Tag for naming reports and logs
            
        Returns:
            Dictionary of performance metrics, or None if evaluation failed
        """
        self.logger.info(f"EvaluatorAgent run started for model: {model_name_tag}")
        self.logger.info(f"Model path: {model_path or self.default_model_load_path}")
        
        try:
            # Set evaluation environment
            self.set_env(eval_env)
            
            # Determine model path
            model_to_load = model_path or self.default_model_load_path
            if not model_to_load:
                self.logger.error("No model path provided or configured. Cannot run evaluation.")
                return None

            # Load model
            if not self.load_model(model_to_load, algorithm_name):
                self.logger.error(f"Failed to load model {model_to_load}. Evaluation aborted.")
                return None

            # Run backtest
            trade_log_df, portfolio_history = self.run_backtest(deterministic=True)
            
            # Calculate metrics
            initial_capital = getattr(self.evaluation_env, 'initial_capital', 0.0)
            metrics = self.calculate_metrics(
                trade_log_df if trade_log_df is not None else pd.DataFrame(),
                portfolio_history if portfolio_history is not None else pd.Series(dtype=float),
                initial_capital
            )
            
            # Generate report
            report_path = self.generate_report(metrics, trade_log_df, model_name_tag)
            
            self.logger.info(f"Evaluation for {model_name_tag} complete.")
            self.logger.info(f"Report saved to: {report_path}")
            self.logger.info(f"Final metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Exception during evaluation run: {e}", exc_info=True)
            return None
    
    def get_component_status(self) -> Dict[str, Any]:
        """
        Get status information about all components.
        
        Returns:
            Dictionary containing component status information
        """
        return {
            'model_loader': self.model_loader.get_model_info(),
            'evaluation_env_set': self.evaluation_env is not None,
            'reports_directory': self.report_generator.reports_dir,
            'configured_metrics': self.metrics_calculator.eval_metrics_config
        }


def main():
    """
    Main function for standalone debugging and testing of EvaluatorAgent.
    
    This allows running the evaluator independently for testing purposes.
    Uses dummy configuration initialization similar to orchestrator_agent.py.
    """
    import sys
    import yaml
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import required modules
    try:
        from src.gym_env.intraday_trading_env import IntradayTradingEnv
        from src.agents.base_agent import BaseAgent
        from src.evaluation import MetricsCalculator, ModelLoader, BacktestRunner, ReportGenerator
    except ImportError:
        # Handle relative imports when running as script
        import sys
        sys.path.append(str(project_root))
        from src.gym_env.intraday_trading_env import IntradayTradingEnv
        from src.agents.base_agent import BaseAgent
        from src.evaluation import MetricsCalculator, ModelLoader, BacktestRunner, ReportGenerator
    
    import yfinance as yf
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("EvaluatorAgent Standalone Debug Mode")
    print("=" * 70)
    
    # Create debug configuration directories
    CONFIG_DIR = "config_evaluator_debug"
    REPORTS_DIR = "reports_evaluator_debug"
    MODELS_DIR = "models_evaluator_debug"
    
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Debug parameters - easily modifiable for different test scenarios
    DEBUG_PARAMS = {
        'symbol': 'AAPL',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'interval': '1h',
        'initial_capital': 10000.0,
        'algorithm': 'DQN',
        'model_name': 'debug_evaluator_model',
        'use_dummy_model': True,  # Set to False if you have a real model to test
        'real_model_path': None   # Set path here if use_dummy_model is False
    }
    
    # Create dummy evaluator configuration
    evaluator_config_path = os.path.join(CONFIG_DIR, "evaluator_debug_config.yaml")
    dummy_evaluator_config = {
        'reports_dir': REPORTS_DIR,
        'eval_metrics': [
            'total_return_pct', 
            'sharpe_ratio', 
            'max_drawdown_pct', 
            'num_trades', 
            'turnover_ratio_period',
            'win_rate_pct'
        ],
        'model_load_path': None,  # Will be set dynamically
        'risk_free_rate_annual': 0.02,  # 2% annual risk-free rate
        'periods_per_year_for_sharpe': 252,
        'backtest_progress_log_interval': 25,
        'environment_config': {
            'transaction_cost': 0.001,  # 0.1% transaction cost
            'max_position_size': 1.0,   # 100% of capital
            'lookback_window': 20       # 20-period lookback
        }
    }
    
    with open(evaluator_config_path, 'w') as f:
        yaml.dump(dummy_evaluator_config, f)
    
    logging.info(f"Debug config created: {evaluator_config_path}")
    
    # Display debug parameters
    print("Debug Parameters:")
    for key, value in DEBUG_PARAMS.items():
        print(f"  {key}: {value}")
    print("-" * 70)
    
    try:
        print("\n1. Initializing EvaluatorAgent...")
        
        # Load configuration
        with open(evaluator_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        evaluator = EvaluatorAgent(config)
        
        print("2. Fetching evaluation data...")
        # Fetch data using yfinance
        ticker = yf.Ticker(DEBUG_PARAMS['symbol'])
        data = ticker.history(
            start=DEBUG_PARAMS['start_date'], 
            end=DEBUG_PARAMS['end_date'], 
            interval=DEBUG_PARAMS['interval']
        )
        
        if data.empty:
            raise ValueError(f"No data found for symbol {DEBUG_PARAMS['symbol']} in the specified date range")
        
        print(f"   Retrieved {len(data)} data points for {DEBUG_PARAMS['symbol']}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        
        print("3. Creating evaluation environment...")
        # Prepare data for IntradayTradingEnv
        # For debugging, we'll create simple features from the price data
        price_data = data['Close']
        
        # Create basic features (RSI, EMA, etc.) - simplified for debugging
        features_df = pd.DataFrame(index=data.index)
        features_df['returns'] = data['Close'].pct_change().fillna(0)
        features_df['volume_norm'] = (data['Volume'] / data['Volume'].rolling(20).mean()).fillna(1)
        features_df['hour'] = data.index.hour / 24.0  # Normalized hour
        
        # Convert to numpy array for the environment
        processed_feature_data = features_df.fillna(0).values
        
        # Create evaluation environment with proper parameters
        env_config = config['environment_config']
        eval_env = IntradayTradingEnv(
            processed_feature_data=processed_feature_data,
            price_data=price_data,
            initial_capital=DEBUG_PARAMS['initial_capital'],
            transaction_cost_pct=env_config['transaction_cost'],
            lookback_window=env_config['lookback_window'],
            log_trades=True,
            position_sizing_pct_capital=env_config['max_position_size']
        )
        
        print("4. Setting up model...")
        # Handle model setup
        if DEBUG_PARAMS['use_dummy_model'] or not DEBUG_PARAMS['real_model_path']:
            # Create a dummy model file for testing
            dummy_model_path = Path(MODELS_DIR) / "dummy_model_for_debug"
            dummy_model_path.with_suffix('.dummy').touch()
            model_path = str(dummy_model_path)
            print(f"   Created dummy model for testing: {model_path}.dummy")
        else:
            model_path = DEBUG_PARAMS['real_model_path']
            print(f"   Using real model: {model_path}")
        
        print("5. Running evaluation...")
        # Run the evaluation
        results = evaluator.run(
            eval_env=eval_env,
            model_path=model_path,
            algorithm_name=DEBUG_PARAMS['algorithm'],
            model_name_tag=DEBUG_PARAMS['model_name']
        )
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        if results:
            print("Performance Metrics:")
            for key, value in results.items():
                if isinstance(value, float):
                    if 'pct' in key.lower():
                        print(f"  {key.replace('_', ' ').title()}: {value:.4f}%")
                    elif 'ratio' in key.lower():
                        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                    elif 'capital' in key.lower() or 'value' in key.lower():
                        print(f"  {key.replace('_', ' ').title()}: ${value:,.2f}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            
            print(f"\nComponent Status:")
            status = evaluator.get_component_status()
            for key, value in status.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
                
            print(f"\nFiles Generated:")
            print(f"  Reports Directory: {REPORTS_DIR}")
            print(f"  Config Directory: {CONFIG_DIR}")
            print(f"  Models Directory: {MODELS_DIR}")
            
            print("\n✅ Evaluation completed successfully!")
            
            # Hook functions for demonstration (similar to orchestrator)
            def after_evaluation_hook(eval_results=None, **kwargs):
                logging.info(f"[HOOK] Evaluation completed. Results summary:")
                if eval_results:
                    for key, value in eval_results.items():
                        if key in ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct']:
                            logging.info(f"[HOOK]   {key}: {value}")
            
            # Call the hook
            after_evaluation_hook(eval_results=results)
            
        else:
            print("❌ Evaluation failed. Check logs for details.")
            return 1
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
