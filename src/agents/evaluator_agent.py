# src/agents/evaluator_agent.py
# src/agents/evaluator_agent.py
import os
import logging
import pandas as pd
import numpy as np

# Attempt to import Stable-Baselines3 components if needed for model loading
try:
    from stable_baselines3 import DQN # Example, replace with actual algorithms used
    # from sb3_contrib import C51 # If C51 is from contrib
    SB3_MODEL_CLASSES = {
        'DQN': DQN,
        # 'C51': C51, # Add if used
    }
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    SB3_MODEL_CLASSES = {} # No real models can be loaded if SB3 is not available
    logging.warning("Stable-Baselines3 not found. EvaluatorAgent can only work with dummy models if SB3 is unavailable.")

from .base_agent import BaseAgent
from src.gym_env.intraday_trading_env import IntradayTradingEnv 
# Assuming performance_metrics.py will be created in src/utils/
# from ..utils import performance_metrics as pm

# Placeholder for performance metrics until utils.performance_metrics is created
def calculate_sharpe_ratio(returns_series, risk_free_rate=0.0, periods_per_year=252):
    if returns_series.std() == 0: return 0.0 # Avoid division by zero
    excess_returns = returns_series - risk_free_rate / periods_per_year
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    if portfolio_values.empty: return 0.0
    peak = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min() # Returns as a negative value or zero


class EvaluatorAgent(BaseAgent):
    """
    EvaluatorAgent is responsible for:
    1. Running backtests of a trained model on a held-out (evaluation) dataset.
    2. Computing performance metrics: Sharpe ratio, max drawdown, total return, turnover.
    3. Generating evaluation reports and logging full trade history from the backtest.
    """
    def __init__(self, config: dict, eval_env: IntradayTradingEnv = None):
        super().__init__(agent_name="EvaluatorAgent", config=config)
        
        self.reports_dir = self.config.get('reports_dir', 'reports/')
        # Default metrics if not specified in config
        default_metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'num_trades', 'turnover_ratio_period']
        self.eval_metrics_config = self.config.get('eval_metrics', default_metrics)
        self.default_model_load_path = self.config.get('model_load_path', None)
        
        os.makedirs(self.reports_dir, exist_ok=True)

        self.evaluation_env = eval_env 
        self.model_to_evaluate = None

        self.logger.info("EvaluatorAgent initialized.")
        self.logger.info(f"Evaluation reports will be saved to: {self.reports_dir}")
        self.logger.info(f"Metrics to calculate: {self.eval_metrics_config}")

    def set_env(self, env: IntradayTradingEnv):
        if not isinstance(env, IntradayTradingEnv):
            self.logger.error("Invalid environment type provided to EvaluatorAgent.")
            raise ValueError("Environment must be an instance of IntradayTradingEnv.")
        self.evaluation_env = env
        self.logger.info("Evaluation environment set for EvaluatorAgent.")

    def load_model(self, model_path: str, algorithm_name: str = "DQN"):
        # Convert algorithm_name to upper for case-insensitive matching
        algo_name_upper = algorithm_name.upper()

        if not os.path.exists(model_path) and not os.path.exists(model_path + ".dummy"):
            self.logger.error(f"Model file not found at {model_path} or {model_path}.dummy")
            self.model_to_evaluate = None
            return False
        
        self.logger.info(f"Loading model for evaluation from: {model_path} (Algorithm: {algo_name_upper})")
        
        if SB3_AVAILABLE and algo_name_upper in SB3_MODEL_CLASSES:
            ModelClass = SB3_MODEL_CLASSES[algo_name_upper]
            try:
                # Pass self.evaluation_env if the model needs env details for loading some policies
                # For prediction, env is often not strictly needed at load time if observation/action spaces match.
                self.model_to_evaluate = ModelClass.load(model_path, env=None) # Pass env if model policy needs it for setup
                self.logger.info(f"Successfully loaded SB3 model {algo_name_upper} from {model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading SB3 model {algo_name_upper} from {model_path}: {e}", exc_info=True)
                self.model_to_evaluate = None
                return False
        elif os.path.exists(model_path + ".dummy"): # Fallback to dummy model loading if SB3 not available or algo unknown to SB3 list
            self.logger.warning(f"SB3 model class for {algo_name_upper} not available or SB3 itself not installed. Attempting to load as dummy.")
            class DummyEvalModel: # Minimal dummy model for predict()
                def __init__(self, path_loaded_from): self.path = path_loaded_from; self.logger = logging.getLogger("DummyEvalModel")
                def predict(self, obs, deterministic=True):
                    # Assuming action space is Discrete(3) from IntradayTradingEnv
                    return self.logger.parent.evaluation_env.action_space.sample(), None # Sample a random action
                @classmethod
                def load(cls, path, env=None): 
                    logger = logging.getLogger("DummyEvalModel.Load")
                    if os.path.exists(path + ".dummy"):
                        logger.info(f"DummyEvalModel loaded (simulated) from {path}")
                        return cls(path)
                    logger.error(f"Dummy model file {path}.dummy not found.")
                    raise FileNotFoundError(f"No dummy model found at {path}.dummy")
            try:
                # Pass self to the dummy model so it can access self.evaluation_env for action_space.sample()
                self.model_to_evaluate = DummyEvalModel.load(model_path, env=self) 
                self.logger.info("Dummy model loaded successfully for evaluation.")
                return True
            except Exception as e:
                self.logger.error(f"Error loading dummy model from {model_path}.dummy: {e}", exc_info=True)
                self.model_to_evaluate = None
                return False
        else:
            self.logger.error(f"Cannot load model: SB3 is not available for algorithm '{algo_name_upper}' and no dummy file found at {model_path}.dummy")
            self.model_to_evaluate = None
            return False


    def run_backtest(self, deterministic: bool = True) -> tuple[pd.DataFrame | None, pd.Series | None]:
        """
        Runs the loaded model on the evaluation environment.

        Args:
            deterministic (bool): Whether to use deterministic actions from the model. True for evaluation.

        Returns:
            tuple[pd.DataFrame | None, pd.Series | None]: 
                - DataFrame of trade history from the backtest.
                - Series of portfolio values over time.
                Returns (None, None) if failed.
        """
        if self.model_to_evaluate is None:
            self.logger.error("No model loaded. Cannot run backtest.")
            return None, None
        if self.evaluation_env is None:
            self.logger.error("Evaluation environment not set. Cannot run backtest.")
            return None, None

        self.logger.info(f"Starting backtest on evaluation environment. Deterministic: {deterministic}")
        
        obs, info = self.evaluation_env.reset()
        terminated = False
        truncated = False
        total_steps = 0

        # Ensure the environment's portfolio_history is reset/cleared if it's not done in env.reset()
        # self.evaluation_env.portfolio_history = [self.evaluation_env.initial_capital] # Already done in env.reset()

        while not (terminated or truncated):
            action, _states = self.model_to_evaluate.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.evaluation_env.step(action)
            total_steps += 1
            if total_steps % 100 == 0: # Log progress
                 self.logger.debug(f"Backtest step {total_steps}, Action: {action}, Reward: {reward:.3f}, Portfolio Value: {info.get('portfolio_value', 'N/A'):.2f}")

        self.logger.info(f"Backtest finished after {total_steps} steps.")
        self.logger.info(f"Final portfolio value: {self.evaluation_env.portfolio_value:.2f}")
        
        trade_log_df = self.evaluation_env.get_trade_log()
        portfolio_history_s = self.evaluation_env.get_portfolio_history()

        if trade_log_df.empty:
            self.logger.warning("Backtest completed, but no trades were logged by the environment.")
        else:
            self.logger.info(f"Retrieved trade log with {len(trade_log_df)} entries.")
        
        if portfolio_history_s.empty:
            self.logger.warning("Portfolio history is empty after backtest.")
        else:
            self.logger.info(f"Retrieved portfolio history with {len(portfolio_history_s)} entries.")
            
        return trade_log_df, portfolio_history_s

    def _calculate_metrics(self, trade_log_df: pd.DataFrame, portfolio_history: pd.Series, initial_capital: float) -> dict:
        """
        Calculates performance metrics.
        
        Args:
            trade_log_df (pd.DataFrame): Log of trades.
            portfolio_history (pd.Series): Series of portfolio values over time, indexed by datetime.
            initial_capital (float): Initial capital.
            
        Returns:
            dict: Dictionary of calculated metrics.
        """
        metrics = {}
        if portfolio_history.empty:
            self.logger.warning("Portfolio history is empty, cannot calculate most metrics.")
            metrics['total_return_pct'] = 0.0
            metrics['initial_capital'] = round(initial_capital, 2)
            metrics['final_capital'] = round(initial_capital,2) # Assume no change if no history
            metrics['num_trades'] = len(trade_log_df)
            return metrics

        final_capital = portfolio_history.iloc[-1]

        # 1. Total Return
        if 'total_return_pct' in self.eval_metrics_config:
            total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100 if initial_capital else 0
            metrics['total_return_pct'] = round(total_return_pct, 4)
            metrics['initial_capital'] = round(initial_capital, 2)
            metrics['final_capital'] = round(final_capital, 2)
            self.logger.info(f"Calculated Total Return: {metrics['total_return_pct']:.2f}%")

        # Calculate daily returns if possible (assuming DatetimeIndex for portfolio_history)
        daily_returns = pd.Series([]) # Default to empty
        if isinstance(portfolio_history.index, pd.DatetimeIndex):
            # Resample to daily, then calculate percentage change.
            # Use 'B' (business day) or 'D' (calendar day) depending on data frequency.
            # If data is intraday, resample to daily first.
            # Assuming 'price_data' (and thus portfolio_history) index has sufficient frequency for daily resample.
            daily_portfolio_values = portfolio_history.resample('D').last().ffill()
            daily_returns = daily_portfolio_values.pct_change().dropna()
        else:
            self.logger.warning("Portfolio history does not have a DatetimeIndex. Cannot calculate daily returns for Sharpe/Sortino.")


        # 2. Sharpe Ratio
        if 'sharpe_ratio' in self.eval_metrics_config:
            if not daily_returns.empty:
                # Assuming risk_free_rate_annual = 0 for simplicity, can be configured
                risk_free_rate_annual = self.config.get('risk_free_rate_annual', 0.0) 
                # Assuming 252 trading days in a year
                periods_per_year = self.config.get('periods_per_year_for_sharpe', 252) 
                metrics['sharpe_ratio'] = round(calculate_sharpe_ratio(daily_returns, risk_free_rate_annual, periods_per_year), 4)
            else:
                metrics['sharpe_ratio'] = 0.0
            self.logger.info(f"Calculated Sharpe Ratio: {metrics['sharpe_ratio']}")

        # 3. Max Drawdown
        if 'max_drawdown_pct' in self.eval_metrics_config:
            max_dd = calculate_max_drawdown(portfolio_history) * 100 # Convert to percentage
            metrics['max_drawdown_pct'] = round(max_dd, 4)
            self.logger.info(f"Calculated Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

        # 4. Turnover
        if 'turnover_ratio_period' in self.eval_metrics_config:
            if not trade_log_df.empty and 'trade_value' in trade_log_df.columns:
                total_traded_value = trade_log_df['trade_value'].sum()
                # Turnover for the period = Total Traded Value / Average Capital (or Initial Capital)
                # Using initial capital for simplicity here. Average capital would be more accurate.
                avg_capital = initial_capital # Could also use portfolio_history.mean()
                turnover_ratio = total_traded_value / avg_capital if avg_capital else 0
                metrics['turnover_ratio_period'] = round(turnover_ratio, 4)
                metrics['total_traded_value'] = round(total_traded_value, 2)
            else:
                metrics['turnover_ratio_period'] = 0.0
                metrics['total_traded_value'] = 0.0
            self.logger.info(f"Calculated Turnover Ratio (Period): {metrics['turnover_ratio_period']}")
        
        # 5. Number of Trades
        metrics['num_trades'] = len(trade_log_df)
        self.logger.info(f"Number of trades in backtest: {metrics['num_trades']}")

        # TODO: Implement other metrics from self.eval_metrics_config as needed:
        # - Sortino Ratio (needs downside deviation of returns)
        # - Calmar Ratio (Total Return / Max Drawdown)
        # - Win Rate, Avg Win/Loss, Profit Factor (needs P&L per trade from trade_log_df)
        # - Avg Trade Duration (needs entry/exit times per trade from trade_log_df)
        if 'sortino_ratio' in self.eval_metrics_config: metrics['sortino_ratio'] = "TODO"
        if 'calmar_ratio' in self.eval_metrics_config: metrics['calmar_ratio'] = "TODO"
        if 'win_rate_pct' in self.eval_metrics_config: metrics['win_rate_pct'] = "TODO"
        
        return metrics

    def generate_report(self, metrics: dict, trade_log_df: pd.DataFrame, model_name: str) -> str:
        """
        Generates a human-readable report and saves trade log.

        Args:
            metrics (dict): Calculated performance metrics.
            trade_log_df (pd.DataFrame): DataFrame of trades.
            model_name (str): Name of the model evaluated, for report naming.

        Returns:
            str: Path to the saved text report.
        """
        report_name_base = f"eval_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        report_txt_path = os.path.join(self.reports_dir, f"{report_name_base}_summary.txt")
        report_trades_csv_path = os.path.join(self.reports_dir, f"{report_name_base}_trades.csv")

        report_content = f"Evaluation Report for Model: {model_name}\n"
        report_content += f"Report Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += "=" * 40 + "\n"
        report_content += "Performance Metrics:\n"
        for key, value in metrics.items():
            report_content += f"  {key.replace('_', ' ').title()}: {value}\n"
        report_content += "=" * 40 + "\n"

        try:
            with open(report_txt_path, 'w') as f:
                f.write(report_content)
            self.logger.info(f"Evaluation summary report saved to: {report_txt_path}")
        except Exception as e:
            self.logger.error(f"Error saving evaluation report: {e}")

        if trade_log_df is not None and not trade_log_df.empty:
            try:
                trade_log_df.to_csv(report_trades_csv_path, index=False)
                self.logger.info(f"Trade log saved to: {report_trades_csv_path}")
            except Exception as e:
                self.logger.error(f"Error saving trade log CSV: {e}")
        else:
            self.logger.info("No trades to save for this evaluation run.")
            
        return report_txt_path


    def run(self,
            eval_env: IntradayTradingEnv, # EnvAgent provides this
            model_path: str = None,       # TrainerAgent provides this
            algorithm_name: str = "DQN",  # From config
            model_name_tag: str = "model" # For report naming
           ) -> dict | None:
        """
        Main method for EvaluatorAgent: loads model, runs backtest, calculates metrics, generates report.

        Args:
            eval_env (IntradayTradingEnv): The environment to evaluate on.
            model_path (str, optional): Path to the trained model. Uses default if None.
            algorithm_name (str): Algorithm of the loaded model.
            model_name_tag (str): A tag/name for the model being evaluated, used in report filenames.

        Returns:
            dict or None: Dictionary of performance metrics, or None if evaluation failed.
        """
        self.logger.info(f"EvaluatorAgent run started for model: {model_name_tag} from path: {model_path or self.default_model_load_path}")
        self.set_env(eval_env) # Set the environment

        model_to_load = model_path or self.default_model_load_path
        if not model_to_load:
            self.logger.error("No model path provided or configured. Cannot run evaluation.")
            return None
            
        if not self.load_model(model_to_load, algorithm_name):
            self.logger.error(f"Failed to load model {model_to_load}. Evaluation aborted.")
            return None

        trade_log_df = self.run_backtest(deterministic=True)
        # trade_log_df could be None if backtest fails, or empty if no trades.
        # _calculate_metrics and generate_report should handle this.

        initial_capital = self.evaluation_env.initial_capital # Get from env instance
        # price_data for detailed metrics (e.g. full portfolio value series)
        # eval_price_series = self.evaluation_env.price_data # The price series used by the env

        metrics = self._calculate_metrics(trade_log_df if trade_log_df is not None else pd.DataFrame(),
                                          initial_capital)
        
        self.generate_report(metrics, trade_log_df, model_name_tag)
        
        self.logger.info(f"Evaluation for {model_name_tag} complete. Metrics: {metrics}")
        return metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock Evaluation Environment (from EnvAgent or IntradayTradingEnv directly) ---
    num_eval_steps = 150
    lookback_eval = 5
    num_features_eval = 4

    if lookback_eval > 1:
        mock_eval_features = np.random.rand(num_eval_steps, lookback_eval, num_features_eval).astype(np.float32)
    else:
        mock_eval_features = np.random.rand(num_eval_steps, num_features_eval).astype(np.float32)
    
    mock_eval_prices_arr = 150 + np.cumsum(np.random.randn(num_eval_steps) * 0.5)
    mock_eval_dates = pd.to_datetime(pd.date_range(start='2023-04-01', periods=num_eval_steps, freq='1min'))
    mock_eval_price_series = pd.Series(mock_eval_prices_arr, index=mock_eval_dates, name=COL_CLOSE)

    # Ensure the eval env has log_trades=True so EvaluatorAgent can get the log
    eval_env_instance = IntradayTradingEnv(
        processed_feature_data=mock_eval_features,
        price_data=mock_eval_price_series,
        initial_capital=200000,
        lookback_window=lookback_eval,
        max_daily_drawdown_pct=0.05,
        transaction_cost_pct=0.001,
        log_trades=True # Crucial for EvaluatorAgent
    )
    print(f"Mock evaluation environment created: {eval_env_instance}")

    # --- Mock Model (simulate a saved model from TrainerAgent) ---
    # TrainerAgent's dummy model saves a `.zip.dummy` file.
    mock_model_dir = "models/test_evaluator/"
    mock_model_name = "dummy_DQN_model.zip" # SB3 saves as .zip
    mock_model_path = os.path.join(mock_model_dir, mock_model_name)
    
    os.makedirs(mock_model_dir, exist_ok=True)
    with open(mock_model_path + ".dummy", "w") as f: # Create the dummy file
        f.write("This is a dummy trained model.")
    print(f"Mock model file created at: {mock_model_path}.dummy")
    
    # --- EvaluatorAgent Configuration ---
    eval_config = {
        'reports_dir': 'reports/test_evaluator',
        'eval_metrics': ['total_return', 'sharpe', 'max_drawdown', 'turnover', 'num_trades'],
        'model_load_path': mock_model_path # Default model to load
    }

    # --- Initialize and Run EvaluatorAgent ---
    evaluator_agent = EvaluatorAgent(config=eval_config)
    
    print("\nStarting EvaluatorAgent run...")
    # In a pipeline, eval_env_instance would come from EnvAgent, model_path from TrainerAgent.
    evaluation_results = evaluator_agent.run(
        eval_env=eval_env_instance,
        # model_path=mock_model_path, # Can omit if using default from config
        algorithm_name="DQN", # Must match how model was saved/what it is
        model_name_tag="TestModel_Run123"
    )

    if evaluation_results:
        print("\nEvaluatorAgent run completed. Metrics:")
        for metric, value in evaluation_results.items():
            print(f"  {metric}: {value}")
        
        # Check for report files
        report_dir = eval_config['reports_dir']
        txt_reports = [f for f in os.listdir(report_dir) if f.startswith('eval_TestModel_Run123') and f.endswith('_summary.txt')]
        csv_reports = [f for f in os.listdir(report_dir) if f.startswith('eval_TestModel_Run123') and f.endswith('_trades.csv')]

        if txt_reports: print(f"Summary report found: {os.path.join(report_dir, txt_reports[0])}")
        else: print("Summary report NOT found.")
        if csv_reports: print(f"Trades CSV found: {os.path.join(report_dir, csv_reports[0])}")
        else: print("Trades CSV NOT found (possibly no trades or log empty).")
    else:
        print("\nEvaluatorAgent run failed or returned no metrics.")

    eval_env_instance.close()
    # Clean up dummy model file
    if os.path.exists(mock_model_path + ".dummy"): os.remove(mock_model_path + ".dummy")
    print("\nEvaluatorAgent example run complete.")
