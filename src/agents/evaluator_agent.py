# src/agents/evaluator_agent.py
import os
import logging
import pandas as pd
import numpy as np
# from stable_baselines3.common.base_class import BaseAlgorithm # For type hinting loaded model

from .base_agent import BaseAgent
from src.gym_env.intraday_trading_env import IntradayTradingEnv # For running backtests
# from ..utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_turnover # Example

class EvaluatorAgent(BaseAgent):
    """
    EvaluatorAgent is responsible for:
    1. Running backtests of a trained model on a held-out (evaluation) dataset.
    2. Computing performance metrics: Sharpe ratio, max drawdown, total return, turnover.
    3. Generating evaluation reports and logging full trade history from the backtest.
    4. Supporting walk-forward validation by evaluating on different time windows.
    """
    def __init__(self, config: dict, eval_env: IntradayTradingEnv = None):
        """
        Initializes the EvaluatorAgent.

        Args:
            config (dict): Configuration dictionary. Expected keys:
                           'reports_dir': Path to save evaluation reports and trade logs.
                           'eval_metrics': List of metrics to compute (e.g., ['sharpe', 'max_drawdown']).
                           'model_load_path': Default path to load model if not provided in run().
            eval_env (IntradayTradingEnv, optional): Pre-initialized evaluation environment.
                                                     If None, it's expected to be created or set.
        """
        super().__init__(agent_name="EvaluatorAgent", config=config)
        
        self.reports_dir = self.config.get('reports_dir', 'reports/')
        self.eval_metrics_config = self.config.get('eval_metrics', ['sharpe', 'max_drawdown', 'total_return', 'turnover'])
        self.default_model_load_path = self.config.get('model_load_path', None)
        
        os.makedirs(self.reports_dir, exist_ok=True)

        self.evaluation_env = eval_env # Can be set via set_env or created with data
        self.model_to_evaluate = None

        self.logger.info("EvaluatorAgent initialized.")
        self.logger.info(f"Evaluation reports will be saved to: {self.reports_dir}")

    def set_env(self, env: IntradayTradingEnv):
        """Sets the environment to be used for evaluation."""
        if not isinstance(env, IntradayTradingEnv):
            self.logger.error("Invalid environment type provided to EvaluatorAgent.")
            raise ValueError("Environment must be an instance of IntradayTradingEnv.")
        self.evaluation_env = env
        # Ensure the eval env does not log trades itself if EvaluatorAgent handles it, or ensure coordination.
        # The env's log_trades_flag might be set to False if EvaluatorAgent wants to control logging.
        # Or, EvaluatorAgent can just retrieve the log from the env after the run.
        self.logger.info("Evaluation environment set for EvaluatorAgent.")

    def load_model(self, model_path: str, algorithm_name: str = "DQN"):
        """
        Loads a trained SB3 model for evaluation.
        
        Args:
            model_path (str): Path to the saved model (.zip file).
            algorithm_name (str): The name of the algorithm the model was trained with (e.g., 'DQN', 'C51').
                                  Used to call the correct SB3 load method.
        """
        if not os.path.exists(model_path) and not os.path.exists(model_path + ".dummy"): # Check for dummy too
            self.logger.error(f"Model file not found at {model_path}")
            self.model_to_evaluate = None
            return False
        
        self.logger.info(f"Loading model for evaluation from: {model_path} (Algorithm: {algorithm_name})")
        try:
            # Actual SB3 model loading:
            # if algorithm_name.upper() == 'DQN':
            #     from stable_baselines3 import DQN
            #     self.model_to_evaluate = DQN.load(model_path, env=self.evaluation_env) # Env can be optional for predict
            # elif algorithm_name.upper() == 'C51':
            #     from stable_baselines3 import C51 # Or from contrib
            #     self.model_to_evaluate = C51.load(model_path, env=self.evaluation_env)
            # else:
            #     self.logger.error(f"Unsupported algorithm '{algorithm_name}' for loading.")
            #     return False

            # SKELETON: Using the DummySB3Model.load from TrainerAgent for consistency
            from .trainer_agent import TrainerAgent # Access DummySB3Model
            # This creates a circular dependency for the __main__ block, handle carefully.
            # In a real project, DummySB3Model might be in a common utils/testing module.
            # For this skeleton structure, this import is problematic if running evaluator_agent.py directly
            # unless trainer_agent.py defines DummySB3Model at the global scope without its own __main__ causing issues.
            # Let's redefine a minimal DummyModel here for Evaluator's __main__ to work standalone.
            class DummyEvalModel:
                def __init__(self, path): self.path = path; self.logger = logging.getLogger("DummyEvalModel")
                def predict(self, obs, deterministic=True):
                    self.logger.debug(f"DummyEvalModel predicting for obs of shape {obs.shape if isinstance(obs, np.ndarray) else 'dict'}. Deterministic: {deterministic}")
                    # Based on observation space of IntradayTradingEnv (can be Box or Dict)
                    # Action is Discrete(3). Return a random action.
                    # This needs to know the action space. Assume Discrete(3) for now.
                    return np.random.randint(0, 3), None # action, state (state is for recurrent)
                @classmethod
                def load(cls, path, env=None): # env is optional for predict, but good for consistency
                    logger = logging.getLogger("DummyEvalModel.Load")
                    if os.path.exists(path + ".dummy"): # Check for dummy file from TrainerAgent
                        logger.info(f"DummyEvalModel loaded (simulated) from {path}")
                        return cls(path)
                    logger.error(f"Dummy model file {path}.dummy not found for loading.")
                    raise FileNotFoundError(f"No dummy model found at {path}.dummy")

            self.model_to_evaluate = DummyEvalModel.load(model_path, env=self.evaluation_env)
            self.logger.info("Model loaded successfully for evaluation.")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
            self.model_to_evaluate = None
            return False

    def run_backtest(self, deterministic: bool = True) -> pd.DataFrame | None:
        """
        Runs the loaded model on the evaluation environment.

        Args:
            deterministic (bool): Whether to use deterministic actions from the model. True for evaluation.

        Returns:
            pd.DataFrame or None: DataFrame of trade history from the backtest, or None if failed.
        """
        if self.model_to_evaluate is None:
            self.logger.error("No model loaded. Cannot run backtest.")
            return None
        if self.evaluation_env is None:
            self.logger.error("Evaluation environment not set. Cannot run backtest.")
            return None

        self.logger.info(f"Starting backtest on evaluation environment. Deterministic: {deterministic}")
        
        obs, info = self.evaluation_env.reset()
        terminated = False
        truncated = False
        total_steps = 0

        while not (terminated or truncated):
            action, _states = self.model_to_evaluate.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.evaluation_env.step(action)
            total_steps += 1
            if total_steps % 100 == 0: # Log progress
                 self.logger.debug(f"Backtest step {total_steps}, Action: {action}, Reward: {reward:.2f}, Capital: {info['current_capital']:.2f}")

        self.logger.info(f"Backtest finished after {total_steps} steps.")
        self.logger.info(f"Final capital: {self.evaluation_env.current_capital:.2f}")
        
        trade_log_df = self.evaluation_env.get_trade_log()
        if trade_log_df.empty:
            self.logger.warning("Backtest completed, but no trades were logged by the environment.")
        else:
            self.logger.info(f"Retrieved trade log with {len(trade_log_df)} entries from environment.")
        
        return trade_log_df

    def _calculate_metrics(self, trade_log_df: pd.DataFrame, initial_capital: float, prices_df: pd.Series = None) -> dict:
        """
        Calculates performance metrics from the trade log and environment state.
        
        Args:
            trade_log_df (pd.DataFrame): Log of trades from the backtest.
            initial_capital (float): Initial capital from the environment.
            prices_df (pd.Series): Series of prices during the backtest period, if needed for some metrics.
                                   The environment itself tracks capital, so this might be redundant
                                   if all metrics can be derived from final capital and trade log.

        Returns:
            dict: Dictionary of calculated metrics.
        """
        metrics = {}
        final_capital = self.evaluation_env.current_capital # Get from env after run

        # 1. Total Return
        if 'total_return' in self.eval_metrics_config:
            total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
            metrics['total_return_pct'] = round(total_return_pct, 4)
            metrics['initial_capital'] = round(initial_capital, 2)
            metrics['final_capital'] = round(final_capital, 2)
            self.logger.info(f"Calculated Total Return: {total_return_pct:.2f}%")

        # TODO: Implement other metrics. These are placeholders.
        # Requires prices_df (portfolio value over time) for accurate Sharpe and Max Drawdown.
        # The IntradayTradingEnv could be modified to log portfolio value at each step.
        # Or, reconstruct portfolio value history from trade_log_df and price_data.
        
        # For simplicity, let's assume we have a series of portfolio values.
        # This would ideally be logged by the environment or reconstructed.
        # For skeleton, we'll use dummy values or skip complex metrics.
        
        # Placeholder: Get portfolio history (list of capital values at each step)
        # This should be a proper output from the environment run or reconstructed.
        # portfolio_history = getattr(self.evaluation_env, 'capital_history', [initial_capital, final_capital])
        # For now, let's assume the trade_log_df contains enough info or these are simplified.
        
        # 2. Sharpe Ratio (Simplified: using daily/periodic returns)
        if 'sharpe' in self.eval_metrics_config:
            # TODO: Need daily/periodic returns. From trade_log or env's capital history.
            # Placeholder:
            # returns_series = pd.Series(portfolio_history).pct_change().dropna()
            # if not returns_series.empty and returns_series.std() != 0:
            #     sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252) # Annualized (if daily)
            #     metrics['sharpe_ratio'] = round(sharpe, 4)
            # else:
            #     metrics['sharpe_ratio'] = 0.0
            metrics['sharpe_ratio'] = "TODO: Implement Sharpe Ratio"
            self.logger.info(f"Calculated Sharpe Ratio: {metrics['sharpe_ratio']}")


        # 3. Max Drawdown
        if 'max_drawdown' in self.eval_metrics_config:
            # TODO: Need portfolio value history.
            # Placeholder:
            # p_values = pd.Series(portfolio_history)
            # peak = p_values.expanding(min_periods=1).max()
            # drawdown = (p_values - peak) / peak
            # max_dd = drawdown.min() * 100 # Percentage
            # metrics['max_drawdown_pct'] = round(max_dd, 4)
            metrics['max_drawdown_pct'] = "TODO: Implement Max Drawdown"
            self.logger.info(f"Calculated Max Drawdown: {metrics['max_drawdown_pct']}")

        # 4. Turnover
        if 'turnover' in self.eval_metrics_config and not trade_log_df.empty:
            # Sum of absolute value of all trades / (Number of days * Average Capital) -> Annualized
            # Simplified: Total traded value / initial capital for the period
            # `trade_value` should be calculated correctly in `record_trade` of RiskAgent/Env.
            # Assuming trade_log_df has 'entry_price' and some quantity, or direct 'trade_value'.
            # Let's assume the env's trade log has a 'cost_entry' or 'cost_exit' that reflects value.
            # For simplicity, let's sum up 'profit' and 'cost' columns if they exist,
            # though 'profit' is not trade value. This needs a proper 'trade_value' column.
            # Example: `total_traded_value = (trade_log_df['abs_trade_value']).sum()`
            # For now, if 'cost_entry' and 'cost_exit' exist (from IntradayTradingEnv log):
            total_traded_value = 0
            if 'cost_entry' in trade_log_df.columns: # Value of entered position
                # Cost is transaction cost. Need actual value.
                # If entry_price and a hypothetical 'shares' column existed:
                # total_traded_value += (trade_log_df['entry_price'] * trade_log_df['shares']).sum()
                # This metric is underspecified with current IntradayTradingEnv trade log.
                # Let's make a placeholder.
                pass # TODO: Implement turnover based on actual trade values.
            metrics['turnover_ratio'] = "TODO: Implement Turnover" # E.g. total_traded_value / initial_capital
            self.logger.info(f"Calculated Turnover: {metrics['turnover_ratio']}")
        elif 'turnover' in self.eval_metrics_config:
             metrics['turnover_ratio'] = 0.0 # No trades

        metrics['num_trades'] = len(trade_log_df)
        self.logger.info(f"Number of trades in backtest: {metrics['num_trades']}")
        
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
    mock_eval_price_series = pd.Series(mock_eval_prices_arr, index=mock_eval_dates, name='close')

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
