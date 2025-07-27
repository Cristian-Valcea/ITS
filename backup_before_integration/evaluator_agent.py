import os
import logging
from typing import Optional, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np

try:
    from stable_baselines3 import DQN
    SB3_MODEL_CLASSES = {'DQN': DQN}
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    SB3_MODEL_CLASSES = {}
    logging.warning("Stable-Baselines3 not found. EvaluatorAgent can only work with dummy models if SB3 is unavailable.")

from .base_agent import BaseAgent
from src.gym_env.intraday_trading_env import IntradayTradingEnv

def calculate_sharpe_ratio(returns_series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    if returns_series.std() == 0 or returns_series.empty:
        return 0.0
    excess_returns = returns_series - risk_free_rate / periods_per_year
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    if portfolio_values.empty:
        return 0.0
    peak = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min()

class EvaluatorAgent(BaseAgent):
    """
    EvaluatorAgent is responsible for:
    1. Running backtests of a trained model on a held-out (evaluation) dataset.
    2. Computing performance metrics: Sharpe ratio, max drawdown, total return, turnover.
    3. Generating evaluation reports and logging full trade history from the backtest.
    """
    def __init__(self, config: dict, eval_env: Optional[IntradayTradingEnv] = None):
        super().__init__(agent_name="EvaluatorAgent", config=config)
        self.reports_dir = self.config.get('reports_dir', 'reports/')
        self.eval_metrics_config = self.config.get(
            'eval_metrics',
            ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'num_trades', 'turnover_ratio_period']
        )
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

    def load_model(self, model_path: str, algorithm_name: str = "DQN") -> bool:
        algo_name_upper = algorithm_name.upper()
        if not isinstance(model_path, str) or not model_path:
            self.logger.error("Invalid model path provided.")
            self.model_to_evaluate = None
            return False

        if not os.path.exists(model_path) and not os.path.exists(model_path + ".dummy"):
            self.logger.error(f"Model file not found at {model_path} or {model_path}.dummy")
            self.model_to_evaluate = None
            return False

        self.logger.info(f"Loading model for evaluation from: {model_path} (Algorithm: {algo_name_upper})")
        if SB3_AVAILABLE and algo_name_upper in SB3_MODEL_CLASSES:
            ModelClass = SB3_MODEL_CLASSES[algo_name_upper]
            try:
                self.model_to_evaluate = ModelClass.load(model_path, env=None)
                self.logger.info(f"Successfully loaded SB3 model {algo_name_upper} from {model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading SB3 model {algo_name_upper} from {model_path}: {e}", exc_info=True)
                self.model_to_evaluate = None
                return False
        elif os.path.exists(model_path + ".dummy"):
            self.logger.warning(f"SB3 model class for {algo_name_upper} not available or SB3 not installed. Attempting to load as dummy.")
            class DummyEvalModel:
                def __init__(self, path_loaded_from):
                    self.path = path_loaded_from
                    self.logger = logging.getLogger("DummyEvalModel")
                def predict(self, obs, deterministic=True):
                    # Use the environment's action_space if available
                    if hasattr(self, 'env') and hasattr(self.env, 'action_space'):
                        return self.env.action_space.sample(), None
                    return 0, None
                @classmethod
                def load(cls, path, env=None):
                    logger = logging.getLogger("DummyEvalModel.Load")
                    if os.path.exists(path + ".dummy"):
                        logger.info(f"DummyEvalModel loaded (simulated) from {path}")
                        instance = cls(path)
                        if env is not None:
                            instance.env = env.evaluation_env if hasattr(env, 'evaluation_env') else env
                        return instance
                    logger.error(f"Dummy model file {path}.dummy not found.")
                    raise FileNotFoundError(f"No dummy model found at {path}.dummy")
            try:
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

    def run_backtest(self, deterministic: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Runs the loaded model on the evaluation environment.
        Returns (trade_log_df, portfolio_history_s) or (None, None) if failed.
        """
        if self.model_to_evaluate is None:
            self.logger.error("No model loaded. Cannot run backtest.")
            return None, None
        if self.evaluation_env is None:
            self.logger.error("Evaluation environment not set. Cannot run backtest.")
            return None, None

        self.logger.info(f"Starting backtest on evaluation environment. Deterministic: {deterministic}")
        try:
            obs, info = self.evaluation_env.reset()
            terminated = False
            truncated = False
            total_steps = 0

            while not (terminated or truncated):
                action, _states = self.model_to_evaluate.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.evaluation_env.step(action)
                total_steps += 1
                if total_steps % 100 == 0:
                    self.logger.debug(f"Backtest step {total_steps}, Action: {action}, Reward: {reward:.3f}, Portfolio Value: {info.get('portfolio_value', 'N/A'):.2f}")

            self.logger.info(f"Backtest finished after {total_steps} steps.")
            self.logger.info(f"Final portfolio value: {getattr(self.evaluation_env, 'portfolio_value', 0):.2f}")

            trade_log_df = self.evaluation_env.get_trade_log()
            portfolio_history_s = self.evaluation_env.get_portfolio_history()

            if trade_log_df is not None and trade_log_df.empty:
                self.logger.warning("Backtest completed, but no trades were logged by the environment.")
            elif trade_log_df is not None:
                self.logger.info(f"Retrieved trade log with {len(trade_log_df)} entries.")

            if portfolio_history_s is not None and portfolio_history_s.empty:
                self.logger.warning("Portfolio history is empty after backtest.")
            elif portfolio_history_s is not None:
                self.logger.info(f"Retrieved portfolio history with {len(portfolio_history_s)} entries.")

            return trade_log_df, portfolio_history_s
        except Exception as e:
            self.logger.error(f"Exception during backtest: {e}", exc_info=True)
            return None, None

    def _calculate_metrics(self, trade_log_df: pd.DataFrame, portfolio_history: pd.Series, initial_capital: float) -> Dict[str, Any]:
        """
        Calculates performance metrics.
        """
        metrics = {}
        if portfolio_history is None or portfolio_history.empty:
            self.logger.warning("Portfolio history is empty, cannot calculate most metrics.")
            metrics['total_return_pct'] = 0.0
            metrics['initial_capital'] = round(initial_capital, 2)
            metrics['final_capital'] = round(initial_capital, 2)
            metrics['num_trades'] = len(trade_log_df) if trade_log_df is not None else 0
            return metrics

        final_capital = portfolio_history.iloc[-1]
        if 'total_return_pct' in self.eval_metrics_config:
            total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100 if initial_capital else 0
            metrics['total_return_pct'] = round(total_return_pct, 4)
            metrics['initial_capital'] = round(initial_capital, 2)
            metrics['final_capital'] = round(final_capital, 2)
            self.logger.info(f"Calculated Total Return: {metrics['total_return_pct']:.2f}%")

        daily_returns = pd.Series(dtype=float)
        if isinstance(portfolio_history.index, pd.DatetimeIndex):
            daily_portfolio_values = portfolio_history.resample('D').last().ffill()
            daily_returns = daily_portfolio_values.pct_change().dropna()
        else:
            self.logger.warning("Portfolio history does not have a DatetimeIndex. Cannot calculate daily returns for Sharpe/Sortino.")

        if 'sharpe_ratio' in self.eval_metrics_config:
            if not daily_returns.empty:
                risk_free_rate_annual = self.config.get('risk_free_rate_annual', 0.0)
                periods_per_year = self.config.get('periods_per_year_for_sharpe', 252)
                metrics['sharpe_ratio'] = round(calculate_sharpe_ratio(daily_returns, risk_free_rate_annual, periods_per_year), 4)
            else:
                metrics['sharpe_ratio'] = 0.0
            self.logger.info(f"Calculated Sharpe Ratio: {metrics['sharpe_ratio']}")

        if 'max_drawdown_pct' in self.eval_metrics_config:
            max_dd = calculate_max_drawdown(portfolio_history) * 100
            metrics['max_drawdown_pct'] = round(max_dd, 4)
            self.logger.info(f"Calculated Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

        if 'turnover_ratio_period' in self.eval_metrics_config:
            if trade_log_df is not None and not trade_log_df.empty and 'trade_value' in trade_log_df.columns:
                total_traded_value = trade_log_df['trade_value'].sum()
                avg_capital = initial_capital
                turnover_ratio = total_traded_value / avg_capital if avg_capital else 0
                metrics['turnover_ratio_period'] = round(turnover_ratio, 4)
                metrics['total_traded_value'] = round(total_traded_value, 2)
            else:
                metrics['turnover_ratio_period'] = 0.0
                metrics['total_traded_value'] = 0.0
            self.logger.info(f"Calculated Turnover Ratio (Period): {metrics['turnover_ratio_period']}")

        metrics['num_trades'] = len(trade_log_df) if trade_log_df is not None else 0
        self.logger.info(f"Number of trades in backtest: {metrics['num_trades']}")

        # Placeholders for additional metrics
        for metric in ['sortino_ratio', 'calmar_ratio', 'win_rate_pct']:
            if metric in self.eval_metrics_config:
                metrics[metric] = "TODO"

        return metrics

    def generate_report(self, metrics: Dict[str, Any], trade_log_df: Optional[pd.DataFrame], model_name: str) -> str:
        """
        Generates a human-readable report and saves trade log.
        Returns the path to the saved text report.
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
            self.logger.error(f"Error saving evaluation report: {e}", exc_info=True)

        if trade_log_df is not None and not trade_log_df.empty:
            try:
                trade_log_df.to_csv(report_trades_csv_path, index=False)
                self.logger.info(f"Trade log saved to: {report_trades_csv_path}")
            except Exception as e:
                self.logger.error(f"Error saving trade log CSV: {e}", exc_info=True)
        else:
            self.logger.info("No trades to save for this evaluation run.")

        return report_txt_path

    def run(
        self,
        eval_env: IntradayTradingEnv,
        model_path: Optional[str] = None,
        algorithm_name: str = "DQN",
        model_name_tag: str = "model"
    ) -> Optional[Dict[str, Any]]:
        """
        Main method for EvaluatorAgent: loads model, runs backtest, calculates metrics, generates report.
        Returns a dictionary of performance metrics, or None if evaluation failed.
        """
        self.logger.info(f"EvaluatorAgent run started for model: {model_name_tag} from path: {model_path or self.default_model_load_path}")
        try:
            self.set_env(eval_env)
            model_to_load = model_path or self.default_model_load_path
            if not model_to_load:
                self.logger.error("No model path provided or configured. Cannot run evaluation.")
                return None

            if not self.load_model(model_to_load, algorithm_name):
                self.logger.error(f"Failed to load model {model_to_load}. Evaluation aborted.")
                return None

            trade_log_df, portfolio_history = self.run_backtest(deterministic=True)
            initial_capital = getattr(self.evaluation_env, 'initial_capital', 0.0)
            metrics = self._calculate_metrics(
                trade_log_df if trade_log_df is not None else pd.DataFrame(),
                portfolio_history if portfolio_history is not None else pd.Series(dtype=float),
                initial_capital
            )
            self.generate_report(metrics, trade_log_df, model_name_tag)
            self.logger.info(f"Evaluation for {model_name_tag} complete. Metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Exception during evaluation run: {e}", exc_info=True)
            return None
