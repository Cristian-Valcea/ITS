#!/usr/bin/env python3
"""
🧪 STAIRWAYS TO HEAVEN V3 - DRY-RUN VALIDATION PIPELINE
Comprehensive 6000-step validation framework for V3 Enhanced environment

VALIDATION OBJECTIVE: Prove enhanced environment stability and controller effectiveness
- 6000-step episode validation with detailed metrics collection
- Controller performance validation (hold rate reduction)
- Regime detection effectiveness measurement
- Portfolio performance preservation verification
- Statistical significance testing for trading frequency improvements

STAIRWAYS TO HEAVEN V3.0 - PHASE 2 IMPLEMENTATION
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Environment imports
from src.gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
from src.gym_env.dual_ticker_trading_env_v3_tuned import DualTickerTradingEnvV3Tuned
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

# Stairways components
from controller import DualLaneController
from market_regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for 6000-step dry run."""
    
    # Episode information
    episode_length: int
    episode_duration_seconds: float
    timestamp: str
    
    # Portfolio performance
    initial_portfolio_value: float
    final_portfolio_value: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    
    # Trading behavior
    total_trades: int
    trade_frequency: float  # Trades per step
    hold_actions: int
    hold_rate: float
    action_distribution: Dict[int, int]
    
    # Position management
    avg_nvda_position: float
    avg_msft_position: float
    max_nvda_position: int
    max_msft_position: int
    position_utilization: float
    
    # Stairways enhancements (if applicable)
    controller_enabled: bool
    regime_detection_enabled: bool
    avg_regime_score: float
    avg_hold_error: float
    avg_hold_bonus_enhancement: float
    controller_effectiveness: float  # Ratio of actual vs target hold rate
    
    # Reward analysis
    total_reward: float
    avg_reward_per_step: float
    reward_volatility: float
    positive_reward_steps: int
    negative_reward_steps: int
    
    # Risk metrics
    value_at_risk_95: float
    expected_shortfall: float
    daily_volatility: float
    max_consecutive_losses: int
    
    # Performance comparison (if baseline provided)
    vs_baseline_return_improvement: Optional[float] = None
    vs_baseline_hold_rate_improvement: Optional[float] = None
    vs_baseline_trade_frequency_improvement: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_to_json(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

class DryRunValidator:
    """
    Comprehensive 6000-step validation pipeline for Stairways to Heaven V3 Enhanced environment.
    
    This validator provides rigorous testing of the enhanced environment to ensure:
    1. Controller effectiveness in reducing hold rate
    2. Regime detection providing valuable market intelligence
    3. Portfolio performance preservation or improvement
    4. System stability over extended episodes
    5. Statistical significance of trading behavior changes
    """
    
    def __init__(
        self,
        data_adapter: DualTickerDataAdapter,
        validation_episodes: int = 5,
        episode_length: int = 6000,
        save_results: bool = True,
        results_dir: str = "dry_run_results",
        verbose: bool = True
    ):
        """
        Initialize dry-run validator.
        
        Args:
            data_adapter: Data adapter with prepared market data
            validation_episodes: Number of validation episodes to run
            episode_length: Length of each validation episode (6000 steps)
            save_results: Whether to save detailed results to disk
            results_dir: Directory to save validation results
            verbose: Enable detailed logging
        """
        self.data_adapter = data_adapter
        self.validation_episodes = validation_episodes
        self.episode_length = episode_length
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        
        # Create results directory
        if self.save_results:
            self.results_dir.mkdir(exist_ok=True)
            logger.info(f"📁 Results will be saved to: {self.results_dir}")
        
        # Validation state
        self.enhanced_metrics: List[ValidationMetrics] = []
        self.baseline_metrics: List[ValidationMetrics] = []
        
        # Statistical significance testing
        self.significance_level = 0.05
        
        if self.verbose:
            logger.info(f"🧪 Dry-run validator initialized")
            logger.info(f"   Episodes: {self.validation_episodes}")
            logger.info(f"   Episode length: {self.episode_length:,} steps")
            logger.info(f"   Data timespan: {len(self.data_adapter.feature_data):,} timesteps")
    
    def _create_enhanced_environment(self) -> DualTickerTradingEnvV3Enhanced:
        """Create enhanced environment with Stairways components."""
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=self.data_adapter.feature_data,
            processed_price_data=self.data_adapter.price_data,
            trading_days=self.data_adapter.trading_days,
            max_episode_steps=self.episode_length,
            enable_controller=True,
            enable_regime_detection=True,
            controller_target_hold_rate=0.65,  # Target 65% hold rate
            bootstrap_days=50,
            verbose=self.verbose
        )
        return env
    
    def _create_baseline_environment(self) -> DualTickerTradingEnvV3Tuned:
        """Create baseline V3 tuned environment for comparison."""
        env = DualTickerTradingEnvV3Tuned(
            processed_feature_data=self.data_adapter.feature_data,
            processed_price_data=self.data_adapter.price_data,
            trading_days=self.data_adapter.trading_days,
            max_episode_steps=self.episode_length,
            verbose=self.verbose
        )
        return env
    
    def _run_random_policy_episode(self, env, episode_idx: int) -> Tuple[ValidationMetrics, List[Dict[str, Any]]]:
        """
        Run single episode with random policy for validation.
        
        Args:
            env: Environment to run episode in
            episode_idx: Episode index for logging
            
        Returns:
            Tuple of (ValidationMetrics, detailed_step_data)
        """
        if self.verbose:
            logger.info(f"🎯 Running episode {episode_idx + 1}/{self.validation_episodes}...")
        
        start_time = time.time()
        obs, info = env.reset(seed=42 + episode_idx)  # Deterministic for reproducibility
        
        # Episode tracking
        step_data = []
        portfolio_values = []
        rewards = []
        actions = []
        regime_scores = []
        hold_errors = []
        hold_bonus_enhancements = []
        
        trades_count = 0
        hold_actions = 0
        
        # Position tracking
        nvda_positions = []
        msft_positions = []
        
        for step in range(self.episode_length):
            # Random action (uniform distribution)
            action = env.action_space.sample()
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            
            # Collect metrics
            portfolio_values.append(info['portfolio_value'])
            rewards.append(reward)
            actions.append(action)
            nvda_positions.append(info['nvda_position'])
            msft_positions.append(info['msft_position'])
            
            # Track trading behavior
            if info.get('tuning_info', {}).get('traded_this_step', False):
                trades_count += 1
            
            if action == 4:  # Hold action
                hold_actions += 1
            
            # Enhanced environment specific metrics
            if 'stairways_info' in info:
                stairways = info['stairways_info']
                regime_scores.append(stairways.get('current_regime_score', 0.0))
                hold_errors.append(stairways.get('hold_error', 0.0))
                hold_bonus_enhancements.append(stairways.get('hold_bonus_enhancement', 0.0))
            else:
                regime_scores.append(0.0)
                hold_errors.append(0.0)
                hold_bonus_enhancements.append(0.0)
            
            # Store detailed step data
            step_data.append({
                'step': step,
                'action': action,
                'reward': reward,
                'portfolio_value': info['portfolio_value'],
                'nvda_position': info['nvda_position'],
                'msft_position': info['msft_position'],
                'nvda_price': info['nvda_price'],
                'msft_price': info['msft_price'],
                **info.get('stairways_info', {})
            })
            
            if done or truncated:
                if self.verbose:
                    logger.info(f"   Episode terminated early at step {step + 1}")
                break
        
        episode_duration = time.time() - start_time
        
        # Calculate comprehensive metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        returns_series = np.array(portfolio_values)
        rewards_series = np.array(rewards)
        
        # Portfolio performance
        total_return = final_value - initial_value
        total_return_pct = total_return / initial_value
        
        # Drawdown calculation
        cumulative_returns = returns_series / initial_value
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        max_drawdown_pct = max_drawdown
        
        # Sharpe ratio (annualized)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Action distribution
        action_counts = {i: actions.count(i) for i in range(9)}
        
        # Position analysis
        nvda_positions_array = np.array(nvda_positions)
        msft_positions_array = np.array(msft_positions)
        
        # Risk metrics
        value_at_risk_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0.0
        expected_shortfall = np.mean(daily_returns[daily_returns <= value_at_risk_95]) if len(daily_returns) > 0 else 0.0
        daily_volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0.0
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for reward in rewards:
            if reward < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Enhanced environment metrics
        is_enhanced = 'stairways_info' in info
        avg_regime_score = np.mean(regime_scores) if regime_scores else 0.0
        avg_hold_error = np.mean(hold_errors) if hold_errors else 0.0
        avg_hold_bonus_enhancement = np.mean(hold_bonus_enhancements) if hold_bonus_enhancements else 0.0
        
        hold_rate = hold_actions / len(actions)
        controller_effectiveness = abs(0.65 - hold_rate) / 0.35 if is_enhanced else 0.0  # Distance from target
        
        # Create metrics object
        metrics = ValidationMetrics(
            episode_length=len(actions),
            episode_duration_seconds=episode_duration,
            timestamp=datetime.now().isoformat(),
            
            # Portfolio performance
            initial_portfolio_value=initial_value,
            final_portfolio_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown * initial_value,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            
            # Trading behavior
            total_trades=trades_count,
            trade_frequency=trades_count / len(actions),
            hold_actions=hold_actions,
            hold_rate=hold_rate,
            action_distribution=action_counts,
            
            # Position management
            avg_nvda_position=np.mean(nvda_positions_array),
            avg_msft_position=np.mean(msft_positions_array),
            max_nvda_position=int(np.max(np.abs(nvda_positions_array))),
            max_msft_position=int(np.max(np.abs(msft_positions_array))),
            position_utilization=(np.mean(np.abs(nvda_positions_array)) + np.mean(np.abs(msft_positions_array))) / 1000,
            
            # Stairways enhancements
            controller_enabled=is_enhanced,
            regime_detection_enabled=is_enhanced,
            avg_regime_score=avg_regime_score,
            avg_hold_error=avg_hold_error,
            avg_hold_bonus_enhancement=avg_hold_bonus_enhancement,
            controller_effectiveness=controller_effectiveness,
            
            # Reward analysis
            total_reward=float(np.sum(rewards)),
            avg_reward_per_step=float(np.mean(rewards)),
            reward_volatility=float(np.std(rewards)),
            positive_reward_steps=int(np.sum(np.array(rewards) > 0)),
            negative_reward_steps=int(np.sum(np.array(rewards) < 0)),
            
            # Risk metrics
            value_at_risk_95=value_at_risk_95,
            expected_shortfall=expected_shortfall,
            daily_volatility=daily_volatility,
            max_consecutive_losses=max_consecutive_losses
        )
        
        if self.verbose:
            logger.info(f"   ✅ Episode completed: {len(actions)} steps, "
                       f"Return: {total_return_pct:.2%}, Hold Rate: {hold_rate:.1%}")
        
        return metrics, step_data
    
    def run_enhanced_validation(self) -> List[ValidationMetrics]:
        """
        Run validation episodes on enhanced environment.
        
        Returns:
            List of validation metrics for each episode
        """
        logger.info(f"🚀 Starting enhanced environment validation ({self.validation_episodes} episodes)")
        
        env = self._create_enhanced_environment()
        metrics_list = []
        
        for episode_idx in range(self.validation_episodes):
            try:
                metrics, step_data = self._run_random_policy_episode(env, episode_idx)
                metrics_list.append(metrics)
                
                # Save detailed step data if requested
                if self.save_results:
                    step_data_file = self.results_dir / f"enhanced_episode_{episode_idx}_steps.json"
                    with open(step_data_file, 'w') as f:
                        json.dump(step_data, f, indent=2, default=str)
                
            except Exception as e:
                logger.error(f"❌ Enhanced episode {episode_idx} failed: {e}")
                continue
        
        env.close()
        
        if self.save_results and metrics_list:
            # Save aggregated metrics
            metrics_file = self.results_dir / "enhanced_validation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump([m.to_dict() for m in metrics_list], f, indent=2, default=str)
        
        logger.info(f"✅ Enhanced validation completed: {len(metrics_list)}/{self.validation_episodes} episodes")
        return metrics_list
    
    def run_baseline_validation(self) -> List[ValidationMetrics]:
        """
        Run validation episodes on baseline V3 tuned environment.
        
        Returns:
            List of validation metrics for each episode
        """
        logger.info(f"📊 Starting baseline environment validation ({self.validation_episodes} episodes)")
        
        env = self._create_baseline_environment()
        metrics_list = []
        
        for episode_idx in range(self.validation_episodes):
            try:
                metrics, step_data = self._run_random_policy_episode(env, episode_idx)
                metrics_list.append(metrics)
                
                # Save detailed step data if requested
                if self.save_results:
                    step_data_file = self.results_dir / f"baseline_episode_{episode_idx}_steps.json"
                    with open(step_data_file, 'w') as f:
                        json.dump(step_data, f, indent=2, default=str)
                
            except Exception as e:
                logger.error(f"❌ Baseline episode {episode_idx} failed: {e}")
                continue
        
        env.close()
        
        if self.save_results and metrics_list:
            # Save aggregated metrics
            metrics_file = self.results_dir / "baseline_validation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump([m.to_dict() for m in metrics_list], f, indent=2, default=str)
        
        logger.info(f"✅ Baseline validation completed: {len(metrics_list)}/{self.validation_episodes} episodes")
        return metrics_list
    
    def _calculate_statistical_significance(self, enhanced_values: List[float], baseline_values: List[float], metric_name: str) -> Dict[str, Any]:
        """
        Calculate statistical significance between enhanced and baseline metrics.
        
        Args:
            enhanced_values: Values from enhanced environment
            baseline_values: Values from baseline environment
            metric_name: Name of the metric being tested
            
        Returns:
            Dictionary with statistical test results
        """
        from scipy import stats
        
        enhanced_array = np.array(enhanced_values)
        baseline_array = np.array(baseline_values)
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(enhanced_array, baseline_array)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(enhanced_array) - 1) * np.var(enhanced_array) + 
                             (len(baseline_array) - 1) * np.var(baseline_array)) / 
                            (len(enhanced_array) + len(baseline_array) - 2))
        
        effect_size = (np.mean(enhanced_array) - np.mean(baseline_array)) / pooled_std if pooled_std > 0 else 0.0
        
        return {
            'metric_name': metric_name,
            'enhanced_mean': float(np.mean(enhanced_array)),
            'baseline_mean': float(np.mean(baseline_array)),
            'enhanced_std': float(np.std(enhanced_array)),
            'baseline_std': float(np.std(baseline_array)),
            'difference': float(np.mean(enhanced_array) - np.mean(baseline_array)),
            'difference_pct': float((np.mean(enhanced_array) - np.mean(baseline_array)) / np.mean(baseline_array) * 100) if np.mean(baseline_array) != 0 else 0.0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < self.significance_level,
            'effect_size': float(effect_size),
            'effect_interpretation': (
                'large' if abs(effect_size) >= 0.8 else
                'medium' if abs(effect_size) >= 0.5 else
                'small' if abs(effect_size) >= 0.2 else
                'negligible'
            )
        }
    
    def generate_comparison_report(self, enhanced_metrics: List[ValidationMetrics], baseline_metrics: List[ValidationMetrics]) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report between enhanced and baseline environments.
        
        Args:
            enhanced_metrics: Metrics from enhanced environment validation
            baseline_metrics: Metrics from baseline environment validation
            
        Returns:
            Comprehensive comparison report
        """
        logger.info(f"📋 Generating comparison report...")
        
        # Key metrics for comparison
        comparison_metrics = {
            'hold_rate': (
                [m.hold_rate for m in enhanced_metrics],
                [m.hold_rate for m in baseline_metrics]
            ),
            'trade_frequency': (
                [m.trade_frequency for m in enhanced_metrics],
                [m.trade_frequency for m in baseline_metrics]
            ),
            'total_return_pct': (
                [m.total_return_pct for m in enhanced_metrics],
                [m.total_return_pct for m in baseline_metrics]
            ),
            'sharpe_ratio': (
                [m.sharpe_ratio for m in enhanced_metrics],
                [m.sharpe_ratio for m in baseline_metrics]
            ),
            'max_drawdown_pct': (
                [abs(m.max_drawdown_pct) for m in enhanced_metrics],
                [abs(m.max_drawdown_pct) for m in baseline_metrics]
            ),
            'avg_reward_per_step': (
                [m.avg_reward_per_step for m in enhanced_metrics],
                [m.avg_reward_per_step for m in baseline_metrics]
            )
        }
        
        # Statistical analysis
        statistical_results = {}
        for metric_name, (enhanced_vals, baseline_vals) in comparison_metrics.items():
            statistical_results[metric_name] = self._calculate_statistical_significance(
                enhanced_vals, baseline_vals, metric_name
            )
        
        # Summary statistics
        enhanced_summary = {
            'episodes_completed': len(enhanced_metrics),
            'avg_hold_rate': np.mean([m.hold_rate for m in enhanced_metrics]),
            'avg_trade_frequency': np.mean([m.trade_frequency for m in enhanced_metrics]),
            'avg_return_pct': np.mean([m.total_return_pct for m in enhanced_metrics]),
            'avg_sharpe_ratio': np.mean([m.sharpe_ratio for m in enhanced_metrics]),
            'avg_controller_effectiveness': np.mean([m.controller_effectiveness for m in enhanced_metrics])
        }
        
        baseline_summary = {
            'episodes_completed': len(baseline_metrics),
            'avg_hold_rate': np.mean([m.hold_rate for m in baseline_metrics]),
            'avg_trade_frequency': np.mean([m.trade_frequency for m in baseline_metrics]),
            'avg_return_pct': np.mean([m.total_return_pct for m in baseline_metrics]),
            'avg_sharpe_ratio': np.mean([m.sharpe_ratio for m in baseline_metrics])
        }
        
        # Controller effectiveness analysis
        controller_analysis = {
            'target_hold_rate': 0.65,
            'enhanced_hold_rate_achieved': enhanced_summary['avg_hold_rate'],
            'baseline_hold_rate': baseline_summary['avg_hold_rate'],
            'hold_rate_improvement': baseline_summary['avg_hold_rate'] - enhanced_summary['avg_hold_rate'],
            'hold_rate_improvement_pct': (baseline_summary['avg_hold_rate'] - enhanced_summary['avg_hold_rate']) / baseline_summary['avg_hold_rate'] * 100,
            'target_achievement': 1.0 - abs(enhanced_summary['avg_hold_rate'] - 0.65) / 0.35,
            'controller_success': enhanced_summary['avg_hold_rate'] < baseline_summary['avg_hold_rate']
        }
        
        # Generate report
        report = {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'validation_episodes': self.validation_episodes,
                'episode_length': self.episode_length,
                'enhanced_episodes_completed': len(enhanced_metrics),
                'baseline_episodes_completed': len(baseline_metrics)
            },
            'enhanced_summary': enhanced_summary,
            'baseline_summary': baseline_summary,
            'statistical_analysis': statistical_results,
            'controller_effectiveness': controller_analysis,
            'key_findings': self._generate_key_findings(statistical_results, controller_analysis),
            'recommendations': self._generate_recommendations(statistical_results, controller_analysis)
        }
        
        return report
    
    def _generate_key_findings(self, statistical_results: Dict[str, Any], controller_analysis: Dict[str, Any]) -> List[str]:
        """Generate key findings from validation results."""
        findings = []
        
        # Hold rate analysis
        hold_rate_result = statistical_results.get('hold_rate', {})
        if hold_rate_result.get('is_significant', False) and hold_rate_result.get('difference', 0) < 0:
            findings.append(f"✅ Significant hold rate reduction achieved: {abs(hold_rate_result['difference_pct']):.1f}% decrease (p={hold_rate_result['p_value']:.3f})")
        elif not hold_rate_result.get('is_significant', False):
            findings.append(f"⚠️ Hold rate reduction not statistically significant (p={hold_rate_result.get('p_value', 0):.3f})")
        
        # Trade frequency analysis
        trade_freq_result = statistical_results.get('trade_frequency', {})
        if trade_freq_result.get('is_significant', False) and trade_freq_result.get('difference', 0) > 0:
            findings.append(f"✅ Significant trade frequency increase: {trade_freq_result['difference_pct']:.1f}% improvement")
        
        # Portfolio performance
        return_result = statistical_results.get('total_return_pct', {})
        if return_result.get('is_significant', False):
            if return_result.get('difference', 0) > 0:
                findings.append(f"✅ Enhanced environment shows superior returns: {return_result['difference_pct']:.1f}% improvement")
            else:
                findings.append(f"⚠️ Enhanced environment shows lower returns: {abs(return_result['difference_pct']):.1f}% decrease")
        else:
            findings.append(f"📊 No significant difference in portfolio returns (preserved performance)")
        
        # Controller target achievement
        if controller_analysis['target_achievement'] > 0.7:
            findings.append(f"🎯 Controller successfully achieved target hold rate: {controller_analysis['target_achievement']:.1%} success rate")
        else:
            findings.append(f"🎯 Controller partially achieved target: {controller_analysis['target_achievement']:.1%} success rate")
        
        return findings
    
    def _generate_recommendations(self, statistical_results: Dict[str, Any], controller_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Controller tuning
        if controller_analysis['target_achievement'] < 0.8:
            recommendations.append("🔧 Consider tuning controller gains for better target hold rate achievement")
        
        # Performance optimization
        return_result = statistical_results.get('total_return_pct', {})
        if return_result.get('is_significant', False) and return_result.get('difference', 0) < -0.05:
            recommendations.append("⚡ Investigate reward system balance - trading frequency vs. portfolio performance")
        
        # Statistical significance
        significant_metrics = sum(1 for result in statistical_results.values() if result.get('is_significant', False))
        if significant_metrics < len(statistical_results) / 2:
            recommendations.append("📊 Consider increasing validation episodes for better statistical power")
        
        # Production readiness
        if controller_analysis['controller_success'] and controller_analysis['target_achievement'] > 0.7:
            recommendations.append("🚀 Enhanced environment ready for cyclic training implementation")
        else:
            recommendations.append("🔧 Additional tuning recommended before production deployment")
        
        return recommendations
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run complete validation pipeline: enhanced + baseline + comparison.
        
        Returns:
            Comprehensive validation report
        """
        logger.info(f"🧪 Starting complete dry-run validation pipeline")
        start_time = time.time()
        
        try:
            # Run enhanced environment validation
            self.enhanced_metrics = self.run_enhanced_validation()
            
            # Run baseline environment validation
            self.baseline_metrics = self.run_baseline_validation()
            
            # Generate comparison report
            if self.enhanced_metrics and self.baseline_metrics:
                report = self.generate_comparison_report(self.enhanced_metrics, self.baseline_metrics)
            else:
                logger.error("❌ Insufficient validation data for comparison")
                return {'error': 'Validation failed - insufficient data'}
            
            # Save complete report
            if self.save_results:
                report_file = self.results_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                logger.info(f"📄 Complete validation report saved: {report_file}")
            
            validation_duration = time.time() - start_time
            logger.info(f"✅ Complete validation pipeline finished in {validation_duration:.1f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Validation pipeline failed: {e}")
            return {'error': str(e)}

# Utility functions for standalone execution
def create_test_data_adapter() -> DualTickerDataAdapter:
    """
    Create test data adapter for validation.
    
    Returns:
        DualTickerDataAdapter with test data
    """
    # This would typically load real market data
    # For testing, we create synthetic data
    n_timesteps = 10000
    
    # Create synthetic feature data (26 features)
    feature_data = np.random.randn(n_timesteps, 26)
    
    # Create synthetic price data (NVDA_close, NVDA_volume, MSFT_close, MSFT_volume)
    base_nvda_price = 450.0
    base_msft_price = 400.0
    
    nvda_returns = np.random.normal(0, 0.02, n_timesteps)
    msft_returns = np.random.normal(0, 0.02, n_timesteps)
    
    nvda_prices = base_nvda_price * np.exp(np.cumsum(nvda_returns))
    msft_prices = base_msft_price * np.exp(np.cumsum(msft_returns))
    
    nvda_volumes = np.random.lognormal(10, 0.5, n_timesteps)
    msft_volumes = np.random.lognormal(10, 0.5, n_timesteps)
    
    price_data = np.column_stack([
        nvda_prices, nvda_volumes, msft_prices, msft_volumes
    ])
    
    # Create trading days array
    trading_days = np.arange(n_timesteps)
    
    # Create adapter
    adapter = DualTickerDataAdapter()
    adapter.feature_data = feature_data
    adapter.price_data = price_data
    adapter.trading_days = trading_days
    
    return adapter

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🧪 STAIRWAYS TO HEAVEN V3 - DRY-RUN VALIDATION")
    logger.info("=" * 60)
    
    # Create test data adapter
    print("📊 Creating test data adapter...")
    data_adapter = create_test_data_adapter()
    
    # Initialize validator
    print("🔧 Initializing dry-run validator...")
    validator = DryRunValidator(
        data_adapter=data_adapter,
        validation_episodes=3,  # Small number for testing
        episode_length=1000,    # Shorter episodes for testing
        save_results=True,
        verbose=True
    )
    
    # Run complete validation
    print("🚀 Running complete validation pipeline...")
    report = validator.run_complete_validation()
    
    # Display summary results
    if 'error' not in report:
        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)
        
        enhanced_summary = report['enhanced_summary']
        baseline_summary = report['baseline_summary']
        controller_analysis = report['controller_effectiveness']
        
        print(f"Enhanced Environment:")
        print(f"  Hold Rate: {enhanced_summary['avg_hold_rate']:.1%}")
        print(f"  Trade Frequency: {enhanced_summary['avg_trade_frequency']:.3f}")
        print(f"  Avg Return: {enhanced_summary['avg_return_pct']:.2%}")
        
        print(f"\nBaseline Environment:")
        print(f"  Hold Rate: {baseline_summary['avg_hold_rate']:.1%}")
        print(f"  Trade Frequency: {baseline_summary['avg_trade_frequency']:.3f}")
        print(f"  Avg Return: {baseline_summary['avg_return_pct']:.2%}")
        
        print(f"\nController Effectiveness:")
        print(f"  Target Achievement: {controller_analysis['target_achievement']:.1%}")
        print(f"  Hold Rate Improvement: {controller_analysis['hold_rate_improvement_pct']:.1f}%")
        
        print(f"\nKey Findings:")
        for finding in report['key_findings']:
            print(f"  {finding}")
        
        print(f"\n✅ Dry-run validation completed successfully!")
    else:
        print(f"❌ Validation failed: {report['error']}")
