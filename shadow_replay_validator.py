#!/usr/bin/env python3
"""
🎬 STAIRWAYS TO HEAVEN V3 - SHADOW REPLAY VALIDATOR
Enhanced shadow replay validation with 3-day tick-for-tick validation and deterministic testing

VALIDATION OBJECTIVE: Prove enhanced environment deterministic reproducibility and controller consistency
- 3-day continuous replay validation with identical market conditions
- Deterministic seed-based testing for reproducible results
- Full tick storage and replay capability for debugging
- Market regime consistency verification across replays
- Controller state persistence and recovery validation  
- Statistical correlation analysis between replay runs

STAIRWAYS TO HEAVEN V3.0 - PHASE 2 IMPLEMENTATION
"""

import numpy as np
import pandas as pd
import logging
import json
import sqlite3
import gzip
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import time

# Environment imports
from src.gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

# Stairways components
from controller import DualLaneController
from market_regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

@dataclass
class ReplayTick:
    """Single tick record for shadow replay storage."""
    
    timestamp: str
    step_number: int
    
    # Market data
    nvda_price: float
    msft_price: float
    nvda_volume: float
    msft_volume: float
    
    # Environment state
    portfolio_value: float
    cash: float
    nvda_position: int
    msft_position: int
    
    # Agent action and reward
    action: int
    reward: float
    
    # Stairways intelligence
    regime_score: float
    hold_error: float
    hold_bonus_enhancement: float
    current_hold_rate: float
    
    # Controller state
    controller_fast_gain: float
    controller_slow_gain: float
    controller_slow_adjustment: float
    controller_step_count: int
    
    # Regime detector state
    momentum_buffer_size: int
    volatility_buffer_size: int
    divergence_buffer_size: int
    regime_bootstrap_progress: float
    
    # Market features (first 5 for compactness)
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_step_info(cls, step_number: int, action: int, reward: float, info: Dict[str, Any], features: np.ndarray) -> 'ReplayTick':
        """Create ReplayTick from environment step information."""
        stairways_info = info.get('stairways_info', {})
        controller_health = stairways_info.get('controller_health', {})
        regime_health = stairways_info.get('regime_detector_health', {})
        
        return cls(
            timestamp=datetime.now().isoformat(),
            step_number=step_number,
            
            # Market data
            nvda_price=info.get('nvda_price', 0.0),
            msft_price=info.get('msft_price', 0.0),
            nvda_volume=0.0,  # Would be extracted from price data if available
            msft_volume=0.0,  # Would be extracted from price data if available
            
            # Environment state
            portfolio_value=info.get('portfolio_value', 0.0),
            cash=info.get('cash', 0.0),
            nvda_position=info.get('nvda_position', 0),
            msft_position=info.get('msft_position', 0),
            
            # Agent action and reward
            action=action,
            reward=reward,
            
            # Stairways intelligence
            regime_score=stairways_info.get('current_regime_score', 0.0),
            hold_error=stairways_info.get('hold_error', 0.0),
            hold_bonus_enhancement=stairways_info.get('hold_bonus_enhancement', 0.0),
            current_hold_rate=stairways_info.get('current_hold_rate', 0.0),
            
            # Controller state
            controller_fast_gain=controller_health.get('fast_gain', 0.0),
            controller_slow_gain=controller_health.get('slow_gain', 0.0),
            controller_slow_adjustment=controller_health.get('slow_adjustment', 0.0),
            controller_step_count=controller_health.get('step_count', 0),
            
            # Regime detector state
            momentum_buffer_size=regime_health.get('momentum_buffer_size', 0),
            volatility_buffer_size=regime_health.get('volatility_buffer_size', 0),
            divergence_buffer_size=regime_health.get('divergence_buffer_size', 0),
            regime_bootstrap_progress=regime_health.get('bootstrap_progress', 0.0),
            
            # Market features (first 5 for compactness)
            feature_0=features[0] if len(features) > 0 else 0.0,
            feature_1=features[1] if len(features) > 1 else 0.0,
            feature_2=features[2] if len(features) > 2 else 0.0,
            feature_3=features[3] if len(features) > 3 else 0.0,
            feature_4=features[4] if len(features) > 4 else 0.0
        )

@dataclass
class ReplayValidationResult:
    """Results from shadow replay validation."""
    
    replay_id: str
    timestamp: str
    
    # Replay configuration
    seed: int
    episode_length: int
    validation_days: int
    
    # Deterministic validation
    is_deterministic: bool
    tick_hash_original: str
    tick_hash_replay: str
    hash_match: bool
    
    # Statistical consistency
    portfolio_correlation: float
    action_consistency_rate: float
    reward_correlation: float
    regime_score_correlation: float
    
    # Controller consistency
    controller_state_drift: float
    hold_rate_consistency: float
    hold_error_correlation: float
    
    # Performance metrics
    original_final_portfolio: float
    replay_final_portfolio: float
    portfolio_difference: float
    portfolio_difference_pct: float
    
    # Regime detection consistency
    regime_bootstrap_consistency: bool
    regime_buffer_size_consistency: bool
    regime_score_drift: float
    
    # Validation summary
    validation_passed: bool
    validation_score: float  # Overall score [0, 1]
    issues_detected: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

class ShadowReplayValidator:
    """
    Enhanced shadow replay validator for Stairways to Heaven V3 environment.
    
    This validator provides comprehensive replay testing to ensure:
    1. Deterministic reproducibility with identical seeds
    2. Controller state consistency across replays
    3. Market regime detection stability
    4. Full tick-level audit trail for debugging
    5. Statistical correlation analysis for validation
    6. SQLite WAL mode for high-performance tick storage
    """
    
    def __init__(
        self,
        data_adapter: DualTickerDataAdapter,
        validation_days: int = 3,
        database_path: str = "shadow_replay.db",
        enable_compression: bool = True,
        correlation_threshold: float = 0.95,
        consistency_threshold: float = 0.98,
        verbose: bool = True
    ):
        """
        Initialize shadow replay validator.
        
        Args:
            data_adapter: Data adapter with prepared market data
            validation_days: Number of days to validate (3-day default)
            database_path: SQLite database path for tick storage
            enable_compression: Enable gzip compression for tick data
            correlation_threshold: Minimum correlation for statistical validation
            consistency_threshold: Minimum consistency rate for deterministic validation
            verbose: Enable detailed logging
        """
        self.data_adapter = data_adapter
        self.validation_days = validation_days
        self.database_path = Path(database_path)
        self.enable_compression = enable_compression
        self.correlation_threshold = correlation_threshold
        self.consistency_threshold = consistency_threshold
        self.verbose = verbose
        
        # Calculate episode length (approximately 3 trading days)
        # Assuming ~390 minutes per trading day
        self.episode_length = validation_days * 390
        
        # Initialize database with WAL mode (reviewer requirement)
        self._initialize_database()
        
        # Validation state
        self.replay_results: List[ReplayValidationResult] = []
        
        if self.verbose:
            logger.info(f"🎬 Shadow replay validator initialized")
            logger.info(f"   Validation days: {self.validation_days}")
            logger.info(f"   Episode length: {self.episode_length:,} steps")
            logger.info(f"   Database: {self.database_path} (WAL mode)")
            logger.info(f"   Compression: {self.enable_compression}")
            logger.info(f"   Correlation threshold: {self.correlation_threshold}")
    
    def _initialize_database(self):
        """
        Initialize SQLite database with WAL mode for tick storage.
        
        REVIEWER REQUIREMENT: SQLite WAL mode for high-performance concurrent access.
        """
        conn = sqlite3.connect(self.database_path)
        
        # Enable WAL mode (reviewer requirement)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=memory")
        
        # Create replay runs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS replay_runs (
                replay_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                seed INTEGER NOT NULL,
                episode_length INTEGER NOT NULL,
                validation_days INTEGER NOT NULL,
                final_portfolio REAL,
                validation_passed BOOLEAN,
                validation_score REAL,
                metadata TEXT
            )
        """)
        
        # Create ticks table with compression support
        conn.execute("""
            CREATE TABLE IF NOT EXISTS replay_ticks (
                replay_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                tick_data BLOB NOT NULL,
                tick_hash TEXT NOT NULL,
                PRIMARY KEY (replay_id, step_number),
                FOREIGN KEY (replay_id) REFERENCES replay_runs(replay_id)
            )
        """)
        
        # Create indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_replay_step ON replay_ticks(replay_id, step_number)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_seed ON replay_runs(seed)")
        
        conn.commit()
        conn.close()
        
        if self.verbose:
            logger.info(f"✅ Database initialized with WAL mode: {self.database_path}")
    
    def _serialize_tick(self, tick: ReplayTick) -> bytes:
        """
        Serialize tick data for storage.
        
        Args:
            tick: ReplayTick to serialize
            
        Returns:
            Serialized tick data (optionally compressed)
        """
        data = pickle.dumps(tick)
        
        if self.enable_compression:
            data = gzip.compress(data)
        
        return data
    
    def _deserialize_tick(self, data: bytes) -> ReplayTick:
        """
        Deserialize tick data from storage.
        
        Args:
            data: Serialized tick data
            
        Returns:
            Deserialized ReplayTick
        """
        if self.enable_compression:
            data = gzip.decompress(data)
        
        return pickle.loads(data)
    
    def _calculate_tick_hash(self, tick: ReplayTick) -> str:
        """
        Calculate hash of tick data for deterministic validation.
        
        Args:
            tick: ReplayTick to hash
            
        Returns:
            SHA-256 hash of tick data
        """
        # Create deterministic representation
        tick_repr = f"{tick.step_number}:{tick.action}:{tick.reward:.6f}:{tick.portfolio_value:.2f}:{tick.regime_score:.6f}"
        return hashlib.sha256(tick_repr.encode()).hexdigest()[:16]  # Short hash for storage
    
    def _store_replay_run(self, replay_id: str, seed: int, ticks: List[ReplayTick], metadata: Dict[str, Any]) -> None:
        """
        Store complete replay run in database.
        
        Args:
            replay_id: Unique replay identifier
            seed: Random seed used for replay
            ticks: List of all ticks from replay
            metadata: Additional metadata about the run
        """
        conn = sqlite3.connect(self.database_path)
        
        try:
            # Store replay run metadata
            final_portfolio = ticks[-1].portfolio_value if ticks else 0.0
            
            conn.execute("""
                INSERT OR REPLACE INTO replay_runs 
                (replay_id, timestamp, seed, episode_length, validation_days, final_portfolio, validation_passed, validation_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                replay_id,
                datetime.now().isoformat(),
                seed,
                len(ticks),
                self.validation_days,
                final_portfolio,
                metadata.get('validation_passed', False),
                metadata.get('validation_score', 0.0),
                json.dumps(metadata)
            ))
            
            # Store all ticks
            for tick in ticks:
                tick_data = self._serialize_tick(tick)
                tick_hash = self._calculate_tick_hash(tick)
                
                conn.execute("""
                    INSERT OR REPLACE INTO replay_ticks 
                    (replay_id, step_number, tick_data, tick_hash)
                    VALUES (?, ?, ?, ?)
                """, (replay_id, tick.step_number, tick_data, tick_hash))
            
            conn.commit()
            
            if self.verbose:
                logger.info(f"✅ Stored replay run: {replay_id} ({len(ticks)} ticks)")
        
        finally:
            conn.close()
    
    def _load_replay_run(self, replay_id: str) -> Tuple[Dict[str, Any], List[ReplayTick]]:
        """
        Load replay run from database.
        
        Args:
            replay_id: Unique replay identifier
            
        Returns:
            Tuple of (metadata, ticks)
        """
        conn = sqlite3.connect(self.database_path)
        
        try:
            # Load metadata
            cursor = conn.execute("""
                SELECT metadata FROM replay_runs WHERE replay_id = ?
            """, (replay_id,))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Replay run not found: {replay_id}")
            
            metadata = json.loads(result[0])
            
            # Load ticks
            cursor = conn.execute("""
                SELECT tick_data FROM replay_ticks 
                WHERE replay_id = ? 
                ORDER BY step_number
            """, (replay_id,))
            
            ticks = []
            for (tick_data,) in cursor.fetchall():
                tick = self._deserialize_tick(tick_data)
                ticks.append(tick)
            
            return metadata, ticks
        
        finally:
            conn.close()
    
    def _run_single_replay(self, seed: int, policy_function=None) -> Tuple[str, List[ReplayTick]]:
        """
        Run single replay episode with deterministic seed.
        
        Args:
            seed: Random seed for deterministic replay
            policy_function: Policy function (default: random policy)
            
        Returns:
            Tuple of (replay_id, ticks)
        """
        replay_id = f"replay_{seed}_{int(time.time())}"
        
        if self.verbose:
            logger.info(f"🎬 Running replay: {replay_id} (seed={seed})")
        
        # Create enhanced environment
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=self.data_adapter.feature_data,
            processed_price_data=self.data_adapter.price_data,
            trading_days=self.data_adapter.trading_days,
            max_episode_steps=self.episode_length,
            enable_controller=True,
            enable_regime_detection=True,
            controller_target_hold_rate=0.65,
            bootstrap_days=50,
            verbose=False  # Reduce verbosity during replay
        )
        
        # Reset with deterministic seed
        obs, info = env.reset(seed=seed)
        
        ticks = []
        
        # Use default random policy if none provided
        if policy_function is None:
            np.random.seed(seed)  # Ensure deterministic action selection
            policy_function = lambda obs: np.random.randint(0, 9)
        
        for step in range(self.episode_length):
            # Select action
            action = policy_function(obs)
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            
            # Create and store tick
            tick = ReplayTick.from_step_info(
                step_number=step,
                action=action,
                reward=reward,
                info=info,
                features=obs
            )
            
            ticks.append(tick)
            
            if done or truncated:
                if self.verbose:
                    logger.info(f"   Replay terminated early at step {step + 1}")
                break
        
        env.close()
        
        if self.verbose:
            final_portfolio = ticks[-1].portfolio_value if ticks else 0.0
            logger.info(f"   ✅ Replay completed: {len(ticks)} steps, Portfolio: ${final_portfolio:,.2f}")
        
        return replay_id, ticks
    
    def _calculate_replay_correlations(self, original_ticks: List[ReplayTick], replay_ticks: List[ReplayTick]) -> Dict[str, float]:
        """
        Calculate statistical correlations between original and replay runs.
        
        Args:
            original_ticks: Ticks from original run
            replay_ticks: Ticks from replay run
            
        Returns:
            Dictionary of correlation coefficients
        """
        min_length = min(len(original_ticks), len(replay_ticks))
        
        if min_length < 2:
            return {key: 0.0 for key in ['portfolio', 'reward', 'regime_score', 'hold_error']}
        
        # Extract time series
        orig_portfolio = [tick.portfolio_value for tick in original_ticks[:min_length]]
        replay_portfolio = [tick.portfolio_value for tick in replay_ticks[:min_length]]
        
        orig_rewards = [tick.reward for tick in original_ticks[:min_length]]
        replay_rewards = [tick.reward for tick in replay_ticks[:min_length]]
        
        orig_regime = [tick.regime_score for tick in original_ticks[:min_length]]
        replay_regime = [tick.regime_score for tick in replay_ticks[:min_length]]
        
        orig_hold_error = [tick.hold_error for tick in original_ticks[:min_length]]
        replay_hold_error = [tick.hold_error for tick in replay_ticks[:min_length]]
        
        # Calculate correlations
        correlations = {}
        
        try:
            correlations['portfolio'] = float(np.corrcoef(orig_portfolio, replay_portfolio)[0, 1])
        except:
            correlations['portfolio'] = 0.0
        
        try:
            correlations['reward'] = float(np.corrcoef(orig_rewards, replay_rewards)[0, 1])
        except:
            correlations['reward'] = 0.0
        
        try:
            correlations['regime_score'] = float(np.corrcoef(orig_regime, replay_regime)[0, 1])
        except:
            correlations['regime_score'] = 0.0
        
        try:
            correlations['hold_error'] = float(np.corrcoef(orig_hold_error, replay_hold_error)[0, 1])
        except:
            correlations['hold_error'] = 0.0
        
        # Replace NaN values with 0.0
        for key in correlations:
            if np.isnan(correlations[key]) or np.isinf(correlations[key]):
                correlations[key] = 0.0
        
        return correlations
    
    def _validate_replay_consistency(self, original_ticks: List[ReplayTick], replay_ticks: List[ReplayTick]) -> ReplayValidationResult:
        """
        Validate consistency between original and replay runs.
        
        Args:
            original_ticks: Ticks from original run
            replay_ticks: Ticks from replay run
            
        Returns:
            Comprehensive replay validation result
        """
        min_length = min(len(original_ticks), len(replay_ticks))
        
        # Calculate tick hashes for deterministic validation
        orig_hash_parts = [self._calculate_tick_hash(tick) for tick in original_ticks[:min_length]]
        replay_hash_parts = [self._calculate_tick_hash(tick) for tick in replay_ticks[:min_length]]
        
        orig_hash = hashlib.sha256(''.join(orig_hash_parts).encode()).hexdigest()
        replay_hash = hashlib.sha256(''.join(replay_hash_parts).encode()).hexdigest()
        
        hash_match = orig_hash == replay_hash
        
        # Calculate statistical correlations
        correlations = self._calculate_replay_correlations(original_ticks, replay_ticks)
        
        # Action consistency
        action_matches = sum(1 for i in range(min_length) if original_ticks[i].action == replay_ticks[i].action)
        action_consistency_rate = action_matches / min_length if min_length > 0 else 0.0
        
        # Controller state consistency
        orig_controller_states = [(tick.controller_slow_adjustment, tick.controller_step_count) for tick in original_ticks[:min_length]]
        replay_controller_states = [(tick.controller_slow_adjustment, tick.controller_step_count) for tick in replay_ticks[:min_length]]
        
        controller_drift = 0.0
        if orig_controller_states and replay_controller_states:
            state_diffs = [abs(o[0] - r[0]) + abs(o[1] - r[1]) for o, r in zip(orig_controller_states, replay_controller_states)]
            controller_drift = np.mean(state_diffs)
        
        # Hold rate consistency
        orig_hold_rates = [tick.current_hold_rate for tick in original_ticks[:min_length]]
        replay_hold_rates = [tick.current_hold_rate for tick in replay_ticks[:min_length]]
        
        hold_rate_consistency = 1.0 - np.mean([abs(o - r) for o, r in zip(orig_hold_rates, replay_hold_rates)]) if orig_hold_rates else 0.0
        
        # Regime detection consistency
        orig_final_bootstrap = original_ticks[-1].regime_bootstrap_progress if original_ticks else 0.0
        replay_final_bootstrap = replay_ticks[-1].regime_bootstrap_progress if replay_ticks else 0.0
        regime_bootstrap_consistency = abs(orig_final_bootstrap - replay_final_bootstrap) < 0.01
        
        orig_final_buffer_size = original_ticks[-1].momentum_buffer_size if original_ticks else 0
        replay_final_buffer_size = replay_ticks[-1].momentum_buffer_size if replay_ticks else 0
        regime_buffer_size_consistency = orig_final_buffer_size == replay_final_buffer_size
        
        # Portfolio performance consistency
        orig_final_portfolio = original_ticks[-1].portfolio_value if original_ticks else 0.0
        replay_final_portfolio = replay_ticks[-1].portfolio_value if replay_ticks else 0.0
        portfolio_difference = replay_final_portfolio - orig_final_portfolio
        portfolio_difference_pct = portfolio_difference / orig_final_portfolio * 100 if orig_final_portfolio != 0 else 0.0
        
        # R4 FIX: Check PnL delta against spec (max 0.1% deviation)
        pnl_delta_threshold = 0.001  # 0.1% threshold
        pnl_within_bounds = abs(portfolio_difference_pct / 100) <= pnl_delta_threshold
        
        # Regime score drift
        regime_score_drift = np.mean([abs(o.regime_score - r.regime_score) for o, r in zip(original_ticks[:min_length], replay_ticks[:min_length])]) if min_length > 0 else 0.0
        
        # Validation scoring
        validation_components = {
            'hash_match': 1.0 if hash_match else 0.0,
            'action_consistency': action_consistency_rate,
            'portfolio_correlation': correlations['portfolio'],
            'reward_correlation': correlations['reward'],
            'regime_correlation': correlations['regime_score'],
            'controller_consistency': max(0.0, 1.0 - controller_drift / 10.0),  # Normalize drift
            'hold_rate_consistency': hold_rate_consistency,
            'regime_bootstrap_consistency': 1.0 if regime_bootstrap_consistency else 0.0,
            'pnl_consistency': 1.0 if pnl_within_bounds else 0.0  # R4 FIX: Add PnL consistency
        }
        
        validation_score = np.mean(list(validation_components.values()))
        validation_passed = validation_score >= self.consistency_threshold
        
        # Identify issues
        issues = []
        if not hash_match:
            issues.append("Deterministic hash mismatch - non-reproducible behavior")
        if action_consistency_rate < self.consistency_threshold:
            issues.append(f"Low action consistency: {action_consistency_rate:.2%}")
        if correlations['portfolio'] < self.correlation_threshold:
            issues.append(f"Low portfolio correlation: {correlations['portfolio']:.3f}")
        if controller_drift > 1.0:
            issues.append(f"Controller state drift detected: {controller_drift:.3f}")
        if not regime_bootstrap_consistency:
            issues.append("Regime detector bootstrap inconsistency")
        if not pnl_within_bounds:
            issues.append(f"PnL deviation exceeds threshold: {abs(portfolio_difference_pct):.3f}% > {pnl_delta_threshold*100:.1f}%")
        
        return ReplayValidationResult(
            replay_id=f"validation_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            seed=original_ticks[0].step_number if original_ticks else 0,  # Placeholder
            episode_length=min_length,
            validation_days=self.validation_days,
            
            # Deterministic validation
            is_deterministic=hash_match,
            tick_hash_original=orig_hash,
            tick_hash_replay=replay_hash,
            hash_match=hash_match,
            
            # Statistical consistency
            portfolio_correlation=correlations['portfolio'],
            action_consistency_rate=action_consistency_rate,
            reward_correlation=correlations['reward'],
            regime_score_correlation=correlations['regime_score'],
            
            # Controller consistency
            controller_state_drift=controller_drift,
            hold_rate_consistency=hold_rate_consistency,
            hold_error_correlation=correlations['hold_error'],
            
            # Performance metrics
            original_final_portfolio=orig_final_portfolio,
            replay_final_portfolio=replay_final_portfolio,
            portfolio_difference=portfolio_difference,
            portfolio_difference_pct=portfolio_difference_pct,
            
            # Regime detection consistency
            regime_bootstrap_consistency=regime_bootstrap_consistency,
            regime_buffer_size_consistency=regime_buffer_size_consistency,
            regime_score_drift=regime_score_drift,
            
            # Validation summary
            validation_passed=validation_passed,
            validation_score=validation_score,
            issues_detected=issues
        )
    
    def run_shadow_replay_validation(self, test_seeds: List[int] = None, num_replays: int = 3) -> List[ReplayValidationResult]:
        """
        Run complete shadow replay validation.
        
        Args:
            test_seeds: List of seeds to test (default: [42, 123, 456])
            num_replays: Number of replay attempts per seed
            
        Returns:
            List of validation results
        """
        if test_seeds is None:
            test_seeds = [42, 123, 456]  # Deterministic seeds for reproducible testing
        
        logger.info(f"🎬 Starting shadow replay validation")
        logger.info(f"   Test seeds: {test_seeds}")
        logger.info(f"   Replays per seed: {num_replays}")
        logger.info(f"   Episode length: {self.episode_length:,} steps")
        
        validation_results = []
        
        for seed in test_seeds:
            logger.info(f"🎲 Testing seed: {seed}")
            
            # Run original episode
            original_id, original_ticks = self._run_single_replay(seed)
            
            # Store original run
            self._store_replay_run(
                replay_id=original_id,
                seed=seed,
                ticks=original_ticks,
                metadata={'run_type': 'original', 'seed': seed}
            )
            
            # Run replay episodes
            for replay_idx in range(num_replays):
                logger.info(f"   Replay {replay_idx + 1}/{num_replays}")
                
                # Run replay with same seed
                replay_id, replay_ticks = self._run_single_replay(seed)
                
                # Store replay run
                self._store_replay_run(
                    replay_id=replay_id,
                    seed=seed,
                    ticks=replay_ticks,
                    metadata={'run_type': 'replay', 'seed': seed, 'replay_index': replay_idx}
                )
                
                # Validate consistency
                validation_result = self._validate_replay_consistency(original_ticks, replay_ticks)
                validation_result.seed = seed
                validation_result.replay_id = f"{seed}_replay_{replay_idx}"
                
                validation_results.append(validation_result)
                
                if self.verbose:
                    logger.info(f"     Validation score: {validation_result.validation_score:.3f} "
                               f"({'PASS' if validation_result.validation_passed else 'FAIL'})")
        
        # Store validation results
        self.replay_results = validation_results
        
        logger.info(f"✅ Shadow replay validation completed: {len(validation_results)} results")
        
        return validation_results
    
    def generate_validation_report(self, validation_results: List[ReplayValidationResult]) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_results: Results from shadow replay validation
            
        Returns:
            Comprehensive validation report
        """
        logger.info(f"📋 Generating shadow replay validation report...")
        
        if not validation_results:
            return {'error': 'No validation results available'}
        
        # Overall statistics
        total_validations = len(validation_results)
        passed_validations = sum(1 for result in validation_results if result.validation_passed)
        pass_rate = passed_validations / total_validations
        
        # Score statistics
        scores = [result.validation_score for result in validation_results]
        avg_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        std_score = np.std(scores)
        
        # Deterministic validation statistics
        hash_matches = sum(1 for result in validation_results if result.hash_match)
        deterministic_rate = hash_matches / total_validations
        
        # Correlation statistics
        portfolio_correlations = [result.portfolio_correlation for result in validation_results]
        avg_portfolio_correlation = np.mean(portfolio_correlations)
        
        action_consistencies = [result.action_consistency_rate for result in validation_results]
        avg_action_consistency = np.mean(action_consistencies)
        
        # Controller drift statistics
        controller_drifts = [result.controller_state_drift for result in validation_results]
        avg_controller_drift = np.mean(controller_drifts)
        max_controller_drift = np.max(controller_drifts)
        
        # Issue analysis
        all_issues = []
        for result in validation_results:
            all_issues.extend(result.issues_detected)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Generate findings
        findings = []
        if deterministic_rate >= 0.95:
            findings.append(f"✅ Excellent deterministic reproducibility: {deterministic_rate:.1%} hash matches")
        elif deterministic_rate >= 0.8:
            findings.append(f"⚠️ Good deterministic reproducibility: {deterministic_rate:.1%} hash matches")
        else:
            findings.append(f"❌ Poor deterministic reproducibility: {deterministic_rate:.1%} hash matches")
        
        if avg_portfolio_correlation >= self.correlation_threshold:
            findings.append(f"✅ Strong portfolio correlation: {avg_portfolio_correlation:.3f}")
        else:
            findings.append(f"⚠️ Weak portfolio correlation: {avg_portfolio_correlation:.3f}")
        
        if avg_controller_drift < 0.1:
            findings.append(f"✅ Minimal controller drift: {avg_controller_drift:.3f}")
        elif avg_controller_drift < 1.0:
            findings.append(f"⚠️ Moderate controller drift: {avg_controller_drift:.3f}")
        else:
            findings.append(f"❌ Significant controller drift: {avg_controller_drift:.3f}")
        
        # Generate recommendations
        recommendations = []
        if pass_rate < 0.8:
            recommendations.append("🔧 Environment determinism needs improvement - investigate random seed handling")
        
        if avg_portfolio_correlation < self.correlation_threshold:
            recommendations.append("📊 Portfolio correlation below threshold - review reward system consistency")
        
        if max_controller_drift > 2.0:
            recommendations.append("⚙️ Controller state drift detected - verify reset() and state management")
        
        if deterministic_rate >= 0.95 and avg_portfolio_correlation >= self.correlation_threshold:
            recommendations.append("🚀 Shadow replay validation passed - environment ready for production")
        
        # Create comprehensive report
        report = {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_validations': total_validations,
                'passed_validations': passed_validations,
                'pass_rate': pass_rate,
                'validation_days': self.validation_days,
                'episode_length': self.episode_length
            },
            'score_statistics': {
                'average_score': avg_score,
                'minimum_score': min_score,
                'maximum_score': max_score,
                'score_std_dev': std_score,
                'consistency_threshold': self.consistency_threshold
            },
            'deterministic_validation': {
                'hash_match_rate': deterministic_rate,
                'total_hash_matches': hash_matches,
                'is_deterministic': deterministic_rate >= 0.95
            },
            'correlation_analysis': {
                'avg_portfolio_correlation': avg_portfolio_correlation,
                'avg_action_consistency': avg_action_consistency,
                'correlation_threshold': self.correlation_threshold,
                'correlations_above_threshold': sum(1 for corr in portfolio_correlations if corr >= self.correlation_threshold)
            },
            'controller_analysis': {
                'avg_controller_drift': avg_controller_drift,
                'max_controller_drift': max_controller_drift,
                'low_drift_rate': sum(1 for drift in controller_drifts if drift < 0.1) / len(controller_drifts)
            },
            'issue_analysis': {
                'total_issues': len(all_issues),
                'unique_issues': len(issue_counts),
                'issue_counts': issue_counts
            },
            'detailed_results': [result.to_dict() for result in validation_results],
            'key_findings': findings,
            'recommendations': recommendations
        }
        
        return report
    
    def run_complete_shadow_validation(self, test_seeds: List[int] = None, num_replays: int = 3) -> Dict[str, Any]:
        """
        Run complete shadow replay validation pipeline.
        
        Args:
            test_seeds: List of seeds to test
            num_replays: Number of replay attempts per seed
            
        Returns:
            Comprehensive validation report
        """
        logger.info(f"🎬 Starting complete shadow replay validation pipeline")
        start_time = time.time()
        
        try:
            # Run shadow replay validation
            validation_results = self.run_shadow_replay_validation(test_seeds, num_replays)
            
            # Generate report
            report = self.generate_validation_report(validation_results)
            
            # Save report to database metadata
            report_metadata = {
                'report_type': 'shadow_replay_complete',
                'validation_results_count': len(validation_results),
                'pass_rate': report['validation_summary']['pass_rate'],
                'deterministic_rate': report['deterministic_validation']['hash_match_rate'],
                'avg_score': report['score_statistics']['average_score']
            }
            
            validation_duration = time.time() - start_time
            logger.info(f"✅ Shadow replay validation completed in {validation_duration:.1f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Shadow replay validation failed: {e}")
            return {'error': str(e)}

# Utility functions for standalone execution
def create_test_shadow_validator() -> ShadowReplayValidator:
    """
    Create test shadow replay validator with synthetic data.
    
    Returns:
        ShadowReplayValidator for testing
    """
    # Create synthetic data adapter (similar to dry_run_validator)
    from dry_run_validator import create_test_data_adapter
    
    data_adapter = create_test_data_adapter()
    
    validator = ShadowReplayValidator(
        data_adapter=data_adapter,
        validation_days=1,  # Shorter for testing
        database_path="test_shadow_replay.db",
        enable_compression=True,
        verbose=True
    )
    
    return validator

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🎬 STAIRWAYS TO HEAVEN V3 - SHADOW REPLAY VALIDATION")
    logger.info("=" * 60)
    
    # Create test validator
    print("🔧 Creating shadow replay validator...")
    validator = create_test_shadow_validator()
    
    # Run complete validation
    print("🎬 Running complete shadow replay validation...")
    report = validator.run_complete_shadow_validation(
        test_seeds=[42, 123],  # Small number for testing
        num_replays=2
    )
    
    # Display summary results
    if 'error' not in report:
        print("\n" + "=" * 60)
        print("📋 SHADOW REPLAY VALIDATION SUMMARY")
        print("=" * 60)
        
        validation_summary = report['validation_summary']
        deterministic_validation = report['deterministic_validation']
        correlation_analysis = report['correlation_analysis']
        
        print(f"Total Validations: {validation_summary['total_validations']}")
        print(f"Pass Rate: {validation_summary['pass_rate']:.1%}")
        print(f"Deterministic Rate: {deterministic_validation['hash_match_rate']:.1%}")
        print(f"Avg Portfolio Correlation: {correlation_analysis['avg_portfolio_correlation']:.3f}")
        
        print(f"\nKey Findings:")
        for finding in report['key_findings']:
            print(f"  {finding}")
        
        print(f"\nRecommendations:")
        for recommendation in report['recommendations']:
            print(f"  {recommendation}")
        
        print(f"\n✅ Shadow replay validation completed successfully!")
    else:
        print(f"❌ Validation failed: {report['error']}")
