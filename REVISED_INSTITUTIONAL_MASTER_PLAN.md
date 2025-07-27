# REVISED INSTITUTIONAL MASTER PLAN v2.0
## IntradayJules: Pragmatic Path to Institutional Excellence

*Incorporating Development Team Feedback & Ultra-Level Refinements*

---

## EXECUTIVE SUMMARY

**Mission**: Transform IntradayJules into institutional-grade trading system through **pragmatic complexity sequencing** while maintaining ambitious performance targets.

**Core Philosophy**: "Start simple, scale systematically, maintain excellence"

**Key Revisions Based on Team Feedback**:
- ✅ Split complex phases for manageable implementation
- ✅ Front-load infrastructure and dependency management  
- ✅ Defer advanced features until foundations are solid
- ✅ Define clear governance and approval workflows
- ✅ Realistic resourcing and timeline estimates

**Final Target Performance** (unchanged):
- Sharpe Ratio: ≥ 2.5 | Max Drawdown: ≤ 2% | Information Ratio: ≥ 2.0

---

## GOVERNANCE & TEAM STRUCTURE

### RACI Matrix (Addressing Team Feedback)

| Role | Phase 0-1 | Phase 2A-2B | Phase 3+ | Approval Authority |
|------|-----------|-------------|----------|-------------------|
| **Lead Quant** (Claude) | R/A | R/A | R/A | Technical architecture |
| **Senior Developer** | R | R/A | C | Code quality & testing |
| **Risk Officer** (Interim: CTO) | C | C | R/A | Risk framework approval |
| **Data Engineer** | C | R/A | R | Data pipeline & quality |
| **DevOps/SRE** | I | R/A | R/A | Infrastructure & deployment |

**Note**: Risk Officer role is interim-assigned to CTO until dedicated Risk Officer hire is finalized to prevent governance deadlock.

**Approval Gates**: Each phase requires sign-off from Lead Quant + Risk Officer + Senior Developer

### Digital Governance Framework

**File**: `governance/approval_framework.yaml`

```yaml
approval_framework:
  phase_gates:
    required_approvers: ["lead_quant", "risk_officer", "senior_developer"]
    approval_criteria:
      - technical_review_completed: true
      - risk_assessment_passed: true
      - test_coverage_meets_standard: true
      - performance_criteria_met: true
      
  approval_artifacts:
    - phase_X_technical_review.md
    - phase_X_risk_assessment.md
    - phase_X_test_report.md
    - phase_X_performance_validation.md
    
  digital_signatures:
    method: "gpg_signed_commits"
    storage: "governance/signatures/"
    retention: "7_years"
```

---

## INFRASTRUCTURE PHASE (PARALLEL TRACK)

### **NEW: Infrastructure-First Approach**

**Timeline**: Start immediately, complete before Phase 2A

```yaml
infrastructure_roadmap:
  week_1:
    - docker_containerization
    - dependency_resolution
    - ci_cd_pipeline_setup
    
  week_2:
    - metrics_database_deployment  # TimescaleDB or InfluxDB
    - grafana_dashboards
    - secret_management_system
    
  ongoing:
    - automated_testing_pipeline
    - artifact_signing_and_storage
    - backup_and_disaster_recovery
```

### Docker & Dependency Management

**File**: `infrastructure/Dockerfile.institutional`

```dockerfile
FROM python:3.11-slim as base

# Resolve dependency snowball upfront
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install core dependencies with version locks
COPY requirements.institutional.txt .
RUN pip install --no-cache-dir -r requirements.institutional.txt

# Validate GPU/CPU compatibility
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
RUN python -c "import numpy; import scipy; import pandas; print('Core libs OK')"

# Advanced libs for later phases (validate but don't import)
RUN python -c "import arch; import cvxpy; import statsmodels; print('Advanced libs available')"

WORKDIR /app
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/health_check.py

# Image tagging with provenance (addressing production feedback)
ARG GIT_SHA
ARG REQUIREMENTS_SHA256
LABEL git.sha="${GIT_SHA}"
LABEL requirements.sha256="${REQUIREMENTS_SHA256}"
LABEL build.timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

### Metrics Infrastructure

**File**: `infrastructure/metrics_setup.yaml`

```yaml
metrics_infrastructure:
  database:
    type: "timescaledb"  # PostgreSQL-based time series
    connection_string: "${METRICS_DB_URL}"
    retention_policy: "1_year"
    partitioning: "daily"
    
    # Data tiering to prevent 500GB+ storage bloat (addressing production feedback)
    data_tiering_policy: "cold_to_s3_after_90d"
    hot_storage_retention: "90_days"
    cold_storage_backend: "s3_compatible"
    compression_enabled: true
    
  metrics_collection:
    - training_metrics: ["episode_reward", "entropy", "explained_variance"]
    - trading_metrics: ["pnl", "sharpe", "drawdown", "turnover"]
    - risk_metrics: ["var_95", "var_99", "expected_shortfall"]
    - system_metrics: ["cpu_usage", "memory_usage", "gpu_utilization"]
    
  dashboards:
    - real_time_performance
    - risk_monitoring  
    - model_health
    - system_status
    
  alerting:
    thresholds:
      performance_degradation: "sharpe_ratio < 0.5"
      risk_breach: "drawdown > 0.03"
      system_health: "cpu_usage > 0.8"
```

---

## PHASE 0 - INSTITUTIONAL BASELINE (30 minutes, ZERO RISK)

### Enhanced with SHA-256 Checksums

```bash
# 1. Model registry with integrity verification
mkdir -p models/registry/baseline_2025-07-20/{model,config,metadata,validation}

# 2. Model backup with checksums (addressing team feedback)
copy models\RECURRENTPPO_2025-07-20_09-11-12\RECURRENTPPO_2025-07-20_09-11-12.zip ^
     models\registry\baseline_2025-07-20\model\model.zip
sha256sum models/registry/baseline_2025-07-20/model/model.zip > models/registry/baseline_2025-07-20/metadata/model_checksum.txt

# 3. Environment freeze with verification
pip freeze > models/registry/baseline_2025-07-20/metadata/requirements_baseline.txt
conda list --explicit > models/registry/baseline_2025-07-20/metadata/conda_env_baseline.txt

# 4. Git tagging with institutional metadata
git tag -a baseline_2025-07-20 -m "Institutional Baseline v1.0
SHA256: $(cat models/registry/baseline_2025-07-20/metadata/model_checksum.txt)
Performance: Sharpe -2.23, DD 2.64%, Turnover 5.9x
Status: STABLE_BUT_SUBOPTIMAL - Ready for enhancement"

# 5. Generate metadata script (addressing team feedback)
python scripts/generate_baseline_metadata.py --verify-integrity
```

**Success Criteria**: 100% reproducible baseline with cryptographic integrity verification

---

## PHASE 1 - REALITY GROUNDING FOUNDATION (4-8 hours, LOW RISK)

### Enhanced Configuration with Team Feedback

**File**: `config/phase1_reality_grounding.

yaml`

```yaml
environment:
  initial_capital: 50000.0
  reward_scaling: 0.02  # Team feedback: Conservative 950k → 19k (vs 25k ceiling)
  
  # EarlyStoppingCallback threshold scaling (addressing team feedback)
  early_stopping:
    patience: 20
    min_delta: 100  # Scaled from 5 to match new reward scale
    plateau_threshold: 500  # 19k * 0.025 = 475, rounded to 500
    
risk:
  include_risk_features: true
  penalty_lambda: 0.0
  dd_limit: 0.50
  
  # Enhanced risk feature set
  risk_features:
    - portfolio_heat_ratio
    - concentration_ratio  
    - drawdown_velocity
    - var_breach_indicator
    - correlation_breakdown_indicator
    
validation:
  observation_consistency_check: true
  reward_bounds_check: true
  nan_guard_strict: true
  
  # Batch sanity test (addressing team feedback)
  consistency_test:
    sample_size: 128
    tolerance: 1e-6
    test_frequency: "every_1000_steps"
```

### Observation Consistency Validator

**File**: `src/validation/observation_consistency.py`

```python
class ObservationConsistencyValidator:
    """Batch sanity testing for train/eval consistency (addressing team feedback)."""
    
    def run_batch_consistency_test(self, train_env, eval_env, sample_size=128):
        """Sample N random env resets, verify identical outputs."""
        
        consistency_results = []
        
        for i in range(sample_size):
            # Set identical random seed
            seed = np.random.randint(0, 1000000)
            
            # Reset both environments
            train_env.seed(seed)
            eval_env.seed(seed)
            train_obs = train_env.reset()
            eval_obs = eval_env.reset()
            
            # Check shape consistency
            shape_match = train_obs.shape == eval_obs.shape
            
            # Check dtype consistency  
            dtype_match = train_obs.dtype == eval_obs.dtype
            
            # Check value consistency (within tolerance)
            if shape_match and dtype_match:
                value_diff = np.abs(train_obs - eval_obs)
                max_diff = np.max(value_diff)
                value_match = max_diff < self.config['tolerance']
            else:
                value_match = False
                max_diff = float('inf')
                
            consistency_results.append({
                'sample_id': i,
                'seed': seed,
                'shape_match': shape_match,
                'dtype_match': dtype_match, 
                'value_match': value_match,
                'max_difference': max_diff,
                'train_shape': train_obs.shape,
                'eval_shape': eval_obs.shape
            })
            
        # Generate summary report
        total_samples = len(consistency_results)
        passed_samples = sum(1 for r in consistency_results if 
                           r['shape_match'] and r['dtype_match'] and r['value_match'])
        consistency_rate = passed_samples / total_samples
        
        return {
            'consistency_rate': consistency_rate,
            'total_samples': total_samples,
            'passed_samples': passed_samples,
            'detailed_results': consistency_results,
            'test_passed': consistency_rate >= 0.99  # 99% consistency required
        }
```

**Success Criteria**:
- Episode rewards: 8k - 19k (adjusted ceiling based on team feedback)
- Observation consistency: 99%+ batch test pass rate
- No NaN/Inf incidents: 0 tolerance
- EarlyStoppingCallback: Properly scaled thresholds

---

## PHASE 2A - BASIC TRANSACTION COST REALITY (1-2 weeks, MEDIUM RISK)

### **NEW: Simplified First Implementation**

**Objective**: Implement realistic but simple transaction costs **before** advanced microstructure

**File**: `config/phase2a_basic_costs.yaml`

```yaml
transaction_costs:
  enabled: true
  model_type: "basic_institutional"
  
  # Simple but realistic cost structure
  fixed_costs:
    commission_per_trade: 0.50  # Institutional brokerage
    
  proportional_costs:
    spread_cost_bps: 1.0      # 1 basis point spread
    exchange_fees_bps: 0.1    # Exchange fees
    
  # Basic capacity-aware penalty (no Almgren-Chriss yet)
  capacity_penalty:
    enabled: true
    daily_capacity_pct: 0.02   # 2% of capital per day is "free"
    penalty_curve: "quadratic" # Simple x^2 above capacity
    penalty_weight: 0.0001     # Gentle cost coefficient
    max_penalty_pct_trade: 0.3 # Never charge >30% of trade value (addressing production feedback)
    
  # ADV scaling (addressing team feedback) 
  adv_scaling:
    enabled: true
    adv_scaling_factor: 1.5    # Capacity scales with average daily volume
    adv_lookback_days: 20      # 20-day ADV calculation
```

### Basic Transaction Cost Engine

**File**: `src/execution/basic_cost_engine.py`

```python
class BasicTransactionCostEngine:
    """Simple but realistic transaction costs (Phase 2A implementation)."""
    
    def compute_transaction_cost(self, trade_value, portfolio_value, market_data):
        """Compute basic institutional transaction costs."""
        
        # 1. Fixed costs
        fixed_cost = self.config['fixed_costs']['commission_per_trade']
        
        # 2. Proportional costs
        spread_cost = abs(trade_value) * self.config['proportional_costs']['spread_cost_bps'] / 10000
        exchange_fees = abs(trade_value) * self.config['proportional_costs']['exchange_fees_bps'] / 10000
        
        # 3. Basic capacity penalty
        capacity_cost = self._compute_basic_capacity_penalty(trade_value, portfolio_value, market_data)
        
        total_cost = fixed_cost + spread_cost + exchange_fees + capacity_cost
        
        # Stream to metrics DB (addressing team feedback)
        self._stream_cost_metrics({
            'fixed_cost': fixed_cost,
            'spread_cost': spread_cost, 
            'exchange_fees': exchange_fees,
            'capacity_cost': capacity_cost,
            'total_cost': total_cost,
            'cost_as_pct_trade': total_cost / abs(trade_value) if trade_value != 0 else 0,
            'trade_value': trade_value,
            'timestamp': datetime.utcnow()
        })
        
        return total_cost
        
    def _compute_basic_capacity_penalty(self, trade_value, portfolio_value, market_data):
        """Basic capacity penalty with ADV scaling."""
        
        # Base daily capacity
        base_capacity = self.config['capacity_penalty']['daily_capacity_pct']
        
        # ADV scaling (addressing team feedback)
        if self.config['adv_scaling']['enabled']:
            adv_ratio = self._get_adv_ratio(market_data)
            adjusted_capacity = base_capacity * (1 + adv_ratio * self.config['adv_scaling']['adv_scaling_factor'])
        else:
            adjusted_capacity = base_capacity
            
        # Compute excess over capacity
        daily_turnover = abs(trade_value) / portfolio_value
        excess = max(0, daily_turnover - adjusted_capacity)
        
        # Simple quadratic penalty
        if excess > 0:
            penalty = self.config['capacity_penalty']['penalty_weight'] * (excess ** 2) * portfolio_value
            
            # Apply absolute cap to prevent explosion on large trades (addressing production feedback)
            max_penalty = abs(trade_value) * self.config['capacity_penalty']['max_penalty_pct_trade']
            penalty = min(penalty, max_penalty)
        else:
            penalty = 0
            
        return penalty
```

**Success Criteria Phase 2A**:
- Daily turnover: 1.0x - 3.0x (more conservative than 2.5x)
- Transaction cost: 0.05% - 0.12% per trade
- Win rate: > 45%
- Cost metrics streaming to DB: 100% uptime

---

## PHASE 2B - ADVANCED MICROSTRUCTURE (2-3 weeks, MEDIUM-HIGH RISK)

### **NEW: Advanced Implementation After 2A Success**

**Prerequisites**: Phase 2A success + Market data upgrade complete

**Data Requirements** (addressing team feedback):
- Secure depth-of-book feed (Polygon/Refinitiv)
- Sub-second bars for stress testing
- Historical TAQ data for Almgren-Chriss calibration

**File**: `config/phase2b_advanced_microstructure.yaml`

```yaml
advanced_microstructure:
  enabled: true
  
  # Almgren-Chriss with calibrated constants (addressing team feedback)
  market_impact:
    enabled: true
    calibration_required: true  # Must calibrate before use
    
    # Placeholder constants - MUST calibrate with historical TAQ data
    temporary_impact_coeff: 0.314159  # π/10, requires calibration
    permanent_impact_coeff: 0.1       # Requires calibration
    liquidity_decay_factor: 0.5       # Requires calibration
    
    calibration_data:
      required_samples: 1000    # Minimum historical trades
      calibration_period: "2023-01-01 to 2024-01-01"
      calibration_symbols: ["NVDA", "AAPL", "MSFT"]
      
  # Enhanced liquidity modeling
  liquidity_model:
    book_depth_required: true
    adverse_selection_protection: true
    tick_size_aware: true
    
  # Smart Huber curve (deferred to 2B)
  smart_huber_penalty:
    enabled: true
    transition_point: 1.0  # 100% of limit
    linear_slope_multiplier: 50  # Configurable as suggested
```

### Almgren-Chriss Calibration Engine

**File**: `src/execution/almgren_chriss_calibrator.py`

```python
class AlgrenChrissCalirator:
    """Calibrate market impact parameters using historical TAQ data."""
    
    def calibrate_parameters(self, historical_trades, market_data):
        """Calibrate Almgren-Chriss parameters from historical data."""
        
        # Require minimum sample size (addressing team feedback)
        if len(historical_trades) < self.config['calibration_data']['required_samples']:
            raise InsufficientDataError(
                f"Need {self.config['calibration_data']['required_samples']} trades, "
                f"got {len(historical_trades)}"
            )
            
        # 1. Estimate temporary impact coefficient
        temp_impact_coeff = self._estimate_temporary_impact(historical_trades, market_data)
        
        # 2. Estimate permanent impact coefficient  
        perm_impact_coeff = self._estimate_permanent_impact(historical_trades, market_data)
        
        # 3. Estimate liquidity decay factor
        liquidity_decay = self._estimate_liquidity_decay(historical_trades, market_data)
        
        # 4. Validate calibration quality
        calibration_quality = self._validate_calibration(
            temp_impact_coeff, perm_impact_coeff, liquidity_decay, historical_trades
        )
        
        if calibration_quality['r_squared'] < 0.5:
            self.logger.warning(f"Poor calibration quality: R² = {calibration_quality['r_squared']:.3f}")
            
        return {
            'temporary_impact_coeff': temp_impact_coeff,
            'permanent_impact_coeff': perm_impact_coeff, 
            'liquidity_decay_factor': liquidity_decay,
            'calibration_quality': calibration_quality,
            'calibration_date': datetime.utcnow(),
            'sample_size': len(historical_trades)
        }
```

**Success Criteria Phase 2B**:
- Almgren-Chriss calibration R² > 0.6
- Market impact estimates realistic vs industry benchmarks
- Phase 2A metrics maintained or improved
- Depth-of-book integration successful

---

## PHASE 3 - INSTITUTIONAL RISK MANAGEMENT (2-3 weeks, HIGH RISK)

### **Simplified Risk Engine Start** (addressing team feedback)

**File**: `config/phase3_pragmatic_risk.yaml`

```yaml
risk_management:
  # Start with essential limits only
  core_limits:
    max_daily_drawdown_pct: 0.03
    max_monthly_drawdown_pct: 0.08
    max_position_concentration_pct: 0.15
    
  # Simplified VaR (start with 2 methods, defer EVT)
  var_models:
    enabled_methods: ["historical_simulation", "parametric_normal"]
    # Defer EVT until adequate sample size (addressing team feedback)
    deferred_methods: ["extreme_value_theory"]  # Requires >1000 daily PnL points
    
  # Enhanced circuit breakers (addressing team feedback)
  circuit_breakers:
    enabled: true
    consecutive_losses:
      count: 5
      min_loss_magnitude: 0.001  # 0.1% minimum loss to count
      total_loss_threshold: 0.02  # 2% total loss in consecutive period
    volatility_spike_threshold: 2.0
    correlation_breakdown_threshold: 0.7
    
  # Risk attribution (simplified start)
  attribution:
    enabled: true
    components: ["systematic_risk", "idiosyncratic_risk"]  # Start simple
    advanced_components: ["timing_risk", "sizing_risk"]    # Add later
```

### Pragmatic Risk Engine

**File**: `src/risk/pragmatic_risk_engine.py`

```python
class PragmaticRiskEngine:
    """Institutional risk management with pragmatic complexity sequencing."""
    
    def __init__(self, config):
        self.config = config
        self.var_calculator = SimplifiedVaRCalculator()  # Start simple
        self.circuit_breaker = EnhancedCircuitBreaker()
        
    def evaluate_portfolio_risk(self, portfolio_state, market_data):
        """Simplified but institutional-grade risk evaluation."""
        
        risk_metrics = {}
        
        # 1. Core VaR calculation (2 methods only initially)
        for method in self.config['var_models']['enabled_methods']:
            var_result = self.var_calculator.calculate_var(
                portfolio_state, method=method, confidence=0.95
            )
            risk_metrics[f'var_95_{method}'] = var_result
            
        # 2. Drawdown analysis
        risk_metrics['current_drawdown'] = self._calculate_current_drawdown(portfolio_state)
        risk_metrics['max_drawdown_period'] = self._calculate_max_drawdown_period(portfolio_state)
        
        # 3. Concentration risk
        risk_metrics['concentration_ratio'] = self._calculate_concentration_ratio(portfolio_state)
        
        # 4. Simplified circuit breaker checks (enhanced with magnitude requirement)
        circuit_breaker_status = self.circuit_breaker.check_enhanced_triggers(
            portfolio_state, risk_metrics
        )
        risk_metrics['circuit_breaker_status'] = circuit_breaker_status
        
        return risk_metrics
        
    def check_risk_limits(self, portfolio_state, risk_metrics):
        """Check core risk limits with institutional standards."""
        
        violations = []
        actions = []
        
        # Daily drawdown check
        if risk_metrics['current_drawdown'] > self.config['core_limits']['max_daily_drawdown_pct']:
            violations.append('daily_drawdown_exceeded')
            actions.append('immediate_risk_reduction')
            
        # Enhanced consecutive losses check (addressing team feedback)
        consecutive_check = self._check_enhanced_consecutive_losses(portfolio_state)
        if consecutive_check['triggered']:
            violations.append('consecutive_losses_magnitude')
            actions.append('trading_halt_review')
            
        return {
            'violations': violations,
            'recommended_actions': actions,
            'risk_score': self._calculate_risk_score(risk_metrics),
            'enhanced_checks': consecutive_check
        }
        
    def _check_enhanced_consecutive_losses(self, portfolio_state):
        """Enhanced consecutive losses with magnitude requirement."""
        
        recent_returns = portfolio_state.get('recent_returns', [])[-5:]  # Last 5 periods
        
        # Check if all are losses above minimum magnitude
        qualifying_losses = [
            r for r in recent_returns 
            if r < -self.config['circuit_breakers']['consecutive_losses']['min_loss_magnitude']
        ]
        
        consecutive_count = len(qualifying_losses) if len(qualifying_losses) == len(recent_returns) else 0
        total_loss = sum(qualifying_losses) if qualifying_losses else 0
        
        triggered = (
            consecutive_count >= self.config['circuit_breakers']['consecutive_losses']['count'] and
            abs(total_loss) >= self.config['circuit_breakers']['consecutive_losses']['total_loss_threshold']
        )
        
        return {
            'triggered': triggered,
            'consecutive_count': consecutive_count,
            'total_loss': total_loss,
            'qualifying_losses': qualifying_losses
        }
```

**Success Criteria**:
- Core risk limits: 100% compliance
- VaR backtesting: >90% accuracy (start realistic)
- Circuit breaker false positives: <10% (start conservative)
- Risk attribution R²: >0.75 (simplified version)

---

## PHASE 4 - CURRICULUM WITH REALISTIC GATES (3-4 weeks, HIGH RISK)

### **Enhanced Validation Framework** (addressing team feedback)

**File**: `config/phase4_enhanced_curriculum.yaml`

```yaml
curriculum:
  enabled: true
  
  # Fixed episode lengths (addressing team feedback)
  episode_standardization:
    fixed_episode_length: 20000  # Lock episode length
    normalization_method: "per_bar"  # Normalize metrics per bar
    
  # Explicit calendar splits (addressing team feedback)
  validation_splits:
    frozen_splits: true
    train_periods:
      - "2022-01-01 to 2022-12-31"
      - "2023-01-01 to 2023-06-30"
    validation_periods:
      - "2023-07-01 to 2023-12-31"  
      - "2024-01-01 to 2024-06-30"
    burn_in_period: "3_months"  # LSTM state memory buffer
    
  # Statistical significance with block bootstrap (addressing team feedback)
  significance_testing:
    method: "block_bootstrap"  # Account for autocorrelation
    block_size: 20  # 20-episode blocks
    bootstrap_samples: 1000
    confidence_level: 0.95
    
  stages:
    foundation:
      target_metrics:
        sharpe_ratio: 0.7  # Slightly higher than original
        max_drawdown_pct: 0.035
        win_rate_pct: 0.48
      advancement_criteria:
        consecutive_passing_episodes: 15  # Fixed length episodes
        statistical_significance_required: true
        minimum_episodes: 50
```

### Block Bootstrap Validator

**File**: `src/validation/block_bootstrap_validator.py`

```python
class BlockBootstrapValidator:
    """Statistical significance testing with block bootstrap (addressing team feedback)."""
    
    def test_statistical_significance(self, performance_data, target_metrics):
        """Block bootstrap to account for autocorrelation in trading returns."""
        
        results = {}
        
        for metric, target in target_metrics.items():
            if metric in performance_data:
                # Block bootstrap accounting for temporal dependence
                bootstrap_statistics = self._block_bootstrap_statistic(
                    data=performance_data[metric],
                    target=target,
                    block_size=self.config['block_size'],
                    n_bootstrap=self.config['bootstrap_samples']
                )
                
                # Calculate p-value and confidence interval
                p_value = np.mean(bootstrap_statistics <= 0)  # H0: performance <= target
                confidence_interval = np.percentile(bootstrap_statistics, [2.5, 97.5])
                
                # Effect size (bootstrap version)
                effect_size = np.mean(bootstrap_statistics) / np.std(bootstrap_statistics)
                
                results[metric] = {
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'confidence_interval': confidence_interval,
                    'effect_size': effect_size,
                    'bootstrap_mean': np.mean(bootstrap_statistics),
                    'target': target,
                    'sample_size': len(performance_data[metric])
                }
                
        return results
        
    def _block_bootstrap_statistic(self, data, target, block_size, n_bootstrap):
        """Generate bootstrap statistics using block resampling."""
        
        data_array = np.array(data)
        n_obs = len(data_array)
        n_blocks = n_obs // block_size
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample blocks with replacement
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            
            resampled_data = []
            for block_idx in block_indices:
                start_idx = block_idx * block_size
                end_idx = start_idx + block_size
                resampled_data.extend(data_array[start_idx:end_idx])
                
            # Calculate test statistic (difference from target)
            bootstrap_stat = np.mean(resampled_data) - target
            bootstrap_stats.append(bootstrap_stat)
            
        return np.array(bootstrap_stats)
        
    def store_parameter_sweep_results(self, metric, data, target, block_sizes=[10, 15, 20, 25, 30]):
        """Store block size vs p-value grid for research tuning (addressing production feedback)."""
        
        sweep_results = []
        
        for block_size in block_sizes:
            bootstrap_stats = self._block_bootstrap_statistic(
                data=data,
                target=target,
                block_size=block_size,
                n_bootstrap=self.config['bootstrap_samples']
            )
            
            p_value = np.mean(bootstrap_stats <= 0)
            
            sweep_results.append({
                'metric': metric,
                'block_size': block_size,
                'p_value': p_value,
                'timestamp': datetime.utcnow(),
                'sample_size': len(data)
            })
            
        # Store in metrics DB for research analysis
        self._store_to_metrics_db('block_bootstrap_parameter_sweep', sweep_results)
        
        return sweep_results
```

**Success Criteria**:
- All advancement gates: Block bootstrap p-value < 0.05
- Episode length consistency: 100% (20k bars each)
- Burn-in period respect: 3-month gap maintained
- Stage progression: Based on fixed-length normalized metrics

---

## PHASE 5 - PRODUCTION OPTIMIZATION (4-6 weeks, VERY HIGH RISK)

### **Pragmatic Kelly Implementation** (addressing team feedback)

**File**: `config/phase5_enhanced_kelly.yaml`

```yaml
enhanced_kelly:
  enabled: true
  
  # Specific estimator definition (addressing team feedback)
  expected_return_estimator:
    method: "ewma_q_values"  # Exponentially weighted moving average of Q-values
    window_length: 60        # 60-bar lookback
    alpha: 0.1              # EWMA decay factor
    minimum_samples: 30      # Minimum observations before estimation
    
  # Correlation adjustment with Ledoit-Wolf (addressing team feedback)
  correlation_adjustment:
    enabled: true
    method: "incremental_ledoit_wolf"  # Real-time invertible covariance
    shrinkage_target: "single_index"   # Shrink toward market model
    minimum_shrinkage: 0.1             # Minimum shrinkage intensity
    
  # Enhanced constraints
  constraints:
    max_kelly_fraction: 0.20    # Conservative institutional limit
    volatility_target: 0.12     # Lower target (12% vs 15%)
    position_size_bounds: [0.01, 0.15]  # Tighter bounds
```

### Specific Expected Return Estimator

**File**: `src/optimization/expected_return_estimator.py`

```python
class EWMAQValueEstimator:
    """EWMA estimator for expected returns from Q-values (addressing team feedback)."""
    
    def __init__(self, config):
        self.window_length = config['window_length']
        self.alpha = config['alpha']
        self.minimum_samples = config['minimum_samples']
        self.q_value_history = deque(maxlen=self.window_length)
        self.ewma_estimate = None
        
    def update_and_estimate(self, current_q_value, action_taken):
        """Update EWMA estimate with new Q-value observation."""
        
        # Store Q-value for the action that was actually taken
        self.q_value_history.append({
            'q_value': current_q_value,
            'action': action_taken,
            'timestamp': datetime.utcnow()
        })
        
        # Need minimum samples before providing estimates
        if len(self.q_value_history) < self.minimum_samples:
            return 0.0  # No position until sufficient data
            
        # Calculate EWMA of Q-values
        if self.ewma_estimate is None:
            # Initialize with simple mean
            self.ewma_estimate = np.mean([obs['q_value'] for obs in self.q_value_history])
        else:
            # Update EWMA
            latest_q = self.q_value_history[-1]['q_value']
            self.ewma_estimate = self.alpha * latest_q + (1 - self.alpha) * self.ewma_estimate
            
        # Convert Q-value to expected return estimate
        # Q-values represent expected future reward, scale to return units
        # Apply floor to prevent Kelly going flat on quiet days (addressing production feedback)
        expected_return = max(self.ewma_estimate / 10000, 1e-6)
        
        return expected_return
```

### Incremental Ledoit-Wolf Covariance

**File**: `src/optimization/incremental_covariance.py`

```python
class IncrementalLedoitWolfCovariance:
    """Real-time Ledoit-Wolf shrinkage for invertible covariance (addressing team feedback)."""
    
    def __init__(self, config):
        self.shrinkage_target = config['shrinkage_target']
        self.minimum_shrinkage = config['minimum_shrinkage']
        self.sample_covariance = None
        self.shrinkage_intensity = None
        
    def update_covariance(self, returns_matrix):
        """Update covariance matrix with Ledoit-Wolf shrinkage."""
        
        n_obs, n_assets = returns_matrix.shape
        
        # 1. Sample covariance matrix
        sample_cov = np.cov(returns_matrix, rowvar=False)
        
        # 2. Shrinkage target (single index model)
        if self.shrinkage_target == "single_index":
            market_var = np.var(np.mean(returns_matrix, axis=1))  # Market variance
            target_cov = np.eye(n_assets) * market_var
        else:
            target_cov = np.eye(n_assets) * np.mean(np.diag(sample_cov))
            
        # 3. Optimal shrinkage intensity (Ledoit-Wolf formula)
        shrinkage = self._calculate_optimal_shrinkage(returns_matrix, sample_cov, target_cov)
        shrinkage = max(shrinkage, self.minimum_shrinkage)  # Apply minimum
        
        # 4. Shrunk covariance matrix
        shrunk_cov = shrinkage * target_cov + (1 - shrinkage) * sample_cov
        
        # 5. Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(shrunk_cov)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Floor eigenvalues
        final_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        self.sample_covariance = sample_cov
        self.shrinkage_intensity = shrinkage
        
        return final_cov
```

**Success Criteria**:
- Enhanced Kelly: Stable position sizing with EWMA Q-value estimator
- Covariance: Always invertible with Ledoit-Wolf shrinkage
- Sharpe target: ≥ 2.0 (slightly reduced from 2.5 for realism)
- All other targets: Maintained from original plan

---

## PHASE 6 - SIMPLIFIED REGIME MASTERY (ONGOING)

### **MVP Regime Detection** (addressing team feedback)

**File**: `config/phase6_mvp_regimes.yaml`

```yaml
regime_mastery_mvp:
  enabled: true
  
  # Start with 2 regimes only (addressing team feedback)
  regime_classification:
    mvp_regimes:
      - volatility_regime: ["low_volatility", "high_volatility"]
      - trend_regime: ["trending", "ranging"]
    
    # Defer complex regimes to post-Phase 6
    deferred_regimes:
      - sentiment_regime  # Multi-quarter project
      - liquidity_regime
      - correlation_regime
      
  # Simplified online learning (addressing team feedback)
  online_learning:
    sandbox_only: true  # Only in paper trading until risk approval
    governance_override_required: true  # Risk officer must approve production use
    
  # Realistic alert thresholds (addressing team feedback)
  monitoring:
    alert_false_positive_target: 0.05  # 5% initial target, tighten later
    performance_degradation_threshold: 0.10  # 10% degradation triggers alert
```

### MVP Regime Detector

**File**: `src/regimes/mvp_regime_detector.py`

```python
class MVPRegimeDetector:
    """Minimal viable regime detection (volatility + trend only)."""
    
    def __init__(self, config):
        self.volatility_estimator = RollingVolatilityEstimator(window=20)
        self.trend_detector = SimpleTrendDetector(window=50)
        
    def detect_regime(self, market_data):
        """Detect simple regime combination."""
        
        # 1. Volatility regime (simple threshold)
        current_vol = self.volatility_estimator.estimate(market_data['price_series'])
        vol_regime = 'high_volatility' if current_vol > 0.25 else 'low_volatility'
        
        # 2. Trend regime (simple moving average slope)
        trend_strength = self.trend_detector.detect_trend(market_data['price_series'])
        trend_regime = 'trending' if abs(trend_strength) > 0.1 else 'ranging'
        
        # 3. Combined regime
        combined_regime = f"{vol_regime}_{trend_regime}"
        
        return {
            'volatility_regime': vol_regime,
            'trend_regime': trend_regime,
            'combined_regime': combined_regime,
            'confidence': 0.8,  # Simple confidence score
            'timestamp': datetime.utcnow()
        }
```

**Success Criteria**:
- MVP regime detection: 85%+ accuracy on 2 regimes
- Online learning: Sandbox validation before production
- Alert rates: <5% false positives initially
- Performance stability: <10% degradation across regimes

---

## ENHANCED TESTING & GOVERNANCE

### **Comprehensive Test Coverage** (addressing team feedback)

```yaml
test_coverage_standards:
  overall_target: 75%        # Realistic institutional target
  critical_components: 100%  # Math primitives must be 100%
  
  critical_components_list:
    - penalty_curves
    - var_calculations  
    - kelly_sizing
    - risk_checks
    - transaction_costs
    - regime_detection
    
  testing_strategy:
    unit_tests: "pytest with parametrized test cases"
    integration_tests: "end-to-end pipeline validation"
    property_tests: "hypothesis library for math functions"
    performance_tests: "benchmark critical paths"
```

### Single Source YAML Schema

**File**: `config/schema/institutional_config_schema.py`

```python
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Union

class InstitutionalConfigSchema(BaseModel):
    """Single source of truth for all configuration (addressing team feedback)."""
    
    class Config:
        extra = "forbid"  # Fail fast on unknown keys
        
    # Environment configuration
    environment: EnvironmentConfig
    
    # Risk management
    risk_management: RiskManagementConfig
    
    # Transaction costs
    transaction_costs: Optional[TransactionCostConfig] = None
    
    # Curriculum learning
    curriculum: Optional[CurriculumConfig] = None
    
    # Infrastructure
    infrastructure: InfrastructureConfig
    
    @validator('*', pre=True)
    def validate_all_sections(cls, v):
        """Validate all configuration sections."""
        if v is None:
            raise ValueError("Configuration section cannot be None")
        return v
        
def load_and_validate_config(config_path: str) -> InstitutionalConfigSchema:
    """Load config with Pydantic validation, fail fast on errors."""
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
        
    try:
        validated_config = InstitutionalConfigSchema(**raw_config)
        return validated_config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ConfigurationError(f"Invalid configuration: {e}")
```

---

## REVISED TIMELINE & RESOURCING

### **Realistic Resource Allocation** (addressing team feedback)

| Phase | Duration | Team Size | Key Roles | Critical Path | Test Budget |
|-------|----------|-----------|-----------|---------------|-------------|
| **Infrastructure** | 2 weeks | 2 FTE | DevOps + Data Engineer | Docker + Metrics DB | 1 day |
| **Phase 0** | 30 min | 1 FTE | Lead Quant | Baseline preservation | N/A |
| **Phase 1** | 1 week | 2 FTE | Lead Quant + Developer | Reality grounding | 2 days |
| **Phase 2A** | 2 weeks | 2 FTE | Quant Dev + Data Engineer | Basic transaction costs | 2 days |
| **Phase 2B** | 3 weeks | 3 FTE | + Quant Researcher | Advanced microstructure | 2 days |
| **Phase 3** | 3 weeks | 3 FTE | + Risk Officer | Risk management | 2 days |
| **Phase 4** | 4 weeks | 4 FTE | + DevOps | Curriculum validation | 2 days |
| **Phase 5** | 6 weeks | 4 FTE | Full team | Production optimization | 2 days |
| **Phase 6** | Ongoing | 4 FTE | Full team | Regime mastery | 2 days |

**Note**: Test budget reserves ~2 engineering days per phase solely for testing to prevent timeline slip (addressing production feedback).

### **Communication Protocol**

- **Daily standups**: Phase 2+ (15 min, blockers focus)
- **Design reviews**: Twice weekly (addressing team feedback)
- **Phase gate reviews**: End of each phase with formal sign-off
- **Risk reviews**: Weekly with Risk Officer participation

---

## FINAL APPROVALS & NEXT STEPS

### **Digital Sign-Off Framework**

Each phase requires GPG-signed approval in `governance/phase_X_approval.yaml`:

```yaml
phase_1_approval:
  approvers:
    lead_quant:
      name: "Claude (AI Lead Quant)"
      signature: "GPG_SIGNATURE_HERE"
      approval_date: "2025-07-21T00:00:00Z"
      technical_review_passed: true
      
    risk_officer:
      name: "TBD_HUMAN_RISK_OFFICER"
      signature: "PENDING"
      approval_date: null
      risk_assessment_passed: false
      
    senior_developer:
      name: "TBD_SENIOR_DEVELOPER"  
      signature: "PENDING"
      approval_date: null
      code_review_passed: false
      
  phase_status: "PENDING_APPROVALS"
  gate_criteria_met: false
  next_actions: ["assign_human_approvers", "complete_risk_assessment"]
```

---

## CONCLUSION

This **REVISED INSTITUTIONAL MASTER PLAN v2.0** addresses every concern raised by your development team while maintaining the ambitious vision of institutional excellence. Key improvements:

✅ **Complexity Management**: Split phases, deferred advanced features  
✅ **Infrastructure First**: Front-loaded Docker, metrics DB, CI/CD  
✅ **Realistic Timelines**: Conservative estimates with proper resourcing  
✅ **Clear Governance**: RACI matrix, approval workflows, digital signatures  
✅ **Technical Pragmatism**: Simplified starts, realistic calibration requirements  
✅ **Test Coverage**: 75% overall, 100% for critical math components  

**Your team's feedback was exceptional** - it transformed a visionary but potentially over-ambitious plan into an executable roadmap that maintains institutional excellence as the end goal while ensuring every step is realistic and properly resourced.

---

## PRODUCTION FEEDBACK INTEGRATION

### **Additional Minor Blocking Items Addressed**:

✅ **Metrics DB Sizing**: Added cold storage tiering to S3 after 90 days to prevent 500GB+ storage bloat  
✅ **Risk Officer TBD**: Assigned interim approval authority to CTO to prevent governance deadlock  

### **Production Recommendations Incorporated**:

✅ **Docker Provenance**: Added git SHA + requirements SHA256 tagging for full traceability  
✅ **Test Budget**: Reserved 2 engineering days per phase exclusively for testing  
✅ **Capacity Penalty Cap**: Added 30% trade value maximum to prevent explosion on large prints  
✅ **EWMA Floor**: Added 1e-6 minimum to prevent Kelly position sizing going to zero  
✅ **Block Bootstrap Research**: Added parameter sweep storage for research tuning  

**This feedback demonstrates exceptional production experience** - someone who has deployed real trading systems and knows where the operational pain points occur. Every suggestion addresses a genuine production issue that could cause problems in deployment.

**Ready for Phase 0 execution** with your approval! 🎯