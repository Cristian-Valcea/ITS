# Project: Intraday Python Trading System

## Context
- Language: Python 3.10+
- Libraries used: ccxt, pandas, asyncio, pytest
- Modules: trade_engine.py, signal_generator.py, order_manager.py, trainer_core.py, policy_export.py
- Goal: Execute intraday trades with latency < 50ms, maintain ≥ 85% trade success rate
## Deployment
- CI pipeline: GitHub Actions → Run pytest, lint with Ruff, build TorchScript `policy_export.py`
- Deploy agents via Kubernetes CronJobs
## Observability
- All logs via `loguru`, JSON formatted
- Monitor latency in `latency_monitor.py`, expose via Prometheus


## Coding Conventions
- Type hints & docstrings for all public APIs
- Black + Ruff formatting
- loguru for logging, structured & async-safe
- CI: pytest with coverage ≥ 85%
- Follow PEP8 naming standards

## Project Structure

C:\Projects\IntradayJules\src
¦   
¦   column_names.py
¦   main.py
¦   
+---agents
¦      base_agent.py
¦      data_agent.py
¦      data_agent_fixed.py
¦      enhanced_data_agent.py
¦      env_agent.py
¦      evaluator_agent.py
¦      feature_agent.py
¦      orchestrator_agent.py
¦      risk_agent.py
¦      trainer_agent.py
¦      __init__.py
¦      
¦           
+---ai_agents
¦      dqn_data_agent_system.py
¦      
¦           
+---api
¦   ¦   config_models.py
¦   ¦   fee_endpoints.py
¦   ¦   main.py
¦   ¦   monitoring_endpoints.py
¦   ¦   request_models.py
¦   ¦   response_models.py
¦   ¦   services.py
¦   ¦   stress_endpoints.py
¦   ¦   __init__.py
¦   ¦   
¦   +---templates
¦          config_editor.html
¦          dashboard.html
¦          live_trading_simple.html
¦          nvda_dqn_training.html
¦          placeholder.placeholder
¦          run_training.html
¦          task_status.html
¦          
¦           
+---backtesting
¦      bias_free_backtester.py
¦      
¦           
+---batch
¦   ¦   end_of_day_processor.py
¦   ¦   __init__.py
¦   ¦   
¦           
+---cli
¦       model_cli.py
¦       
+---compliance
¦      mifid_ii_exporter.py
¦      __init__.py
¦      
¦           
+---data
¦   ¦   bias_aware_data_agent.py
¦   ¦   crsp_delisting_integration.py
¦   ¦   fx_lifecycle.py
¦   ¦   production_survivorship_pipeline.py
¦   ¦   survivorship_bias_handler.py
¦   ¦   
¦   ¦           
¦           
+---deployment
¦   ¦   blue_green_rollout.py
¦   ¦   model_deployment_service.py
¦   ¦   
¦           
+---evaluation
¦   ¦   backtest_runner.py
¦   ¦   metrics_calculator.py
¦   ¦   model_loader.py
¦   ¦   performance_visualizer.py
¦   ¦   report_generator.py
¦   ¦   __init__.py
¦   ¦   
¦           
+---execution
¦   ¦   execution_agent_stub.py
¦   ¦   orchestrator_agent.py
¦   ¦   order_throttling.py
¦   ¦   throttled_execution_agent.py
¦   ¦   __init__.py
¦   ¦   
¦   +---core
¦   ¦      execution_loop.py
¦   ¦      high_perf_audit.py
¦   ¦      latency_monitor.py
¦   ¦      live_data_loader.py
¦   ¦      order_router.py
¦   ¦      pnl_tracker.py
¦   ¦      risk_callbacks.py
¦   ¦      __init__.py
¦         
¦      
+---features
¦      base_calculator.py
¦      config_validator.py
¦      data_processor.py
¦      ema_calculator.py
¦      feature_manager.py
¦      feature_registry.py
¦      market_impact_calculator.py
¦      performance_tracker.py
¦      rsi_calculator.py
¦      time_calculator.py
¦      vwap_calculator.py
¦      __init__.py
¦      
+---governance
¦   ¦   audit_immutable.py
¦   ¦   integration.py
¦   ¦   model_lineage.py
¦   ¦   release_approval.py
¦   ¦   __init__.py
¦   ¦   
¦           
+---graph_ai_agents
¦   ¦   data_provisioning_intraday.py
¦   ¦   orchestrator_data_provisioning.py
¦   ¦   
¦           
+---gym_env
¦   ¦   intraday_trading_env.py
¦   ¦   kyle_lambda_fill_simulator.py
¦   ¦   __init__.py
¦   ¦   
¦           
+---llm_config
¦   ¦   config.py
¦   ¦   
¦   +---shared_work_dir
¦   ¦   +---data
¦   ¦   +---strategies
¦           
+---logs
¦   ¦   episode_summaries.csv
¦   ¦   orchestrator_gpu_fixed.log
¦   ¦   
¦   +---audit_hiperf
¦   ¦       audit_emergency_20250713_184941.bin
¦   ¦       audit_main_20250713_184941.bin
¦   ¦       
¦   +---run_metadata
¦   ¦       run_metadata_episode_1752491336.json
¦   ¦       
¦   +---tensorboard
¦       +---DQN_2025-07-13_18-11-17_1
¦       ¦       events.out.tfevents.1752419479.Dell-XPS-Cris.44716.0
¦       +---monitor_logs
¦               monitor_20250713_181006.csv.monitor.csv
¦               
+---models
¦   +---DQN_2025-07-13_18-10-06
¦   +---DQN_2025-07-13_18-11-17
¦   ¦   ¦   DQN_2025-07-13_18-11-17.zip
¦   ¦   ¦   DQN_2025-07-13_18-11-17_metadata.json
¦   ¦   ¦   DQN_2025-07-13_18-11-17_torchscript.pt
¦   ¦   ¦   policy.pt
¦   ¦   ¦   
¦   ¦   +---checkpoints
¦   +---DQN_2025-07-13_18-14-38
¦   ¦   ¦   DQN_2025-07-13_18-14-38.zip
¦   ¦   ¦   DQN_2025-07-13_18-14-38_metadata.json
¦   ¦   ¦   DQN_2025-07-13_18-14-38_torchscript.pt
¦   ¦   ¦   policy.pt
¦   ¦   ¦   
¦               
+---monitoring
¦   +---alerting
¦       ¦   alert_manager.py
¦       ¦   
¦               
+---reports
¦       eval_NVDA_DQN_2025-07-14_14-08-56_20250714_141039_summary.txt
¦       eval_NVDA_DQN_2025-07-14_14-08-56_20250714_141039_trades.csv
¦       
+---risk
¦   ¦   event_bus.py
¦   ¦   event_types.py
¦   ¦   risk_agent_adapter.py
¦   ¦   risk_agent_v2.py
¦   ¦   rules_engine.py
¦   ¦   stress_runner.py
¦   ¦   __init__.py
¦   ¦   
¦   +---audit
¦   ¦   ¦   audit_logger.py
¦   ¦   ¦   
¦   ¦           
¦   +---calculators
¦   ¦   ¦   adv_participation_calculator.py
¦   ¦   ¦   base_calculator.py
¦   ¦   ¦   concentration_calculator.py
¦   ¦   ¦   depth_shock_calculator.py
¦   ¦   ¦   drawdown_calculator.py
¦   ¦   ¦   drawdown_velocity_calculator.py
¦   ¦   ¦   expected_shortfall_calculator.py
¦   ¦   ¦   feed_staleness_calculator.py
¦   ¦   ¦   greeks_calculator.py
¦   ¦   ¦   kyle_lambda_calculator.py
¦   ¦   ¦   latency_drift_calculator.py
¦   ¦   ¦   market_impact_calculator.py
¦   ¦   ¦   stress_test_calculator.py
¦   ¦   ¦   turnover_calculator.py
¦   ¦   ¦   ulcer_index_calculator.py
¦   ¦   ¦   var_calculator.py
¦   ¦   ¦   volatility_calculator.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦   +---config
¦   ¦   ¦   risk_config_hot_reload.py
¦   ¦   ¦   risk_config_manager.py
¦   ¦   ¦   risk_config_validator.py
¦   ¦   ¦   
¦   ¦           
¦   +---enforcement
¦   ¦   ¦   var_stress_enforcer.py
¦   ¦   ¦   
¦   ¦           
¦   +---metrics
¦   ¦   ¦   prometheus_metrics.py
¦   ¦   ¦   
¦   ¦           
¦   +---obs
¦   ¦   ¦   audit_sink.py
¦   ¦   ¦   enhanced_audit_sink.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦   ¦           
¦   +---sensors
¦   ¦   ¦   base_sensor.py
¦   ¦   ¦   counterparty.py
¦   ¦   ¦   failure_mode_detector.py
¦   ¦   ¦   funding_margin.py
¦   ¦   ¦   liquidity_execution.py
¦   ¦   ¦   operational_tech.py
¦   ¦   ¦   path_fragility.py
¦   ¦   ¦   sensor_pipeline.py
¦   ¦   ¦   sensor_registry.py
¦   ¦   ¦   tail_regime.py
¦   ¦   ¦   var_stress_sensors.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦   ¦           
¦           
+---security
¦   ¦   secrets_manager.py
¦   ¦   
¦           
+---shared
¦   ¦   constants.py
¦   ¦   db_locker.py
¦   ¦   db_pool.py
¦   ¦   deprecation.py
¦   ¦   disk_gc_service.py
¦   ¦   dto.py
¦   ¦   duckdb_manager.py
¦   ¦   evaluation_cache.py
¦   ¦   featurestore_monitoring.py
¦   ¦   feature_store.py
¦   ¦   feature_store_optimized.py
¦   ¦   fee_schedule.py
¦   ¦   manifest_schema.py
¦   ¦   market_impact.py
¦   ¦   risk_utils.py
¦   ¦   version.py
¦   ¦   __init__.py
¦   ¦   
¦           
+---storage
¦   ¦   audit_storage_manager.py
¦   ¦   s3_audit_replication.py
¦   ¦   
¦           
+---tools
¦   ¦   GetIBKRData.py
¦   ¦   ibkr_tools.py
¦   ¦   
¦           
+---training
¦   ¦   enhanced_trainer_agent.py
¦   ¦   experiment_registry.py
¦   ¦   hyperparameter_search.py
¦   ¦   reward_audit_integration.py
¦   ¦   reward_pnl_audit.py
¦   ¦   trainer_agent.py
¦   ¦   __init__.py
¦   ¦   
¦   +---callbacks
¦   ¦   ¦   enhanced_risk_callback.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦   ¦           
¦   +---core
¦   ¦   ¦   early_stopping_callback.py
¦   ¦   ¦   env_builder.py
¦   ¦   ¦   hyperparam_search.py
¦   ¦   ¦   policy_export.py
¦   ¦   ¦   risk_callbacks.py
¦   ¦   ¦   trainer_core.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦   ¦           
¦   +---interfaces
¦   ¦   ¦   risk_advisor.py
¦   ¦   ¦   rl_policy.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦   ¦           
¦   +---policies
¦   ¦   ¦   sb3_policy.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦           
+---utils
¦   ¦   db.py
¦   ¦   __init__.py
¦   ¦   
        

- `orchestrator_agent.py`,`execution_loop.py`: Main entry point. Coordinates the RL trading platform:
    - Loads and validates configuration.
    - Initializes and wires all specialized agents.
    - Manages end-to-end pipelines for data, training, evaluation, and live trading.
    - Supports hooks/callbacks for pipeline events.
- `trade_engine.py`: core trading logic & async order placement
- `signal_generator.py`: computes trading signals
- `order_router.py`: Contains order routing and management logic - handles:
    - Order placement and routing
    - Position sizing calculations
    - Order status tracking
    - Broker communication coordination
- `data_loader.py`: fetches historical data from exchanges
- `risk_agent_v2.py`: Enterprise-grade risk management orchestrator.
    -Subscribes to events, runs calculators, evaluates policies, and enforces actions.
- `trainer_core.py` : trainer core module real-time model training—even in live environment; maintain training loop resilience - handles:
    - Core training coordination
    - Model management and lifecycle
    - Training state management
    - Risk advisor integration
- `policy_export.py` : policy export Core Module - TorchScript export must validate metadata & catch serialization errors -  handles:
    - Model bundle saving
    - TorchScript export for production deployment
    - Metadata generation and validation
    - Model versioning and packaging

## Instructions for Gemini
- 🔍 **Review**: focus on async safety, race conditions, retry robustness
- 📖 **Explain**: module responsibilities and logic flows
- 🧪 **Test**: generate pytest suites with mocks, simulate order failures and recovery
- ✨ **Enhance**: propose adaptive volatility filters, async performance optimizations, CLI for trainer
- 🧼 **Lint**: ensure code meets formatting and logging standards

## Tools Preferences
- Use `ReadManyFiles` for bulk scanning
- Use `glob` to locate specific files
- Apply diffs only after user confirmation

## Memory
- “Always prioritize maintainability and clarity”
- “Flag any missing type annotations or docstring”