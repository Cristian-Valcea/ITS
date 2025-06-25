# src/main.py
import argparse
import logging
import os
import sys

# Ensure the 'src' directory is in the Python path
# This allows for absolute imports like `from agents.orchestrator_agent import OrchestratorAgent`
# when running main.py from the root of the project.
# If main.py is in src/, and you run `python src/main.py`, then `from agents...` works.
# If you run `python main.py` from root, and main.py is in root, then `from src.agents...` works.
# Assuming main.py is in src/, and we run from project root (`python -m src.main`) or `python src/main.py`.
try:
    from agents.orchestrator_agent import OrchestratorAgent
    # If running from project root as `python src/main.py`, this might fail.
    # `PYTHONPATH=. python src/main.py` or `python -m src.main` is better.
    # Or, adjust sys.path:
except ImportError:
    # This adjustment is often needed if you run `python src/main.py` from the project root directory.
    # It adds the project root to sys.path, so `from src.agents...` would work.
    # However, the current agent skeletons use `from .data_agent import DataAgent` (relative imports)
    # within the agents package, and `from src.gym_env...` from agents.
    # The structure `rl_trading_platform/src/main.py` and `rl_trading_platform/src/agents/`
    # means that if main.py is the entry point, Python needs to know about `rl_trading_platform` or `src`.

    # Let's assume we are running this file from the `rl_trading_platform` directory.
    # e.g. `python src/main.py train --symbol AAPL`
    # In this case, Python automatically adds `rl_trading_platform/src` to the path for `main.py`,
    # so `from agents.orchestrator_agent import OrchestratorAgent` should work assuming `agents` is a subdir of `src`.
    
    # If the script is in `rl_trading_platform/src/main.py` and run from `rl_trading_platform/`:
    # `python src/main.py ...`
    # Then `src` is the directory of `main.py`.
    # `from agents.orchestrator_agent import OrchestratorAgent` is correct if `agents` is `src/agents`.
    # This seems to be the standard setup.

    # If there's still an issue, it might be because of how the package 'src' is seen.
    # For a quick fix if running `python src/main.py` from project root:
    # current_dir = os.path.dirname(os.path.abspath(__file__)) # Should be .../rl_trading_platform/src
    # project_root = os.path.dirname(current_dir) # Should be .../rl_trading_platform
    # if project_root not in sys.path:
    #     sys.path.insert(0, project_root)
    # Now `from src.agents.orchestrator_agent import OrchestratorAgent` should work.
    # However, the skeletons use `from .base_agent...` within agents, which implies `agents` is a package.
    # The current imports in agent files like `from .base_agent import BaseAgent` are relative.
    # The import `from src.gym_env.intraday_trading_env import IntradayTradingEnv` in `env_agent.py`
    # implies that `src` must be a top-level package visible in PYTHONPATH.
    
    # To make `from src.gym_env...` work from `src/agents/env_agent.py`,
    # the `rl_trading_platform` directory (parent of `src`) must be in `sys.path`.
    # This is typical if you install your project as a package or run with `python -m src.main`.

    # For simplicity of this skeleton, let's assume the PYTHONPATH is set up correctly
    # (e.g. by running `export PYTHONPATH=.` from the `rl_trading_platform` root before execution,
    # or by an IDE that does this).

    # Let's try a common structure for running scripts within a package:
    if __package__ is None or __package__ == '':
        # If main.py is run as a script, __package__ is None or ''.
        # We need to adjust sys.path to make 'src' a discoverable package root.
        module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if module_path not in sys.path:
            sys.path.append(module_path)
        from src.agents.orchestrator_agent import OrchestratorAgent # Now this should work
    else: # If run as a module (e.g. python -m src.main), imports should be fine.
        from agents.orchestrator_agent import OrchestratorAgent


# Default config file paths (relative to project root, assuming main.py is in src/)
# If main.py is in src, then "../config/" would point to "rl_trading_platform/config/"
DEFAULT_MAIN_CONFIG_PATH = "../config/main_config.yaml"
DEFAULT_MODEL_PARAMS_PATH = "../config/model_params.yaml"
DEFAULT_RISK_LIMITS_PATH = "../config/risk_limits.yaml"

# --- Logging Setup ---
# Basic logging can be configured here. Orchestrator and other agents will use this.
# More advanced logging (e.g., file rotation, specific formatting) can be in a util.
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)] # Log to console
    )
    # TODO: Add FileHandler to log to app.log as per directory structure
    # log_dir = main_config.get('paths', {}).get('log_dir', 'logs') # Need config first for this
    # app_log_file = os.path.join(log_dir, "app.log")
    # file_handler = logging.FileHandler(app_log_file)
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # logging.getLogger().addHandler(file_handler) # Add to root logger


def main():
    setup_logging(logging.INFO) # Default log level
    logger = logging.getLogger("RLTradingPlatform.Main")

    parser = argparse.ArgumentParser(description="RL Intraday Trading Platform Orchestrator")
    parser.add_argument(
        'mode', 
        choices=['train', 'evaluate', 'live', 'schedule_retrain', 'walk_forward'], # Add more modes as needed
        help="Operation mode: 'train' a new model, 'evaluate' an existing model, "
             "'live' trading (conceptual), 'schedule_retrain' (conceptual), "
             "'walk_forward' validation (conceptual)."
    )
    # Config file arguments
    parser.add_argument('--main_config', type=str, default=DEFAULT_MAIN_CONFIG_PATH, help="Path to main configuration YAML file.")
    parser.add_argument('--model_params', type=str, default=DEFAULT_MODEL_PARAMS_PATH, help="Path to model parameters YAML file.")
    parser.add_argument('--risk_limits', type=str, default=DEFAULT_RISK_LIMITS_PATH, help="Path to risk limits YAML file.")

    # Common arguments for train/evaluate
    parser.add_argument('--symbol', type=str, help="Stock symbol (e.g., AAPL, MSFT). Required for train/evaluate.")
    parser.add_argument('--start_date', type=str, help="Start date for data (YYYY-MM-DD).")
    parser.add_argument('--end_date', type=str, help="End date for data (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS for IBKR).")
    parser.add_argument('--interval', type=str, default="1min", help="Data interval (e.g., '1min', '5mins', '1day').")
    
    # Training specific arguments
    parser.add_argument('--use_cached_data', action='store_true', help="Use cached raw data if available (for training/evaluation).")
    parser.add_argument('--continue_from_model', type=str, help="Path to a pre-trained model to continue training from.")

    # Evaluation specific arguments
    parser.add_argument('--model_path', type=str, help="Path to the trained model to evaluate.")

    # Walk-forward specific arguments (conceptual)
    # parser.add_argument('--wf_config_path', type=str, help="Path to walk-forward configuration YAML.")

    args = parser.parse_args()

    logger.info(f"Starting application in '{args.mode}' mode.")
    logger.info(f"Main config: {args.main_config}")
    logger.info(f"Model params config: {args.model_params}")
    logger.info(f"Risk limits config: {args.risk_limits}")

    # --- Initialize Orchestrator ---
    # Ensure config paths are correct relative to the project root if main.py is in src/
    # If main.py is in src/, paths like "../config/" are correct.
    try:
        orchestrator = OrchestratorAgent(
            main_config_path=args.main_config,
            model_params_path=args.model_params,
            risk_limits_path=args.risk_limits
        )
    except ValueError as e:
        logger.error(f"Failed to initialize OrchestratorAgent: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during Orchestrator initialization: {e}", exc_info=True)
        sys.exit(1)


    # --- Execute selected mode ---
    if args.mode == 'train':
        if not all([args.symbol, args.start_date, args.end_date]):
            logger.error("For 'train' mode, --symbol, --start_date, and --end_date are required.")
            sys.exit(1)
        logger.info(f"Initiating training for {args.symbol} from {args.start_date} to {args.end_date} ({args.interval}).")
        trained_model_file = orchestrator.run_training_pipeline(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            use_cached_data=args.use_cached_data,
            continue_from_model=args.continue_from_model
        )
        if trained_model_file:
            logger.info(f"Training pipeline completed. Model saved to: {trained_model_file}")
        else:
            logger.error("Training pipeline failed.")
            sys.exit(1)

    elif args.mode == 'evaluate':
        if not all([args.symbol, args.start_date, args.end_date, args.model_path]):
            logger.error("For 'evaluate' mode, --symbol, --start_date, --end_date, and --model_path are required.")
            sys.exit(1)
        logger.info(f"Initiating evaluation for {args.symbol} from {args.start_date} to {args.end_date} ({args.interval}) on model: {args.model_path}.")
        eval_results = orchestrator.run_evaluation_pipeline(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            model_path=args.model_path,
            use_cached_data=args.use_cached_data
        )
        if eval_results:
            logger.info(f"Evaluation pipeline completed. Results: {eval_results}")
        else:
            logger.error("Evaluation pipeline failed or returned no results.")
            sys.exit(1)

    elif args.mode == 'live':
        if not args.symbol:
            logger.error("For 'live' mode, --symbol is required.")
            sys.exit(1)
        logger.info(f"Initiating live trading for {args.symbol} (Conceptual - Not Implemented).")
        orchestrator.run_live_trading(symbol=args.symbol)
        # This will likely involve a loop and interaction with RiskAgent, IBKR, etc.

    elif args.mode == 'schedule_retrain':
        logger.info("Initiating scheduled retraining (Conceptual - Not Implemented).")
        orchestrator.schedule_weekly_retrain()
        # This would typically set up a job or be run by an external scheduler like cron.

    elif args.mode == 'walk_forward':
        # if not all([args.symbol, args.wf_config_path]): # Assuming wf_config_path holds walk-forward params
        #     logger.error("For 'walk_forward' mode, --symbol and --wf_config_path are required.")
        #     sys.exit(1)
        # wf_config = orchestrator._load_yaml_config(args.wf_config_path, "Walk-Forward Config") # Orchestrator might load this
        # if not wf_config:
        #     sys.exit(1)
        # logger.info(f"Initiating walk-forward validation for {args.symbol} using config: {args.wf_config_path} (Conceptual - Not Implemented).")
        # orchestrator.run_walk_forward_evaluation(symbol=args.symbol, walk_forward_config=wf_config)
        logger.warning("Walk-forward mode is conceptual and not fully implemented in this skeleton.")
        logger.info("To implement, define walk-forward periods in a config and have Orchestrator loop through train/eval cycles.")


    else:
        logger.error(f"Unknown mode: {args.mode}")
        parser.print_help()
        sys.exit(1)

    logger.info(f"Application finished mode '{args.mode}'.")


if __name__ == "__main__":
    # --- Example of how to run from command line ---
    # Assume project root is `rl_trading_platform/` and this file is `rl_trading_platform/src/main.py`
    # Config files are expected in `rl_trading_platform/config/`
    
    # To run training:
    # (Ensure dummy config files from orchestrator_agent.py's __main__ are in ../config/ relative to this script)
    # `python src/main.py train --symbol DUMMYMAIN --start_date 2023-01-01 --end_date 2023-01-10 --interval 1min`
    # `python src/main.py train --symbol DUMMYMAIN --start_date 2023-01-01 --end_date "2023-01-10 23:59:59" --interval 1min --main_config ../config/main_config_orchestrator_test.yaml --model_params ../config/model_params_orchestrator_test.yaml --risk_limits ../config/risk_limits_orchestrator_test.yaml`

    # To run evaluation (after a dummy model is created by train mode):
    # `python src/main.py evaluate --symbol DUMMYMAIN --start_date 2023-02-01 --end_date 2023-02-05 --interval 1min --model_path models/orch_test/DUMMYTRAIN_DQN_final_YYYYMMDD_HHMMSS.zip.dummy`
    # (Replace YYYYMMDD_HHMMSS with actual timestamp from training output, and add .dummy if using dummy trainer)
    # `python src/main.py evaluate --symbol DUMMYMAIN --start_date 2023-02-01 --end_date "2023-02-05 23:59:59" --interval 1min --model_path models/orch_test/DUMMYTRAIN_DQN_final_....zip.dummy --main_config ../config/main_config_orchestrator_test.yaml --model_params ../config/model_params_orchestrator_test.yaml --risk_limits ../config/risk_limits_orchestrator_test.yaml`
    
    # For the `if __name__ == "__main__":` block to run these examples directly using `python src/main.py` without args,
    # you would need to set `sys.argv` manually here, or call `main()` with default test arguments.
    # For this skeleton, the user is expected to run with CLI arguments.

    # Create dummy config files if they don't exist, for easier first-time testing of main.py
    # This duplicates some logic from orchestrator_agent.py's __main__ for convenience if running main.py directly.
    def _ensure_dummy_configs_exist():
        config_dir = os.path.join(os.path.dirname(__file__), "..", "config") # ../config from src/main.py
        os.makedirs(config_dir, exist_ok=True)
        
        paths_to_check = {
            "main": os.path.join(config_dir, "main_config.yaml"),
            "model": os.path.join(config_dir, "model_params.yaml"),
            "risk": os.path.join(config_dir, "risk_limits.yaml")
        }

        if not os.path.exists(paths_to_check["main"]):
            dummy_main_cfg_content = {
                'paths': {'data_dir_raw': '../data/raw_main/', 'model_save_dir': '../models/main/', 'reports_dir': '../reports/main/', 'tensorboard_log_dir': '../logs/tensorboard_main/'},
                'feature_engineering': {'lookback_window': 1, 'features': ['RSI'], 'rsi': {'window':14}},
                'environment': {'initial_capital': 10000}, 'training': {'total_timesteps': 100}, 'evaluation': {}
            } # Simplified
            with open(paths_to_check["main"], 'w') as f: yaml.dump(dummy_main_cfg_content, f)
            print(f"Created dummy main config at {paths_to_check['main']}")

        if not os.path.exists(paths_to_check["model"]):
            dummy_model_cfg_content = {'algorithm_name': 'DQN', 'algorithm_params': {'verbose':0}}
            with open(paths_to_check["model"], 'w') as f: yaml.dump(dummy_model_cfg_content, f)
            print(f"Created dummy model params config at {paths_to_check['model']}")

        if not os.path.exists(paths_to_check["risk"]):
            dummy_risk_cfg_content = {'max_daily_drawdown_pct': 0.05}
            with open(paths_to_check["risk"], 'w') as f: yaml.dump(dummy_risk_cfg_content, f)
            print(f"Created dummy risk limits config at {paths_to_check['risk']}")
            
    _ensure_dummy_configs_exist() # Call this to create defaults if missing.
                                  # Note: These are very minimal and might not match Orchestrator's expectations fully.
                                  # It's better to use the configs generated by Orchestrator's `__main__` block.
                                  # The paths used by `main()` are `../config/main_config.yaml` etc.
                                  # The paths in `_ensure_dummy_configs_exist` also point to `../config/`

    main()
