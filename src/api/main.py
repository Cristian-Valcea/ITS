# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, Any, Optional
import logging
import os
import sys

# Adjust path to import OrchestratorAgent from parent directory (src)
# This is necessary if 'src' is not automatically in PYTHONPATH when running uvicorn from project root.
# Alternatively, run uvicorn with PYTHONPATH="." or install the project as a package.
# current_dir = os.path.dirname(os.path.abspath(__file__)) # src/api
# src_dir = os.path.dirname(current_dir) # src
# project_root = os.path.dirname(src_dir) # rl_trading_platform
# if project_root not in sys.path:
#    sys.path.insert(0, project_root)

# Assuming execution from project root (e.g., uvicorn src.api.main:app)
# or PYTHONPATH is set correctly.
try:
    from src.agents.orchestrator_agent import OrchestratorAgent
    from src.api import config_models, request_models, response_models, services # Placeholder for now
except ModuleNotFoundError:
    # Fallback for simpler execution like `python src/api/main.py` if needed, by adding project root
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from src.agents.orchestrator_agent import OrchestratorAgent
    # The following will be created in subsequent steps, so direct import might fail initially
    # For now, we'll define dummy versions or handle their absence.
    # from src.api import config_models, request_models, response_models, services


# --- Logging Setup ---
# Basic logging, can be expanded using main_config.yaml settings later
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RLTradingPlatform.API")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RL Intraday Trading Platform API",
    description="API to control and interact with the RL trading agent system.",
    version="0.1.0"
)

# --- Global Objects (Simple Approach) ---
# For a more robust setup, especially with async, consider FastAPI's dependency injection for these.
# Config paths relative to project root (assuming API is run from project root or paths are adjusted)
# These paths assume 'config/' is at the same level as 'src/'
# If running `uvicorn src.api.main:app` from project root, `../config` is not correct.
# It should be `config/` directly if project root is the current working directory.
# Let's assume project root is the CWD for uvicorn.
CONFIG_DIR = "config" 
# If this script is in src/api, and config is in root/config, then path needs to be relative to root.
# For Uvicorn running from project root: `config/main_config.yaml`
# For running this file directly from `src/api/`: `../../config/main_config.yaml`

# Determine base path for config files
# This is a common way to make paths robust whether run directly or via uvicorn from root
try:
    # Assumes main.py is in src/api/
    # Adjust if your execution context for uvicorn is different.
    # If uvicorn src.api.main:app is run from project root, os.getcwd() is project root.
    # If python src/api/main.py is run from project root, os.getcwd() is project root.
    # If python main.py is run from src/api/, os.getcwd() is src/api
    
    # Let's assume the script is in src/api and config is in project_root/config
    # This means config is one level up from 'src' directory.
    _project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    DEFAULT_MAIN_CONFIG_PATH = os.path.join(_project_root_path, "config", "main_config.yaml")
    DEFAULT_MODEL_PARAMS_PATH = os.path.join(_project_root_path, "config", "model_params.yaml")
    DEFAULT_RISK_LIMITS_PATH = os.path.join(_project_root_path, "config", "risk_limits.yaml")

except Exception: # Fallback if path logic is tricky during early dev
    logger.warning("Could not determine dynamic config paths, using defaults relative to potential CWD.")
    DEFAULT_MAIN_CONFIG_PATH = "config/main_config.yaml"
    DEFAULT_MODEL_PARAMS_PATH = "config/model_params.yaml"
    DEFAULT_RISK_LIMITS_PATH = "config/risk_limits.yaml"


orchestrator_agent_instance: Optional[OrchestratorAgent] = None
config_service_instance: Optional[Any] = None # Will be services.ConfigService

def get_orchestrator():
    global orchestrator_agent_instance
    if orchestrator_agent_instance is None:
        logger.info(f"Initializing OrchestratorAgent with paths: \nMain: {DEFAULT_MAIN_CONFIG_PATH}\nModel: {DEFAULT_MODEL_PARAMS_PATH}\nRisk: {DEFAULT_RISK_LIMITS_PATH}")
        if not all(os.path.exists(p) for p in [DEFAULT_MAIN_CONFIG_PATH, DEFAULT_MODEL_PARAMS_PATH, DEFAULT_RISK_LIMITS_PATH]):
            logger.error("One or more default config files not found. API cannot fully initialize Orchestrator.")
            # Create dummy files if they don't exist for basic API startup
            for p in [DEFAULT_MAIN_CONFIG_PATH, DEFAULT_MODEL_PARAMS_PATH, DEFAULT_RISK_LIMITS_PATH]:
                if not os.path.exists(p):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, 'w') as f: f.write("{}") # Empty YAML
                    logger.info(f"Created empty dummy config: {p}")
            # Attempt initialization again, it might work with empty configs or use internal defaults
        
        try:
            orchestrator_agent_instance = OrchestratorAgent(
                main_config_path=DEFAULT_MAIN_CONFIG_PATH,
                model_params_path=DEFAULT_MODEL_PARAMS_PATH,
                risk_limits_path=DEFAULT_RISK_LIMITS_PATH
            )
            logger.info("OrchestratorAgent initialized successfully for API.")
        except Exception as e:
            logger.error(f"Failed to initialize OrchestratorAgent for API: {e}", exc_info=True)
            # Not raising HTTPException here as it's at app startup. Endpoints will fail if it's None.
            # Consider a health check endpoint that verifies this.
            raise RuntimeError(f"Failed to initialize OrchestratorAgent: {e}") from e
    return orchestrator_agent_instance

def get_config_service():
    global config_service_instance
    if config_service_instance is None:
        try:
            # This import will be valid after services.py is created.
            from .services import ConfigService 
            config_service_instance = ConfigService(
                main_config_path=DEFAULT_MAIN_CONFIG_PATH,
                model_params_path=DEFAULT_MODEL_PARAMS_PATH,
                risk_limits_path=DEFAULT_RISK_LIMITS_PATH
            )
            logger.info("ConfigService initialized successfully for API.")
        except ImportError:
            logger.warning("ConfigService not yet available (services.py might not be created). Config endpoints will not work.")
            # Define a dummy service if services.py is not yet created
            class DummyConfigService:
                def get_config(self, config_name: str): raise HTTPException(status_code=501, detail="ConfigService not implemented")
                def update_config(self, config_name: str, data: dict): raise HTTPException(status_code=501, detail="ConfigService not implemented")
            config_service_instance = DummyConfigService()
        except Exception as e:
            logger.error(f"Failed to initialize ConfigService for API: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize ConfigService: {e}") from e
    return config_service_instance


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup...")
    # Initialize services on startup to catch errors early
    try:
        get_orchestrator()
        get_config_service()
        logger.info("OrchestratorAgent and ConfigService pre-warmed on startup.")
    except Exception as e:
        logger.critical(f"Critical error during startup initialization: {e}", exc_info=True)
        # Depending on the desired behavior, you might want the app to fail starting
        # or continue with services being None, and endpoints returning errors.

@app.get("/api/v1/status", tags=["General"])
async def get_status():
    """Returns the current status of the API."""
    return {"status": "RL Trading Platform API is running", "timestamp": datetime.now().isoformat()}

# --- Configuration Management Endpoints ---

VALID_CONFIG_NAMES = ["main_config", "model_params", "risk_limits"]

@app.get("/api/v1/config/{config_name}", 
         response_model=response_models.ConfigResponse,
         tags=["Configuration Management"])
async def read_config(config_name: str, service: services.ConfigService = Depends(get_config_service)):
    """
    Retrieve the content of a specified YAML configuration file.
    Valid `config_name` are: "main_config", "model_params", "risk_limits".
    """
    if config_name not in VALID_CONFIG_NAMES:
        raise HTTPException(status_code=404, detail=f"Config '{config_name}' not found. Valid names are: {VALID_CONFIG_NAMES}")
    try:
        config_data = service.get_config(config_name)
        return response_models.ConfigResponse(
            success=True,
            config_name=config_name,
            config_data=config_data,
            message=f"{config_name} fetched successfully."
        )
    except FileNotFoundError:
        logger.error(f"Config file for '{config_name}' not found by service, though _ensure_config_files_exist should prevent this.")
        raise HTTPException(status_code=404, detail=f"Configuration file for '{config_name}' not found on server.")
    except ValueError as e: # Handles parsing errors or other service-level value errors
        logger.error(f"ValueError reading config {config_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing config '{config_name}': {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error fetching config {config_name}: {e}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching config '{config_name}'.")


@app.post("/api/v1/config/main_config", 
          response_model=response_models.ConfigResponse,
          tags=["Configuration Management"])
async def update_main_config(
    new_config: config_models.MainConfig, 
    service: services.ConfigService = Depends(get_config_service)
):
    """Update the main_config.yaml file."""
    try:
        # Pydantic model `new_config` already performed validation on the request body.
        # We need to convert it to a dict for yaml.dump if it's not already.
        # model_dump() is preferred for Pydantic v2, as_dict() for v1 or dict(new_config)
        updated_data = service.update_config("main_config", new_config.model_dump(mode='json'))
        return response_models.ConfigResponse(
            success=True,
            config_name="main_config",
            config_data=updated_data,
            message="main_config updated successfully."
        )
    except ValueError as e:
        logger.error(f"ValueError updating main_config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating main_config: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error updating main_config: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating main_config.")


@app.post("/api/v1/config/model_params", 
          response_model=response_models.ConfigResponse,
          tags=["Configuration Management"])
async def update_model_params_config(
    new_config: config_models.ModelParamsConfig, 
    service: services.ConfigService = Depends(get_config_service)
):
    """Update the model_params.yaml file."""
    try:
        updated_data = service.update_config("model_params", new_config.model_dump(mode='json'))
        return response_models.ConfigResponse(
            success=True,
            config_name="model_params",
            config_data=updated_data,
            message="model_params updated successfully."
        )
    except ValueError as e:
        logger.error(f"ValueError updating model_params: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating model_params: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error updating model_params: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating model_params.")


@app.post("/api/v1/config/risk_limits", 
          response_model=response_models.ConfigResponse,
          tags=["Configuration Management"])
async def update_risk_limits_config(
    new_config: config_models.RiskLimitsConfig, 
    service: services.ConfigService = Depends(get_config_service)
):
    """Update the risk_limits.yaml file."""
    try:
        updated_data = service.update_config("risk_limits", new_config.model_dump(mode='json'))
        return response_models.ConfigResponse(
            success=True,
            config_name="risk_limits",
            config_data=updated_data,
            message="risk_limits updated successfully."
        )
    except ValueError as e:
        logger.error(f"ValueError updating risk_limits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating risk_limits: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error updating risk_limits: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating risk_limits.")


# --- Pipeline Triggering Endpoints ---

@app.post("/api/v1/pipelines/train",
          response_model=response_models.TrainPipelineResponse,
          tags=["Pipelines"])
async def trigger_train_pipeline(
    request: request_models.TrainPipelineRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator)
):
    """
    Triggers the training pipeline.
    This is a synchronous call for now; for long training jobs, consider background tasks.
    """
    logger.info(f"Received request to train pipeline with params: {request.model_dump_json(indent=2)}")
    if orchestrator is None:
        logger.error("OrchestratorAgent not available for training pipeline.")
        raise HTTPException(status_code=503, detail="Orchestrator service is not available. Cannot start training.")

    try:
        # Note: OrchestratorAgent uses the config files as they are on disk when its methods are called.
        # If configs were updated via API, the orchestrator instance (if global) might need re-initialization
        # or a reload_config method to see changes, unless it reads them fresh each time.
        # The current OrchestratorAgent re-initializes its sub-agents (which load configs) each time
        # a pipeline method is called, if those sub-agents are not persistent members.
        # Let's assume OrchestratorAgent re-reads configs or re-initializes agents as needed.
        
        # The OrchestratorAgent.run_training_pipeline expects individual args, not the Pydantic model directly.
        trained_model_path = orchestrator.run_training_pipeline(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            use_cached_data=request.use_cached_data,
            continue_from_model=request.continue_from_model,
            run_evaluation_after_train=request.run_evaluation_after_train,
            eval_start_date=request.eval_start_date,
            eval_end_date=request.eval_end_date,
            eval_interval=request.eval_interval
        )

        if trained_model_path:
            # If evaluation ran, OrchestratorAgent.run_evaluation_pipeline would log results.
            # For this response, we only confirm training model path.
            # A more advanced system might return a task ID for async job and a way to fetch eval results.
            return response_models.TrainPipelineResponse(
                success=True,
                message="Training pipeline completed successfully.",
                model_path=trained_model_path
                # evaluation_results_after_train can be populated if Orchestrator returns it
            )
        else:
            logger.error("Training pipeline did not return a model path.")
            raise HTTPException(status_code=500, detail="Training pipeline completed but no model path was returned.")

    except FileNotFoundError as e: # e.g. if a critical config file was deleted after orchestrator init
        logger.error(f"FileNotFoundError during training pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server configuration error: {str(e)}")
    except ValueError as e: # From Orchestrator or sub-agents for bad params/data issues
        logger.error(f"ValueError during training pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Training pipeline error: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error during training pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during training: {str(e)}")


@app.post("/api/v1/pipelines/evaluate",
          response_model=response_models.EvaluatePipelineResponse,
          tags=["Pipelines"])
async def trigger_evaluate_pipeline(
    request: request_models.EvaluatePipelineRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator)
):
    """
    Triggers the evaluation pipeline for a specified model.
    """
    logger.info(f"Received request to evaluate pipeline with params: {request.model_dump_json(indent=2)}")
    if orchestrator is None:
        logger.error("OrchestratorAgent not available for evaluation pipeline.")
        raise HTTPException(status_code=503, detail="Orchestrator service is not available. Cannot start evaluation.")

    if not os.path.exists(request.model_path) and not os.path.exists(request.model_path + ".dummy"):
        logger.error(f"Model path for evaluation does not exist: {request.model_path}")
        raise HTTPException(status_code=404, detail=f"Model file not found at: {request.model_path}")

    try:
        # OrchestratorAgent.run_evaluation_pipeline needs algorithm_name.
        # This should ideally come from model metadata or be part of EvaluatePipelineRequest.
        # For now, we'll fetch it from model_params.yaml via the orchestrator's loaded config.
        algo_name = orchestrator.model_params_config.get('algorithm_name', 'DQN')
        
        # model_name_tag for report naming
        model_name_tag = os.path.basename(request.model_path).replace(".zip", "").replace(".dummy", "")


        eval_metrics = orchestrator.run_evaluation_pipeline(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            model_path=request.model_path,
            use_cached_data=request.use_cached_data
            # Note: EvaluatorAgent's run method takes algorithm_name and model_name_tag
            # OrchestratorAgent's run_evaluation_pipeline should handle passing these.
            # We need to ensure OrchestratorAgent is updated to pass algo_name & model_name_tag to EvaluatorAgent.
            # For now, let's assume Orchestrator is doing that correctly.
        )

        if eval_metrics is not None:
            # EvaluatorAgent saves reports, paths might not be directly returned by Orchestrator's current eval pipeline.
            # For now, we return metrics. Report paths could be constructed based on conventions if needed.
            return response_models.EvaluatePipelineResponse(
                success=True,
                message="Evaluation pipeline completed successfully.",
                metrics=eval_metrics
                # report_path and trade_log_path would require Orchestrator/Evaluator to return them.
            )
        else:
            logger.error("Evaluation pipeline did not return metrics.")
            raise HTTPException(status_code=500, detail="Evaluation pipeline completed but no metrics were returned.")

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError during evaluation pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server configuration or data file error: {str(e)}")
    except ValueError as e:
        logger.error(f"ValueError during evaluation pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Evaluation pipeline error: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error during evaluation pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during evaluation: {str(e)}")



# To run this API locally (from the project root directory `rl_trading_platform/`):
# Ensure FastAPI and Uvicorn are installed: pip install fastapi "uvicorn[standard]"
# Command: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
#
# Then access in browser:
# API status: http://localhost:8000/api/v1/status
# Interactive API docs (Swagger): http://localhost:8000/docs
# Alternative API docs (ReDoc): http://localhost:8000/redoc

if __name__ == "__main__":
    # This block allows running the API directly with `python src/api/main.py`
    # However, `uvicorn` command is preferred for development and production.
    logger.info("Attempting to run API with Uvicorn directly (for development/testing only).")
    try:
        import uvicorn
        # Make sure config files exist for Orchestrator init when running this way
        # For simplicity, we assume OrchestratorAgent's __main__ or another process
        # has created dummy configs if needed.
        # The get_orchestrator() function called during startup will handle this.
        
        # Check if config files exist, create minimal dummies if not, for direct run
        config_files_to_check = [DEFAULT_MAIN_CONFIG_PATH, DEFAULT_MODEL_PARAMS_PATH, DEFAULT_RISK_LIMITS_PATH]
        for p in config_files_to_check:
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, 'w') as f:
                    yaml_content = {}
                    if "main_config" in p:
                        yaml_content = {"paths": {"log_dir": "../../logs"}} # Minimal for logging
                    elif "model_params" in p:
                        yaml_content = {"algorithm_name": "DQN"}
                    elif "risk_limits" in p:
                        yaml_content = {"max_daily_drawdown_pct": 0.02}
                    import yaml # Local import for this block
                    yaml.dump(yaml_content, f)
                logger.info(f"Created minimal dummy config for direct run: {p}")

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except ImportError:
        logger.error("Uvicorn not found. Please run with `uvicorn src.api.main:app --reload`")
    except Exception as e:
        logger.error(f"Failed to run Uvicorn directly: {e}", exc_info=True)

# Placeholder for Pydantic models and services until they are created
# This is to avoid import errors if this file is checked before others are created
if 'config_models' not in sys.modules:
    class DummyPydanticModel: pass
    config_models = DummyPydanticModel() # type: ignore
    request_models = DummyPydanticModel() # type: ignore
    response_models = DummyPydanticModel() # type: ignore

if 'services' not in sys.modules:
    class DummyServicesModule: pass
    services = DummyServicesModule() # type: ignore
