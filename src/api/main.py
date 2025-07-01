# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, Optional
import logging
import os
import sys
import uuid # For generating task IDs
from datetime import datetime


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
    from src.api import config_models, request_models, response_models, services 
except ModuleNotFoundError:
    # Fallback for simpler execution like `python src/api/main.py` if needed, by adding project root
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from src.agents.orchestrator_agent import OrchestratorAgent
    from src.api import config_models, request_models, response_models, services


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RLTradingPlatform.API")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RL Intraday Trading Platform API",
    description="API to control and interact with the RL trading agent system.",
    version="0.1.0"
)

# --- Global Objects & Task Management (Simple In-Memory) ---
orchestrator_agent_instance: Optional[OrchestratorAgent] = None
config_service_instance: Optional[services.ConfigService] = None

# For managing background task statuses (simple in-memory store)
tasks_status: Dict[str, Dict[str, Any]] = {}

# Determine base path for config files
_project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MAIN_CONFIG_PATH = os.path.join(_project_root_path, "config", "main_config.yaml")
DEFAULT_MODEL_PARAMS_PATH = os.path.join(_project_root_path, "config", "model_params.yaml")
DEFAULT_RISK_LIMITS_PATH = os.path.join(_project_root_path, "config", "risk_limits.yaml")


def get_orchestrator():
    global orchestrator_agent_instance
    if orchestrator_agent_instance is None:
        logger.info(f"Initializing OrchestratorAgent with paths: \nMain: {DEFAULT_MAIN_CONFIG_PATH}\nModel: {DEFAULT_MODEL_PARAMS_PATH}\nRisk: {DEFAULT_RISK_LIMITS_PATH}")
        if not all(os.path.exists(p) for p in [DEFAULT_MAIN_CONFIG_PATH, DEFAULT_MODEL_PARAMS_PATH, DEFAULT_RISK_LIMITS_PATH]):
            logger.error("One or more default config files not found. API cannot fully initialize Orchestrator.")
            for p in [DEFAULT_MAIN_CONFIG_PATH, DEFAULT_MODEL_PARAMS_PATH, DEFAULT_RISK_LIMITS_PATH]:
                if not os.path.exists(p):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, 'w') as f: f.write("{}") 
                    logger.info(f"Created empty dummy config: {p}")
        try:
            orchestrator_agent_instance = OrchestratorAgent(
                main_config_path=DEFAULT_MAIN_CONFIG_PATH,
                model_params_path=DEFAULT_MODEL_PARAMS_PATH,
                risk_limits_path=DEFAULT_RISK_LIMITS_PATH
            )
            logger.info("OrchestratorAgent initialized successfully for API.")
        except Exception as e:
            logger.error(f"Failed to initialize OrchestratorAgent for API: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize OrchestratorAgent: {e}") from e
    return orchestrator_agent_instance

def get_config_service():
    global config_service_instance
    if config_service_instance is None:
        try:
            config_service_instance = services.ConfigService(
                main_config_path=DEFAULT_MAIN_CONFIG_PATH,
                model_params_path=DEFAULT_MODEL_PARAMS_PATH,
                risk_limits_path=DEFAULT_RISK_LIMITS_PATH
            )
            logger.info("ConfigService initialized successfully for API.")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigService for API: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize ConfigService: {e}") from e
    return config_service_instance


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup...")
    try:
        get_orchestrator()
        get_config_service()
        logger.info("OrchestratorAgent and ConfigService pre-warmed on startup.")
    except Exception as e:
        logger.critical(f"Critical error during startup initialization: {e}", exc_info=True)

@app.get("/api/v1/status", tags=["General"], response_model=response_models.StandardResponse)
async def get_status():
    """Returns the current status of the API."""
    return response_models.StandardResponse(
        success=True,
        message="RL Trading Platform API is running",
        data={"timestamp": datetime.now().isoformat()}
    )

# --- Configuration Management Endpoints ---
VALID_CONFIG_NAMES = ["main_config", "model_params", "risk_limits"]

@app.get("/api/v1/config/{config_name}", 
         response_model=response_models.ConfigResponse,
         tags=["Configuration Management"])
async def read_config(config_name: str, service: services.ConfigService = Depends(get_config_service)):
    if config_name not in VALID_CONFIG_NAMES:
        raise HTTPException(status_code=404, detail=f"Config '{config_name}' not found. Valid names are: {VALID_CONFIG_NAMES}")
    try:
        config_data = service.get_config(config_name)
        return response_models.ConfigResponse(
            success=True, config_name=config_name, config_data=config_data,
            message=f"{config_name} fetched successfully."
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Configuration file for '{config_name}' not found on server.")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error processing config '{config_name}': {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error fetching config {config_name}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching config '{config_name}'.")

@app.post("/api/v1/config/main_config", response_model=response_models.ConfigResponse, tags=["Configuration Management"])
async def update_main_config(new_config: config_models.MainConfig, service: services.ConfigService = Depends(get_config_service)):
    try:
        updated_data = service.update_config("main_config", new_config.model_dump(mode='json'))
        # Consider if OrchestratorAgent needs to be reloaded/notified
        # For now, a restart of API would be needed for Orchestrator to pick up changes.
        return response_models.ConfigResponse(success=True, config_name="main_config", config_data=updated_data, message="main_config updated successfully.")
    except Exception as e:
        logger.exception(f"Error updating main_config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/config/model_params", response_model=response_models.ConfigResponse, tags=["Configuration Management"])
async def update_model_params_config(new_config: config_models.ModelParamsConfig, service: services.ConfigService = Depends(get_config_service)):
    try:
        updated_data = service.update_config("model_params", new_config.model_dump(mode='json'))
        return response_models.ConfigResponse(success=True, config_name="model_params", config_data=updated_data, message="model_params updated successfully.")
    except Exception as e:
        logger.exception(f"Error updating model_params_config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/config/risk_limits", response_model=response_models.ConfigResponse, tags=["Configuration Management"])
async def update_risk_limits_config(new_config: config_models.RiskLimitsConfig, service: services.ConfigService = Depends(get_config_service)):
    try:
        updated_data = service.update_config("risk_limits", new_config.model_dump(mode='json'))
        return response_models.ConfigResponse(success=True, config_name="risk_limits", config_data=updated_data, message="risk_limits updated successfully.")
    except Exception as e:
        logger.exception(f"Error updating risk_limits_config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Pipeline Triggering Endpoints ---

@app.post("/api/v1/pipelines/train",
          response_model=response_models.TrainPipelineResponse, # Will be updated for async
          tags=["Pipelines"])
async def trigger_train_pipeline(
    request: request_models.TrainPipelineRequest,
    background_tasks: BackgroundTasks, # Inject BackgroundTasks
    orchestrator: OrchestratorAgent = Depends(get_orchestrator)
):
    """
    Triggers the training pipeline asynchronously.
    """
    logger.info(f"Received request to train pipeline with params: {request.model_dump_json(indent=2)}")
    if orchestrator is None:
        logger.error("OrchestratorAgent not available for training pipeline.")
        raise HTTPException(status_code=503, detail="Orchestrator service is not available. Cannot start training.")

    task_id = str(uuid.uuid4())
    tasks_status[task_id] = {
        "status": "PENDING", 
        "start_time": datetime.now().isoformat(), 
        "result": None, 
        "error": None,
        "description": f"Training for symbol {request.symbol}"
    }

    # Define the actual work to be done in the background
    def training_task_wrapper(task_id: str, req: request_models.TrainPipelineRequest):
        tasks_status[task_id]["status"] = "RUNNING"
        tasks_status[task_id]["start_time"] = datetime.now().isoformat() # More accurate start time
        logger.info(f"Background training task {task_id} started for symbol {req.symbol}.")
        try:
            trained_model_path = orchestrator.run_training_pipeline(
                symbol=req.symbol,
                start_date=req.start_date,
                end_date=req.end_date,
                interval=req.interval,
                use_cached_data=req.use_cached_data,
                continue_from_model=req.continue_from_model,
                run_evaluation_after_train=req.run_evaluation_after_train,
                eval_start_date=req.eval_start_date,
                eval_end_date=req.eval_end_date,
                eval_interval=req.eval_interval
            )
            if trained_model_path:
                tasks_status[task_id]["status"] = "COMPLETED"
                tasks_status[task_id]["result"] = {"model_path": trained_model_path}
                logger.info(f"Background training task {task_id} completed. Model: {trained_model_path}")
            else:
                tasks_status[task_id]["status"] = "FAILED"
                tasks_status[task_id]["error"] = "Training pipeline completed but no model path was returned."
                logger.error(f"Background training task {task_id} failed: No model path returned.")
        except Exception as e:
            logger.exception(f"Exception in background training task {task_id}: {e}")
            tasks_status[task_id]["status"] = "FAILED"
            tasks_status[task_id]["error"] = str(e)
        tasks_status[task_id]["end_time"] = datetime.now().isoformat()

    background_tasks.add_task(training_task_wrapper, task_id, request)
    
    return response_models.TrainPipelineResponse(
        success=True,
        message=f"Training pipeline initiated in background. Task ID: {task_id}",
        task_id=task_id
    )

@app.get("/api/v1/pipelines/train/status/{task_id}", 
         response_model=response_models.TaskStatusResponse, # To be created in response_models.py
         tags=["Pipelines"])
async def get_training_task_status(task_id: str):
    """
    Retrieves the status of a background training task.
    """
    task_info = tasks_status.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")
    
    return response_models.TaskStatusResponse(
        task_id=task_id,
        status=task_info.get("status", "UNKNOWN"),
        start_time=task_info.get("start_time"),
        end_time=task_info.get("end_time"),
        description=task_info.get("description"),
        result=task_info.get("result"),
        error=task_info.get("error")
    )

@app.post("/api/v1/pipelines/evaluate",
          response_model=response_models.EvaluatePipelineResponse,
          tags=["Pipelines"])
async def trigger_evaluate_pipeline(
    request: request_models.EvaluatePipelineRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator)
):
    """
    Triggers the evaluation pipeline for a specified model.
    (This remains synchronous for now, as evaluation is typically faster)
    """
    logger.info(f"Received request to evaluate pipeline with params: {request.model_dump_json(indent=2)}")
    if orchestrator is None:
        logger.error("OrchestratorAgent not available for evaluation pipeline.")
        raise HTTPException(status_code=503, detail="Orchestrator service is not available. Cannot start evaluation.")

    # Check if model_path + .dummy exists if model_path itself doesn't
    actual_model_path_to_check = request.model_path
    if not os.path.exists(request.model_path) and os.path.exists(request.model_path + ".dummy"):
        logger.info(f"Actual model file {request.model_path} not found, but dummy file {request.model_path + '.dummy'} exists. Proceeding with dummy model path.")
        # No change needed for request.model_path if EvaluatorAgent handles the .dummy extension
    elif not os.path.exists(request.model_path):
         logger.error(f"Model path for evaluation does not exist: {request.model_path} (and no .dummy file)")
         raise HTTPException(status_code=404, detail=f"Model file not found at: {request.model_path}")


    try:
        algo_name = orchestrator.model_params_config.get('algorithm_name', 'DQN')
        model_name_tag = os.path.basename(request.model_path).replace(".zip", "").replace(".dummy", "")

        eval_metrics = orchestrator.run_evaluation_pipeline(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            model_path=request.model_path,
            use_cached_data=request.use_cached_data
        )

        if eval_metrics is not None:
            # Assuming EvaluatorAgent's run method returns a dict that includes paths to reports
            # This might need adjustment in EvaluatorAgent or OrchestratorAgent
            report_path = eval_metrics.pop("report_txt_path", None) # Example key
            trade_log_path = eval_metrics.pop("report_trades_csv_path", None) # Example key

            return response_models.EvaluatePipelineResponse(
                success=True,
                message="Evaluation pipeline completed successfully.",
                metrics=eval_metrics,
                report_path=report_path,
                trade_log_path=trade_log_path
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

# All modules should be properly imported at the top of the file
