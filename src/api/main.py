# src/api/main.py
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

# --- Constants & Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DEFAULT_MAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, "main_config.yaml")
DEFAULT_MODEL_PARAMS_PATH = os.path.join(CONFIG_DIR, "model_params.yaml")
DEFAULT_RISK_LIMITS_PATH = os.path.join(CONFIG_DIR, "risk_limits.yaml")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RLTradingPlatform.API")

# --- Dynamic Imports (for local dev and prod) ---
try:
    from src.agents.orchestrator_agent import OrchestratorAgent
    from src.api import config_models, request_models, response_models, services
except ModuleNotFoundError:
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from src.agents.orchestrator_agent import OrchestratorAgent
    from src.api import config_models, request_models, response_models, services

# --- Template Setup ---
if not os.path.isdir(TEMPLATES_DIR):
    logger.warning(f"Templates directory {TEMPLATES_DIR} not found. Creating it now.")
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)
logger.info(f"Jinja2Templates configured with directory: {TEMPLATES_DIR}")

# --- Global State ---
orchestrator_agent_instance: Optional[OrchestratorAgent] = None
config_service_instance: Optional[services.ConfigService] = None
tasks_status: Dict[str, Dict[str, Any]] = {}

PYDANTIC_MODEL_MAP = {
    "main_config": config_models.MainConfig,
    "model_params": config_models.ModelParamsConfig,
    "risk_limits": config_models.RiskLimitsConfig,
}
VALID_CONFIG_NAMES = list(PYDANTIC_MODEL_MAP.keys())

# --- Utility Functions ---
def ensure_config_file(path: str) -> None:
    """Ensure a config file exists at the given path."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump({}, f)
        logger.info(f"Created empty dummy config: {path}")

def get_orchestrator() -> OrchestratorAgent:
    global orchestrator_agent_instance
    if orchestrator_agent_instance is None:
        logger.info(
            f"Initializing OrchestratorAgent with paths:\n"
            f"Main: {DEFAULT_MAIN_CONFIG_PATH}\n"
            f"Model: {DEFAULT_MODEL_PARAMS_PATH}\n"
            f"Risk: {DEFAULT_RISK_LIMITS_PATH}"
        )
        for path in [DEFAULT_MAIN_CONFIG_PATH, DEFAULT_MODEL_PARAMS_PATH, DEFAULT_RISK_LIMITS_PATH]:
            ensure_config_file(path)
        try:
            orchestrator_agent_instance = OrchestratorAgent(
                main_config_path=DEFAULT_MAIN_CONFIG_PATH,
                model_params_path=DEFAULT_MODEL_PARAMS_PATH,
                risk_limits_path=DEFAULT_RISK_LIMITS_PATH
            )
            logger.info("OrchestratorAgent initialized successfully for API.")
        except Exception as e:
            logger.error("Failed to initialize OrchestratorAgent for API", exc_info=True)
            raise RuntimeError(f"Failed to initialize OrchestratorAgent: {e}") from e
    return orchestrator_agent_instance

def get_config_service() -> services.ConfigService:
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
            logger.error("Failed to initialize ConfigService for API", exc_info=True)
            raise RuntimeError(f"Failed to initialize ConfigService: {e}") from e
    return config_service_instance

# --- FastAPI App Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup...")
    try:
        get_config_service()
        get_orchestrator()
        logger.info("OrchestratorAgent and ConfigService pre-warmed on startup.")
    except Exception as e:
        logger.critical("Critical error during startup initialization", exc_info=True)
    yield
    logger.info("Shutting down...")
    # Place resource cleanup here if needed

app = FastAPI(
    title="RL Intraday Trading Platform API",
    description="API to control and interact with the RL trading agent system.",
    version="0.1.0",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.get("/", response_class=RedirectResponse, tags=["Root"])
async def root():
    """Redirect root URL to dashboard"""
    return RedirectResponse(url="/ui/dashboard")

@app.get("/dashboard", response_class=RedirectResponse, tags=["Root"])
async def dashboard_redirect():
    """Redirect /dashboard to /ui/dashboard"""
    return RedirectResponse(url="/ui/dashboard")

@app.get("/api/v1/status", tags=["General API"], response_model=response_models.StandardResponse)
async def get_status_api():
    return response_models.StandardResponse(
        success=True,
        message="RL Trading Platform API is running",
        data={"timestamp": datetime.now().isoformat()}
    )

@app.get("/api/v1/config/{config_name}", response_model=response_models.ConfigResponse, tags=["Configuration Management API"])
async def read_config_api(config_name: str, service: services.ConfigService = Depends(get_config_service)):
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
        logger.exception(f"Unexpected error fetching config {config_name} via API")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching config '{config_name}'.")

@app.post("/api/v1/config/{config_name}", response_model=response_models.ConfigResponse, tags=["Configuration Management API"])
async def update_config_api(
    config_name: str,
    new_config_data: Dict[str, Any],
    service: services.ConfigService = Depends(get_config_service)
):
    if config_name not in VALID_CONFIG_NAMES:
        raise HTTPException(status_code=404, detail=f"Config '{config_name}' not found. Valid names are: {VALID_CONFIG_NAMES}")
    PydanticModel = PYDANTIC_MODEL_MAP[config_name]
    try:
        validated_config = PydanticModel(**new_config_data)
        data_to_save = validated_config.model_dump(mode='json')
        updated_data = service.update_config(config_name, data_to_save)
        return response_models.ConfigResponse(
            success=True, config_name=config_name, config_data=updated_data,
            message=f"{config_name} updated successfully via API. API restart may be needed for changes to take full effect in running pipelines."
        )
    except Exception as e:
        logger.exception(f"Error updating {config_name} via API")
        raise HTTPException(status_code=400, detail=f"Error updating {config_name}: {str(e)}")

# --- UI Endpoints ---

@app.get("/ui/dashboard", response_class=HTMLResponse, tags=["UI"])
async def ui_dashboard(request: Request):
    context = {"request": request, "title": "Trading System Dashboard", "config_names": VALID_CONFIG_NAMES}
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/ui/config/{config_name}", response_class=HTMLResponse, tags=["UI"])
async def ui_edit_config_form(
    request: Request,
    config_name: str,
    service: services.ConfigService = Depends(get_config_service),
    message: Optional[str] = None,
    success: Optional[bool] = None
):
    if config_name not in VALID_CONFIG_NAMES:
        raise HTTPException(status_code=404, detail=f"Configuration editor for '{config_name}' not found.")
    pydantic_model_name = PYDANTIC_MODEL_MAP[config_name].__name__
    try:
        current_config_dict = service.get_config(config_name)
        current_config_yaml = yaml.dump(current_config_dict, sort_keys=False, default_flow_style=False, indent=2)
        editable_config_yaml = current_config_yaml
    except Exception as e:
        logger.error(f"Error loading config {config_name} for UI: {e}")
        current_config_yaml = f"# Error loading configuration: {str(e)}"
        editable_config_yaml = "# Error loading configuration. Check server logs."
        message = message or f"Error loading current configuration: {str(e)}"
        success = False
    context = {
        "request": request,
        "config_name": config_name,
        "config_name_title": config_name.replace("_", " ").title(),
        "current_config_yaml": current_config_yaml,
        "editable_config_yaml": editable_config_yaml,
        "pydantic_model_name": pydantic_model_name,
        "message": message,
        "success": success
    }
    return templates.TemplateResponse("config_editor.html", context)

@app.post("/ui/config/{config_name}", response_class=HTMLResponse, tags=["UI"])
async def ui_update_config_submit(
    request: Request,
    config_name: str,
    config_data_yaml: str = Form(...),
    service: services.ConfigService = Depends(get_config_service)
):
    if config_name not in VALID_CONFIG_NAMES:
        raise HTTPException(status_code=404, detail=f"Cannot update unknown config '{config_name}'.")
    PydanticModel = PYDANTIC_MODEL_MAP[config_name]
    message = None
    success = False
    current_yaml_for_display = config_data_yaml
    editable_yaml = config_data_yaml
    try:
        new_config_dict = yaml.safe_load(config_data_yaml)
        if not isinstance(new_config_dict, dict):
            raise ValueError("Invalid YAML structure. Must be a dictionary (key-value pairs).")
        validated_config = PydanticModel(**new_config_dict)
        data_to_save = validated_config.model_dump(mode='json')
        service.update_config(config_name, data_to_save)
        message = f"{config_name.replace('_', ' ').title()} updated successfully! API may need restart for Orchestrator to use new values."
        success = True
        current_yaml_for_display = yaml.dump(data_to_save, sort_keys=False, default_flow_style=False, indent=2)
        editable_yaml = current_yaml_for_display
    except yaml.YAMLError as ye:
        logger.error(f"YAML parsing error updating {config_name} from UI: {ye}", exc_info=True)
        message = f"Error parsing your YAML input: {str(ye)}"
    except Exception as e:
        logger.error(f"Validation or other error updating {config_name} from UI: {e}", exc_info=True)
        message = f"Error saving configuration: {str(e)}"
    context = {
        "request": request,
        "config_name": config_name,
        "config_name_title": config_name.replace("_", " ").title(),
        "current_config_yaml": current_yaml_for_display,
        "editable_config_yaml": editable_yaml,
        "pydantic_model_name": PydanticModel.__name__,
        "message": message,
        "success": success
    }
    return templates.TemplateResponse("config_editor.html", context)

@app.get("/ui/train", response_class=HTMLResponse, tags=["UI"])
async def ui_run_training_form(
    request: Request,
    message: Optional[str] = None,
    success: Optional[bool] = None,
    task_id: Optional[str] = None
):
    today = datetime.now()
    default_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d') + " 23:59:59"
    context = {
        "request": request,
        "title": "Run Training Pipeline",
        "message": message,
        "success": success,
        "task_id": task_id,
        "default_start_date": default_start_date,
        "default_end_date": default_end_date
    }
    return templates.TemplateResponse("run_training.html", context)

@app.post("/ui/train", response_class=HTMLResponse, tags=["UI"])
async def ui_trigger_training_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    symbol: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    interval: str = Form(...),
    use_cached_data: Optional[bool] = Form(False),
    continue_from_model: Optional[str] = Form(None),
    run_evaluation_after_train: Optional[bool] = Form(False),
    eval_start_date: Optional[str] = Form(None),
    eval_end_date: Optional[str] = Form(None),
    eval_interval: Optional[str] = Form(None)
):
    message = None
    success = None
    task_id_resp = None
    train_request_data = request_models.TrainPipelineRequest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cached_data=bool(use_cached_data),
        continue_from_model=continue_from_model if continue_from_model else None,
        run_evaluation_after_train=bool(run_evaluation_after_train),
        eval_start_date=eval_start_date if eval_start_date else None,
        eval_end_date=eval_end_date if eval_end_date else None,
        eval_interval=eval_interval if eval_interval else None
    )
    logger.info(f"UI request to train pipeline: {train_request_data.model_dump_json(indent=2)}")
    if orchestrator is None:
        message = "Orchestrator service is not available. Cannot start training."
        success = False
    else:
        task_id = str(uuid.uuid4())
        tasks_status[task_id] = {
            "status": "PENDING",
            "start_time": datetime.now().isoformat(),
            "result": None,
            "error": None,
            "description": f"Training for symbol {train_request_data.symbol} via UI"
        }
        def training_task_wrapper(task_id_local: str, req: request_models.TrainPipelineRequest):
            tasks_status[task_id_local]["status"] = "RUNNING"
            tasks_status[task_id_local]["start_time"] = datetime.now().isoformat()
            logger.info(f"Background UI training task {task_id_local} started for {req.symbol}.")
            try:
                trained_model_path = orchestrator.run_training_pipeline(
                    symbol=req.symbol, start_date=req.start_date, end_date=req.end_date,
                    interval=req.interval, use_cached_data=req.use_cached_data,
                    continue_from_model=req.continue_from_model,
                    run_evaluation_after_train=req.run_evaluation_after_train,
                    eval_start_date=req.eval_start_date, eval_end_date=req.eval_end_date,
                    eval_interval=req.eval_interval
                )
                if trained_model_path:
                    tasks_status[task_id_local]["status"] = "COMPLETED"
                    tasks_status[task_id_local]["result"] = {"model_path": trained_model_path}
                    logger.info(f"Background UI training task {task_id_local} completed. Model: {trained_model_path}")
                else:
                    tasks_status[task_id_local]["status"] = "FAILED"
                    tasks_status[task_id_local]["error"] = "Training pipeline (UI) completed but no model path returned."
                    logger.error(f"Background UI training task {task_id_local} failed: No model path.")
            except Exception as e_task:
                logger.exception(f"Exception in background UI training task {task_id_local}: {e_task}")
                tasks_status[task_id_local]["status"] = "FAILED"
                tasks_status[task_id_local]["error"] = str(e_task)
            tasks_status[task_id_local]["end_time"] = datetime.now().isoformat()
        background_tasks.add_task(training_task_wrapper, task_id, train_request_data)
        message = f"Training pipeline initiated in background. Task ID: {task_id}"
        success = True
        task_id_resp = task_id
    today = datetime.now()
    default_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d') + " 23:59:59"
    context = {
        "request": request,
        "title": "Run Training Pipeline",
        "message": message,
        "success": success,
        "task_id": task_id_resp,
        "default_start_date": default_start_date,
        "default_end_date": default_end_date
    }
    return templates.TemplateResponse("run_training.html", context)

@app.get("/ui/tasks/{task_id}", response_class=HTMLResponse, tags=["UI"])
async def ui_task_status_page(request: Request, task_id: str):
    if not tasks_status.get(task_id):
        logger.warning(f"UI request for unknown task_id: {task_id}. Page will be rendered, JS will handle API call.")
    context = {
        "request": request,
        "title": f"Task Status: {task_id}",
        "task_id": task_id
    }
    return templates.TemplateResponse("task_status.html", context)

# --- Pipeline Triggering Endpoints ---

@app.post("/api/v1/pipelines/train", response_model=response_models.TrainPipelineResponse, tags=["Pipelines API"])
async def trigger_train_pipeline(
    request_data: request_models.TrainPipelineRequest,
    background_tasks: BackgroundTasks,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator)
):
    logger.info(f"Received API request to train pipeline with params: {request_data.model_dump_json(indent=2)}")
    if orchestrator is None:
        logger.error("OrchestratorAgent not available for training pipeline.")
        raise HTTPException(status_code=503, detail="Orchestrator service is not available. Cannot start training.")
    task_id = str(uuid.uuid4())
    tasks_status[task_id] = {
        "status": "PENDING",
        "start_time": datetime.now().isoformat(),
        "result": None,
        "error": None,
        "description": f"Training for symbol {request_data.symbol}"
    }
    def training_task_wrapper(task_id: str, req: request_models.TrainPipelineRequest):
        tasks_status[task_id]["status"] = "RUNNING"
        tasks_status[task_id]["start_time"] = datetime.now().isoformat()
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
    background_tasks.add_task(training_task_wrapper, task_id, request_data)
    return response_models.TrainPipelineResponse(
        success=True,
        message=f"Training pipeline initiated in background. Task ID: {task_id}",
        task_id=task_id
    )

@app.get("/api/v1/pipelines/train/status/{task_id}", response_model=response_models.TaskStatusResponse, tags=["Pipelines API"])
async def get_training_task_status(task_id: str):
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
        error=task_info.get("error"),
        message="Task status retrieved."
    )

@app.post("/api/v1/pipelines/evaluate", response_model=response_models.EvaluatePipelineResponse, tags=["Pipelines API"])
async def trigger_evaluate_pipeline(
    request_data: request_models.EvaluatePipelineRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator)
):
    logger.info(f"Received API request to evaluate pipeline with params: {request_data.model_dump_json(indent=2)}")
    if orchestrator is None:
        logger.error("OrchestratorAgent not available for evaluation pipeline.")
        raise HTTPException(status_code=503, detail="Orchestrator service is not available. Cannot start evaluation.")
    actual_model_path_to_check = request_data.model_path
    if not os.path.exists(actual_model_path_to_check) and not os.path.exists(actual_model_path_to_check + ".dummy"):
        logger.error(f"Model path for evaluation does not exist: {actual_model_path_to_check}")
        raise HTTPException(status_code=404, detail=f"Model file not found at: {actual_model_path_to_check}")
    elif not os.path.exists(actual_model_path_to_check) and os.path.exists(actual_model_path_to_check + ".dummy"):
        logger.info(f"Actual model file {actual_model_path_to_check} not found, but dummy file {actual_model_path_to_check + '.dummy'} exists. Proceeding.")
    try:
        algo_name = orchestrator.model_params_config.get('algorithm_name', 'DQN')
        eval_metrics = orchestrator.run_evaluation_pipeline(
            symbol=request_data.symbol,
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            interval=request_data.interval,
            model_path=request_data.model_path,
            use_cached_data=request_data.use_cached_data
        )
        if eval_metrics is not None:
            report_path = eval_metrics.pop("report_txt_path", None)
            trade_log_path = eval_metrics.pop("report_trades_csv_path", None)
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
