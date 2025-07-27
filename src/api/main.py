import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging to stdout with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from fastapi import FastAPI, BackgroundTasks, HTTPException, APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime, timedelta

from src.execution.orchestrator_agent import OrchestratorAgent
from src.shared.duckdb_manager import close_all_duckdb_connections
from src.shared.feature_store import reset_feature_store
from src.api.monitoring import router as monitoring_router
from src.api.live_monitoring import router as live_monitoring_router

# Prometheus metrics endpoint support
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# FeatureStore monitoring endpoints - disable during training to avoid DuckDB conflicts
try:
    # Check environment variable to disable monitoring during training
    import os
    disable_monitoring = os.getenv("DISABLE_FEATURESTORE_MONITORING", "false").lower() == "true"
    
    if disable_monitoring:
        # Monitoring disabled via environment variable
        MONITORING_AVAILABLE = False
        print("⚠️  FeatureStore monitoring endpoints disabled via DISABLE_FEATURESTORE_MONITORING=true")
    else:
        from .monitoring_endpoints import monitoring_router
        MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Fee schedule endpoints
try:
    from .fee_endpoints import router as fee_router
    FEE_ENDPOINTS_AVAILABLE = True
except ImportError:
    FEE_ENDPOINTS_AVAILABLE = False

# Stress testing endpoints
try:
    from .stress_endpoints import router as stress_router
    STRESS_ENDPOINTS_AVAILABLE = True
except ImportError:
    STRESS_ENDPOINTS_AVAILABLE = False

# --- In-memory task store ---
task_store: Dict[str, Dict[str, Any]] = {}

# --- Template Setup ---
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
if not os.path.isdir(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- FastAPI app ---
app = FastAPI(title="RL Platform API")
router = APIRouter()

# --- OrchestratorAgent instance (singleton for demo) ---
def get_orchestrator(use_training_config: bool = False):
    """Get or create orchestrator instance"""
    # Clean up any existing connections and feature store instances
    close_all_duckdb_connections()
    reset_feature_store()
    
    risk_limits_path = "config/risk_limits_training.yaml" if use_training_config else "config/risk_limits_orchestrator_test.yaml"
    
    logger = logging.getLogger("API.OrchestratorFactory")
    config_type = "TRAINING (relaxed limits)" if use_training_config else "LIVE (strict limits)"
    logger.info(f"Creating orchestrator with {config_type} - Risk config: {risk_limits_path}")
    
    return OrchestratorAgent(
        main_config_path="config/main_config_orchestrator_test.yaml",
        model_params_path="config/model_params_orchestrator_test.yaml",
        risk_limits_path=risk_limits_path,
        read_only=True  # API should only read from feature store
    )

# Default orchestrator for non-training operations
orchestrator = get_orchestrator()

# --- Request/Response Models ---
class TrainingRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    interval: str
    use_cached_data: Optional[bool] = False
    continue_from_model: Optional[str] = None

class EvaluationRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    interval: str
    model_path: str
    use_cached_data: Optional[bool] = False

class WalkForwardRequest(BaseModel):
    symbol: str

class LiveTradingRequest(BaseModel):
    symbol: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None

# --- Synchronous wrappers for orchestrator methods ---
def run_training_pipeline_sync(task_id: str, req: TrainingRequest):
    try:
        task_store[task_id]['status'] = 'running'
        
        # Use training-specific orchestrator with relaxed risk limits
        training_orchestrator = get_orchestrator(use_training_config=True)
        
        result = training_orchestrator.run_training_pipeline(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            use_cached_data=req.use_cached_data,
            continue_from_model=req.continue_from_model
        )
        task_store[task_id]['status'] = 'completed'
        task_store[task_id]['result'] = result
    except Exception as e:
        task_store[task_id]['status'] = 'failed'
        task_store[task_id]['error'] = str(e)

def run_evaluation_pipeline_sync(task_id: str, req: EvaluationRequest):
    try:
        task_store[task_id]['status'] = 'running'
        result = orchestrator.run_evaluation_pipeline(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            model_path=req.model_path,
            use_cached_data=req.use_cached_data
        )
        task_store[task_id]['status'] = 'completed'
        task_store[task_id]['result'] = result
    except Exception as e:
        task_store[task_id]['status'] = 'failed'
        task_store[task_id]['error'] = str(e)

def run_walk_forward_sync(task_id: str, req: WalkForwardRequest):
    try:
        task_store[task_id]['status'] = 'running'
        result = orchestrator.run_walk_forward_evaluation(symbol=req.symbol)
        task_store[task_id]['status'] = 'completed'
        task_store[task_id]['result'] = result
    except Exception as e:
        task_store[task_id]['status'] = 'failed'
        task_store[task_id]['error'] = str(e)

def run_scheduled_retrain_sync(task_id: str):
    try:
        task_store[task_id]['status'] = 'running'
        result = orchestrator.schedule_weekly_retrain()
        task_store[task_id]['status'] = 'completed'
        task_store[task_id]['result'] = result
    except Exception as e:
        task_store[task_id]['status'] = 'failed'
        task_store[task_id]['error'] = str(e)

def run_live_trading_sync(task_id: str, req: LiveTradingRequest):
    try:
        task_store[task_id]['status'] = 'running'
        result = orchestrator.run_live_trading(symbol=req.symbol)
        task_store[task_id]['status'] = 'completed'
        task_store[task_id]['result'] = result
    except Exception as e:
        task_store[task_id]['status'] = 'failed'
        task_store[task_id]['error'] = str(e)

# --- API Endpoints ---

@router.post("/train", response_model=TaskStatusResponse)
def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_store[task_id] = {'status': 'pending', 'result': None, 'error': None}
    background_tasks.add_task(run_training_pipeline_sync, task_id, request)
    return TaskStatusResponse(task_id=task_id, status='pending')

@router.post("/evaluate", response_model=TaskStatusResponse)
def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_store[task_id] = {'status': 'pending', 'result': None, 'error': None}
    background_tasks.add_task(run_evaluation_pipeline_sync, task_id, request)
    return TaskStatusResponse(task_id=task_id, status='pending')

@router.post("/walk_forward", response_model=TaskStatusResponse)
def start_walk_forward(request: WalkForwardRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_store[task_id] = {'status': 'pending', 'result': None, 'error': None}
    background_tasks.add_task(run_walk_forward_sync, task_id, request)
    return TaskStatusResponse(task_id=task_id, status='pending')

@router.post("/scheduled_retrain", response_model=TaskStatusResponse)
def start_scheduled_retrain(background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_store[task_id] = {'status': 'pending', 'result': None, 'error': None}
    background_tasks.add_task(run_scheduled_retrain_sync, task_id)
    return TaskStatusResponse(task_id=task_id, status='pending')

@router.post("/live_trading", response_model=TaskStatusResponse)
def start_live_trading(request: LiveTradingRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_store[task_id] = {'status': 'pending', 'result': None, 'error': None}
    background_tasks.add_task(run_live_trading_sync, task_id, request)
    return TaskStatusResponse(task_id=task_id, status='pending')

@router.get("/task_status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return TaskStatusResponse(
        task_id=task_id,
        status=task['status'],
        result=task.get('result'),
        error=task.get('error')
    )

@router.get("/tasks", response_model=List[TaskStatusResponse])
def list_tasks():
    return [
        TaskStatusResponse(
            task_id=tid,
            status=info['status'],
            result=info.get('result'),
            error=info.get('error')
        )
        for tid, info in task_store.items()
    ]

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

@app.get("/health/db")
def db_health_check():
    """Database health check endpoint."""
    try:
        from src.utils.db import health_check, get_db_info
        
        if health_check():
            db_info = get_db_info()
            return {
                "status": "healthy",
                "database": db_info
            }
        else:
            return {
                "status": "unhealthy",
                "error": "Database not accessible"
            }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e)
        }

@app.get("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint for monitoring."""
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content="# Prometheus client not available\n",
            media_type="text/plain"
        )
    
    # Generate metrics in Prometheus format
    metrics_data = generate_latest(REGISTRY)
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/api/v1/reload")
def reload_orchestrator():
    """Reload the orchestrator instance"""
    global orchestrator
    try:
        orchestrator = get_orchestrator()
        return {"status": "success", "message": "Orchestrator reloaded successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to reload orchestrator: {e}"}

# --- UI Endpoints ---

@app.get("/", response_class=RedirectResponse)
async def root():
    """Redirect root URL to dashboard"""
    return RedirectResponse(url="/ui/dashboard")

@app.get("/ui/dashboard", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    context = {
        "request": request, 
        "title": "Trading System Dashboard"
    }
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/ui/train", response_class=HTMLResponse)
async def ui_run_training_form(
    request: Request,
    message: Optional[str] = None,
    success: Optional[bool] = None,
    task_id: Optional[str] = None
):
    """Serves the HTML form for triggering a training pipeline."""
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

@app.post("/ui/train", response_class=HTMLResponse)
async def ui_trigger_training_submit(
    request: Request,
    background_tasks: BackgroundTasks,
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
    """Processes the form submission for triggering a training pipeline."""
    message = None
    success = None
    task_id_resp = None
    
    # Create training request
    train_request = TrainingRequest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cached_data=bool(use_cached_data),
        continue_from_model=continue_from_model if continue_from_model else None
    )
    
    try:
        task_id = str(uuid.uuid4())
        task_store[task_id] = {
            'status': 'pending', 
            'result': None, 
            'error': None,
            'description': f"Training for symbol {train_request.symbol} via UI"
        }
        background_tasks.add_task(run_training_pipeline_sync, task_id, train_request)
        message = f"Training pipeline initiated in background. Task ID: {task_id}"
        success = True
        task_id_resp = task_id
    except Exception as e:
        message = f"Error starting training: {str(e)}"
        success = False
    
    # Re-render form with message
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

@app.get("/ui/nvda-dqn", response_class=HTMLResponse)
async def ui_nvda_dqn_training_form(
    request: Request,
    message: Optional[str] = None,
    success: Optional[bool] = None,
    task_id: Optional[str] = None
):
    """Serves the specialized HTML form for NVDA DQN training."""
    context = {
        "request": request,
        "title": "NVDA DQN Training Pipeline",
        "message": message,
        "success": success,
        "task_id": task_id
    }
    return templates.TemplateResponse("nvda_dqn_training.html", context)

@app.post("/ui/nvda-dqn", response_class=HTMLResponse)
async def ui_nvda_dqn_training_submit(
    request: Request,
    background_tasks: BackgroundTasks,
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
    """Processes the NVDA DQN training form submission."""
    message = None
    success = None
    task_id_resp = None
    
    # Create training request
    train_request = TrainingRequest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cached_data=bool(use_cached_data),
        continue_from_model=continue_from_model if continue_from_model else None
    )
    
    try:
        task_id = str(uuid.uuid4())
        task_store[task_id] = {
            'status': 'pending', 
            'result': None, 
            'error': None,
            'description': f"NVDA DQN Training ({start_date} to {end_date}, {interval})"
        }
        background_tasks.add_task(run_training_pipeline_sync, task_id, train_request)
        message = f"🚀 NVDA DQN Training pipeline initiated! Task ID: {task_id}"
        success = True
        task_id_resp = task_id
    except Exception as e:
        message = f"❌ Error starting NVDA DQN training: {str(e)}"
        success = False
    
    # Re-render form with message
    context = {
        "request": request,
        "title": "NVDA DQN Training Pipeline",
        "message": message,
        "success": success,
        "task_id": task_id_resp
    }
    return templates.TemplateResponse("nvda_dqn_training.html", context)

@app.get("/ui/live-trading", response_class=HTMLResponse)
async def ui_live_trading_page(request: Request):
    """Serves the live trading interface for NVDA."""
    context = {
        "request": request,
        "title": "NVDA Live Trading"
    }
    return templates.TemplateResponse("live_trading_simple.html", context)

@app.get("/ui/tasks/{task_id}", response_class=HTMLResponse)
async def ui_task_status_page(request: Request, task_id: str):
    context = {
        "request": request,
        "title": f"Task Status: {task_id}",
        "task_id": task_id
    }
    return templates.TemplateResponse("task_status.html", context)

@app.get("/ui/tasks", response_class=HTMLResponse)
async def ui_tasks_list(request: Request):
    """Simple task list view"""
    tasks = [
        {
            "task_id": tid,
            "status": info['status'],
            "description": info.get('description', 'N/A'),
            "result": info.get('result'),
            "error": info.get('error')
        }
        for tid, info in task_store.items()
    ]
    
    context = {
        "request": request,
        "title": "All Tasks",
        "tasks": tasks
    }
    
    # Create a simple inline template for task list
    task_list_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Tasks</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        nav ul { list-style-type: none; padding: 0; }
        nav ul li { display: inline; margin-right: 20px; }
        nav ul li a { text-decoration: none; color: blue; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .status-pending { color: orange; }
        .status-running { color: blue; }
        .status-completed { color: green; }
        .status-failed { color: red; }
    </style>
</head>
<body>
    <nav>
        <ul>
            <li><a href="/ui/dashboard">Dashboard</a></li>
            <li><a href="/ui/train">Run Training</a></li>
            <li><a href="/ui/tasks">View All Tasks</a></li>
        </ul>
    </nav>
    <hr>
    
    <h1>All Tasks</h1>
    
    {% if tasks %}
    <table>
        <thead>
            <tr>
                <th>Task ID</th>
                <th>Status</th>
                <th>Description</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {% for task in tasks %}
            <tr>
                <td>{{ task.task_id[:8] }}...</td>
                <td class="status-{{ task.status }}">{{ task.status.upper() }}</td>
                <td>{{ task.description }}</td>
                <td><a href="/ui/tasks/{{ task.task_id }}">View Details</a></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p>No tasks found.</p>
    {% endif %}
    
    <p><a href="/ui/train">Start New Training</a></p>
</body>
</html>
    """
    
    from fastapi.responses import HTMLResponse
    from jinja2 import Template
    
    template = Template(task_list_html)
    html_content = template.render(**context)
    return HTMLResponse(content=html_content)

# Mount routers
app.include_router(router, prefix="/orchestrator")

# Include dual-ticker monitoring endpoints
app.include_router(monitoring_router)
app.include_router(live_monitoring_router)
print("✅ Dual-ticker monitoring endpoints enabled at /monitoring/*")

# Include FeatureStore monitoring endpoints if available
if MONITORING_AVAILABLE:
    from .monitoring_endpoints import monitoring_router as fs_monitoring_router
    app.include_router(fs_monitoring_router, prefix="/api/v1")
    print("✅ FeatureStore monitoring endpoints enabled at /api/v1/monitoring/*")
else:
    print("⚠️  FeatureStore monitoring endpoints not available")

# Include fee schedule endpoints if available
if FEE_ENDPOINTS_AVAILABLE:
    app.include_router(fee_router)
    print("✅ Fee schedule endpoints enabled at /api/v1/fees/*")
else:
    print("⚠️  Fee schedule endpoints not available")

# Include stress testing endpoints if available
if STRESS_ENDPOINTS_AVAILABLE:
    app.include_router(stress_router)
    print("✅ Stress testing endpoints enabled at /api/v1/stress/*")
else:
    print("⚠️  Stress testing endpoints not available")
# --- Example FastAPI run (for local testing) ---
if __name__ == "__main__":
    import sys
    import uvicorn
    
    # Check for --dry-run flag
    if "--dry-run" in sys.argv:
        print("🔍 Running API server in dry-run mode...")
        try:
            from src.utils.db import health_check, get_db_info
            
            print("📊 Performing database health check...")
            if health_check():
                db_info = get_db_info()
                print(f"✅ Database accessible: {db_info['db_path']}")
                print(f"📈 Database size: {db_info['db_size_bytes']} bytes")
                print(f"📋 Tables: {db_info['table_count']}")
                print("🎯 Dry-run completed successfully!")
                sys.exit(0)
            else:
                print("❌ Database health check failed!")
                sys.exit(1)
        except Exception as e:
            print(f"💥 Dry-run failed: {e}")
            sys.exit(1)
    
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
     