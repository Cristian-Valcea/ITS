from fastapi import FastAPI, BackgroundTasks, HTTPException, APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import os
from datetime import datetime, timedelta

from src.agents.orchestrator_agent import OrchestratorAgent

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
orchestrator = OrchestratorAgent(
    main_config_path="config/main_config_orchestrator_test.yaml",
    model_params_path="config/model_params_orchestrator_test.yaml",
    risk_limits_path="config/risk_limits_orchestrator_test.yaml"
)

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
        result = orchestrator.run_training_pipeline(
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
    return {"status": "ok"}

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

# Mount router
app.include_router(router, prefix="/orchestrator")
# --- Example FastAPI run (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
     