# src/api/response_models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union

class StandardResponse(BaseModel):
    """Standard response model for most API calls."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class ConfigResponse(StandardResponse):
    """Response model for fetching configuration."""
    config_name: str
    config_data: Dict[str, Any] # The actual configuration data

class TrainPipelineResponse(StandardResponse):
    """Response model for the training pipeline endpoint."""
    task_id: Optional[str] = Field(None, description="Task ID if training is run asynchronously.")
    model_path: Optional[str] = Field(None, description="Path to the saved trained model if run synchronously and successfully.")
    evaluation_results_after_train: Optional[Dict[str, Any]] = Field(None, description="Results if evaluation was run after training.")

class EvaluatePipelineResponse(StandardResponse):
    """Response model for the evaluation pipeline endpoint."""
    metrics: Optional[Dict[str, Any]] = Field(None, description="Calculated evaluation metrics.")
    report_path: Optional[str] = Field(None, description="Path to the generated evaluation report.")
    trade_log_path: Optional[str] = Field(None, description="Path to the saved trade log from evaluation.")

class ErrorDetail(BaseModel):
    loc: Optional[List[Union[str, int]]] = None
    msg: str
    type: str

class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: bool = False
    error: Dict[str, Any] = Field(..., example={"code": "PIPELINE_FAILED", "message": "Detailed error message here."})
    details: Optional[List[ErrorDetail]] = None # For FastAPI validation errors


if __name__ == "__main__":
    # Example usage:
    success_resp_data = {
        "success": True,
        "message": "Operation successful.",
        "data": {"key": "value"}
    }
    success_response = StandardResponse(**success_resp_data)
    print("Sample StandardResponse (Success):")
    print(success_response.model_dump_json(indent=2))

    config_resp_data = {
        "config_name": "main_config",
        "config_data": {"paths": {"model_save_dir": "models/"}},
        "message": "Main config fetched"
    }
    config_response = ConfigResponse(**config_resp_data)
    print("\nSample ConfigResponse:")
    print(config_response.model_dump_json(indent=2))

    train_resp_data = {
        "message": "Training pipeline completed.",
        "model_path": "models/DUMMYTRAIN/DQN_20230101_120000/DQN_final_20230101_120000.zip",
        "evaluation_results_after_train": {"total_return_pct": 5.2}
    }
    train_response = TrainPipelineResponse(**train_resp_data)
    print("\nSample TrainPipelineResponse:")
    print(train_response.model_dump_json(indent=2))

    eval_resp_data = {
        "message": "Evaluation successful.",
        "metrics": {"sharpe_ratio": 1.5, "max_drawdown_pct": -10.2},
        "report_path": "reports/eval_DUMMYTRAIN_summary_20230101_120500.txt"
    }
    eval_response = EvaluatePipelineResponse(**eval_resp_data)
    print("\nSample EvaluatePipelineResponse:")
    print(eval_response.model_dump_json(indent=2))
    
    error_resp_data = {
        "success": False, # Explicitly set for clarity, though default is True in StandardResponse
        "error": {"code": "VALIDATION_ERROR", "message": "Input validation failed."},
        "details": [{"loc": ["body", "symbol"], "msg": "Field required", "type": "missing"}]
    }
    error_response = ErrorResponse(**error_resp_data)
    print("\nSample ErrorResponse:")
    print(error_response.model_dump_json(indent=2))

# Note: In response_models.py, ErrorDetail and ErrorResponse are structured to be somewhat 
# compatible with FastAPI's default validation error responses, but you can customize them further. 
# The StandardResponse includes success: bool = True by default; for error responses, you'd typically 
# set this to False and populate the error field. The ErrorResponse model explicitly sets success: bool = False. 
# I've used Union for ErrorDetail.loc as FastAPI's location can be a list of strings or integers.