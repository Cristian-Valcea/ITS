# src/api/request_models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class PipelineBaseRequest(BaseModel):
    """Base model for pipeline requests, includes common parameters."""
    symbol: str = Field(..., example="SPY", description="Stock symbol (e.g., AAPL, MSFT).")
    start_date: str = Field(..., example="2023-01-01", description="Start date for data (YYYY-MM-DD).")
    end_date: str = Field(..., example="2023-12-31", description="End date for data (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS for IBKR).")
    interval: str = Field("1min", example="1min", description="Data interval (e.g., '1min', '5mins', '1day').")
    use_cached_data: bool = Field(False, description="Use cached raw data if available.")
    # use_cached_features: bool = Field(False, description="Use cached processed features if available.") # Future enhancement

class TrainPipelineRequest(PipelineBaseRequest):
    """Request model for triggering the training pipeline."""
    continue_from_model: Optional[str] = Field(None, example="models/archive/model_xyz.zip", description="Path to a pre-trained model to continue training from.")
    run_evaluation_after_train: bool = Field(True, description="Automatically run evaluation pipeline after training.")
    eval_start_date: Optional[str] = Field(None, example="2024-01-01", description="Start date for post-training evaluation data. Defaults to day after training end_date.")
    eval_end_date: Optional[str] = Field(None, example="2024-03-31", description="End date for post-training evaluation data. Defaults to eval_start_date + N days from config.")
    eval_interval: Optional[str] = Field(None, example="1min", description="Interval for post-training evaluation data. Defaults to training interval.")


class EvaluatePipelineRequest(PipelineBaseRequest):
    """Request model for triggering the evaluation pipeline."""
    model_path: str = Field(..., example="models/prod/best_model.zip", description="Path to the trained model to evaluate.")
    # `symbol`, `start_date`, `end_date`, `interval`, `use_cached_data` are inherited for the evaluation data window.

# For updating configurations, the Pydantic models from config_models.py will be used directly
# as the request body structure. For example, to update main_config.yaml, the request body
# would conform to `config_models.MainConfig`.

# Example of a more generic config update model if needed, though using specific models is better for validation.
# class ConfigUpdateRequest(BaseModel):
#     config_name: str = Field(..., example="main_config", description="Name of the config to update (main_config, model_params, risk_limits)")
#     data: Dict[str, Any] = Field(..., description="New configuration data as a dictionary")


if __name__ == "__main__":
    # Example usage:
    train_req_data = {
        "symbol": "AAPL",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "interval": "1day",
        "use_cached_data": True,
        "continue_from_model": None,
        "run_evaluation_after_train": False 
    }
    train_request = TrainPipelineRequest(**train_req_data)
    print("Sample TrainPipelineRequest:")
    print(train_request.model_dump_json(indent=2))

    eval_req_data = {
        "symbol": "MSFT",
        "start_date": "2023-01-01",
        "end_date": "2023-03-31",
        "interval": "1hour",
        "model_path": "models/msft_dqn_v1.zip",
        "use_cached_data": True
    }
    eval_request = EvaluatePipelineRequest(**eval_req_data)
    print("\nSample EvaluatePipelineRequest:")
    print(eval_request.model_dump_json(indent=2))
