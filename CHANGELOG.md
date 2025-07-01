# Changelog

## [v0.3.0] - 2025-07-01 - FastAPI Backend Integration

### 🚀 **Major New Features**
- **Complete FastAPI Backend**: Professional REST API for the trading system
- **Interactive API Documentation**: Auto-generated Swagger UI and ReDoc
- **Configuration Management API**: CRUD operations for all YAML configs
- **Pipeline Control API**: Endpoints to trigger training and evaluation
- **Comprehensive Data Validation**: Pydantic models for all requests/responses

### 🔧 **API Endpoints Implemented**
- **Status**: `GET /api/v1/status` - Health check
- **Configuration Management**:
  - `GET /api/v1/config/{config_name}` - Retrieve configurations
  - `POST /api/v1/config/main_config` - Update main configuration
  - `POST /api/v1/config/model_params` - Update model parameters
  - `POST /api/v1/config/risk_limits` - Update risk limits
- **Pipeline Control**:
  - `POST /api/v1/pipelines/train` - Trigger training pipeline
  - `POST /api/v1/pipelines/evaluate` - Trigger evaluation pipeline

### 🛠 **Technical Implementation**
- **FastAPI Framework**: Modern async Python web framework
- **Pydantic Models**: Type-safe request/response validation
- **ConfigService**: Centralized YAML configuration management
- **Orchestrator Integration**: Direct integration with existing agent system
- **Error Handling**: Comprehensive HTTP exception handling
- **Auto Documentation**: Interactive API docs at `/docs` and `/redoc`

### 📋 **API Access**
- **Base URL**: http://127.0.0.1:8000
- **Interactive Docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **Launch Script**: `scripts/run_api.ps1` (Windows) / `scripts/run_api.sh` (Linux)

### ✅ **Integration Status**
- All agents initialize successfully through API
- Configuration system fully functional via REST endpoints
- Training and evaluation pipelines accessible via API
- Comprehensive error handling and validation

## [v0.2.0] - 2025-07-01 - Bug Fixes and Environment Setup

### 🎯 **Major Fixes**
- **Fixed FeatureAgent AttributeError**: Resolved `'FeatureAgent' object has no attribute 'feature_list'` error
- **Fixed YAML Configuration Issues**: Removed invalid markdown code blocks (```) from all config files
- **Fixed VWAP Calculation**: Resolved pandas groupby ambiguity error in daily VWAP calculation
- **Fixed Deprecated Pandas Warning**: Updated `fillna(method='ffill')` to `ffill()`

### 🔧 **Configuration Improvements**
- **Fixed main_config.yaml**: Corrected YAML structure for `sin_cos_encode` and `time_features`
- **Fixed model_params.yaml**: Removed invalid markdown syntax
- **Fixed risk_limits.yaml**: Cleaned up YAML formatting
- **Enhanced FeatureAgent Configuration**: Improved config parsing with fallback logic

### 🚀 **Development Environment**
- **Virtual Environment Setup**: Complete Python 3.10 virtual environment with all dependencies
- **Convenience Scripts**: Added `activate_venv.ps1` and `run_training.ps1` for easy usage
- **Dependency Management**: All required packages installed and tested

### ✅ **System Status**
- **All Agents Initialize Successfully**: DataAgent, FeatureAgent, EnvAgent, TrainerAgent, etc.
- **Configuration Loading Works**: All YAML files parse correctly
- **Feature Engineering Pipeline**: RSI, EMA, VWAP, Time features configured
- **Training Pipeline Starts**: System progresses to environment creation stage

### 🐛 **Known Issues**
- **Insufficient Training Data**: Current dummy data (3 rows) insufficient for technical indicators
- **Environment Shape Mismatch**: Lookback window configuration needs adjustment for small datasets
- **IBKR Connection**: Interactive Brokers connection not available (expected in development)

### 📋 **Next Steps**
- Implement proper data generation for testing with sufficient historical data
- Adjust technical indicator parameters for small datasets
- Complete environment shape validation
- Add comprehensive unit tests

### 🛠 **Technical Details**
- **Python Version**: 3.10.11
- **Key Dependencies**: TensorFlow, PyTorch, Stable-Baselines3, pandas, numpy, yfinance
- **Configuration**: YAML-based modular configuration system
- **Architecture**: Multi-agent system with orchestrator pattern