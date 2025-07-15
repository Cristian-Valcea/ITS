# Project: Intraday Python Trading System

## Context
- Language: Python 3.10+
- Libraries used: ccxt, pandas, asyncio, pytest
- Modules: trade_engine.py, signal_generator.py, order_manager.py, trainer_core.py, policy_export.py

- Goal: low-latency intraday trading with model-driven risk management, production-ready TorchScript exports

## Coding Conventions
- Type hints & docstrings for all public APIs
- Black + Ruff formatting
- loguru for logging, structured & async-safe
- CI: pytest with coverage ≥ 85%
- Follow PEP8 naming standards

## Project Structure
- `trade_engine.py`: core trading logic & async order placement
- `signal_generator.py`: computes trading signals
- `order_manager.py`: handles exchange API calls via ccxt
- `trainer_core.py` : trainer core module - handles:
    - Core training coordination
    - Model management and lifecycle
    - Training state management
    - Risk advisor integration
- `policy_export.py` : policy export Core Module - handles:
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