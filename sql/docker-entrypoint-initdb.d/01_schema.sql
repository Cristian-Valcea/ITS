-- 01_schema.sql - Auto-loaded by TimescaleDB container
-- Creates dual-ticker schema for NVDA + MSFT trading system

\echo 'Creating dual-ticker trading schema...'

-- Create test database for CI
CREATE DATABASE intradayjules_test;

-- Connect to main database
\c intradayjules;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Main dual-ticker bars table
CREATE TABLE IF NOT EXISTS dual_ticker_bars (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    -- OHLCV data
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    -- Technical indicators
    rsi DECIMAL(8,4),
    ema_short DECIMAL(12,4),
    ema_long DECIMAL(12,4),
    vwap DECIMAL(12,4),
    vwap_deviation DECIMAL(8,4),
    -- Time features (cyclical encoding)
    hour_sin DECIMAL(6,4),
    hour_cos DECIMAL(6,4),
    minute_sin DECIMAL(6,4),
    minute_cos DECIMAL(6,4),
    day_of_week SMALLINT,
    -- Market microstructure
    bid_ask_spread DECIMAL(8,4),
    market_impact DECIMAL(8,4),
    -- Metadata
    data_source VARCHAR(20) DEFAULT 'unknown',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (symbol, timestamp)
);

-- Convert to hypertable (TimescaleDB magic)
SELECT create_hypertable('dual_ticker_bars', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_dual_ticker_symbol_time 
    ON dual_ticker_bars (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_dual_ticker_time_symbol 
    ON dual_ticker_bars (timestamp DESC, symbol);
CREATE INDEX IF NOT EXISTS idx_dual_ticker_source 
    ON dual_ticker_bars (data_source, timestamp DESC);

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS portfolio_positions (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    position INTEGER NOT NULL DEFAULT 0, -- -1, 0, 1
    shares DECIMAL(12,4) NOT NULL DEFAULT 0,
    avg_price DECIMAL(12,4),
    unrealized_pnl DECIMAL(12,4) DEFAULT 0,
    realized_pnl DECIMAL(12,4) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (symbol, timestamp)
);

SELECT create_hypertable('portfolio_positions', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Current positions table for OMS (non-hypertable, current state only)
CREATE TABLE IF NOT EXISTS current_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    qty DECIMAL(12,4) NOT NULL DEFAULT 0,
    avg_price DECIMAL(10,4),
    market_value DECIMAL(12,4),
    unrealized_pnl DECIMAL(12,4) DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Initialize with NVDA and MSFT positions (zero quantities)
INSERT INTO current_positions (symbol, qty, avg_price) 
VALUES 
    ('NVDA', 0, NULL),
    ('MSFT', 0, NULL)
ON CONFLICT (symbol) DO NOTHING;

-- Trading actions log
CREATE TABLE IF NOT EXISTS trading_actions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    action INTEGER NOT NULL, -- 0=SELL, 1=HOLD, 2=BUY
    action_name VARCHAR(10) NOT NULL,
    shares DECIMAL(12,4),
    price DECIMAL(12,4),
    transaction_cost DECIMAL(8,4) DEFAULT 0,
    portfolio_value DECIMAL(12,4),
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trading_actions_time 
    ON trading_actions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_actions_symbol 
    ON trading_actions (symbol, timestamp DESC);

-- Risk metrics table
CREATE TABLE IF NOT EXISTS risk_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    portfolio_value DECIMAL(12,4) NOT NULL,
    daily_pnl DECIMAL(12,4) DEFAULT 0,
    drawdown_pct DECIMAL(8,4) DEFAULT 0,
    max_drawdown_pct DECIMAL(8,4) DEFAULT 0,
    turnover_ratio DECIMAL(8,4) DEFAULT 0,
    sharpe_ratio DECIMAL(8,4),
    correlation_nvda_msft DECIMAL(6,4),
    portfolio_beta DECIMAL(6,4),
    concentration_risk DECIMAL(6,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('risk_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Connect to test database and create same schema
\c intradayjules_test;

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Replicate schema in test database (simplified for CI speed)
CREATE TABLE IF NOT EXISTS dual_ticker_bars (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    rsi DECIMAL(8,4),
    ema_short DECIMAL(12,4),
    ema_long DECIMAL(12,4),
    vwap DECIMAL(12,4),
    hour_sin DECIMAL(6,4),
    hour_cos DECIMAL(6,4),
    minute_sin DECIMAL(6,4),
    minute_cos DECIMAL(6,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (symbol, timestamp)
);

SELECT create_hypertable('dual_ticker_bars', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_test_dual_ticker_symbol_time 
    ON dual_ticker_bars (symbol, timestamp DESC);

-- Create a simple view for quick queries
CREATE OR REPLACE VIEW latest_dual_ticker_data AS
SELECT 
    symbol,
    timestamp,
    close,
    volume,
    rsi,
    vwap,
    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
FROM dual_ticker_bars
WHERE timestamp >= NOW() - INTERVAL '1 hour';

\echo 'Dual-ticker schema created successfully!'
\echo 'Tables: dual_ticker_bars, portfolio_positions, trading_actions, risk_metrics'
\echo 'Hypertables enabled for time-series optimization'
\echo 'Ready for NVDA + MSFT dual-ticker trading system!'