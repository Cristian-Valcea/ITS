-- TimescaleDB Initialization Script
-- Sets up dual-ticker trading database with hypertables and logical replication

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create trading schema
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set default search path
ALTER DATABASE trading_data SET search_path TO trading, public;

-- Create market_data hypertable for OHLCV data
CREATE TABLE IF NOT EXISTS trading.market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open DECIMAL(10,4) NOT NULL,
    high DECIMAL(10,4) NOT NULL,
    low DECIMAL(10,4) NOT NULL,
    close DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    source VARCHAR(50) DEFAULT 'polygon',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT market_data_ohlc_check CHECK (
        high >= GREATEST(open, close) AND 
        low <= LEAST(open, close) AND
        volume >= 0
    )
);

-- Convert to hypertable (partitioned by time)
SELECT create_hypertable(
    'trading.market_data', 
    'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
    ON trading.market_data (symbol, timestamp DESC);
    
CREATE INDEX IF NOT EXISTS idx_market_data_source_time
    ON trading.market_data (source, timestamp DESC);

-- Create data quality reports table
CREATE TABLE IF NOT EXISTS trading.data_quality_reports (
    id SERIAL PRIMARY KEY,
    report_date DATE NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) NOT NULL,
    total_checks INTEGER NOT NULL,
    passed_checks INTEGER NOT NULL,
    failed_checks INTEGER NOT NULL,
    pass_rate DECIMAL(5,2) NOT NULL,
    environment VARCHAR(20) DEFAULT 'production',
    details JSONB,
    
    CONSTRAINT valid_status CHECK (status IN ('PASS', 'FAIL', 'ERROR'))
);

-- Convert quality reports to hypertable
SELECT create_hypertable(
    'trading.data_quality_reports',
    'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Create orders table for OMS integration
CREATE TABLE IF NOT EXISTS trading.orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity DECIMAL(10,4) NOT NULL CHECK (quantity > 0),
    order_type VARCHAR(20) DEFAULT 'MARKET',
    price DECIMAL(10,4),
    status VARCHAR(20) DEFAULT 'PENDING',
    filled_quantity DECIMAL(10,4) DEFAULT 0,
    avg_fill_price DECIMAL(10,4),
    commission DECIMAL(8,4) DEFAULT 0,
    source VARCHAR(50) DEFAULT 'algorithm',
    metadata JSONB,
    
    CONSTRAINT valid_order_status CHECK (
        status IN ('PENDING', 'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED')
    )
);

-- Orders hypertable  
SELECT create_hypertable(
    'trading.orders',
    'timestamp', 
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Create positions table
CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(10,4) NOT NULL,
    avg_cost DECIMAL(10,4) NOT NULL,
    market_value DECIMAL(12,4),
    unrealized_pnl DECIMAL(12,4),
    realized_pnl DECIMAL(12,4) DEFAULT 0,
    
    UNIQUE(symbol, timestamp)
);

-- Positions hypertable
SELECT create_hypertable(
    'trading.positions',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day', 
    if_not_exists => TRUE
);

-- Create analytics views for dual-ticker analysis
CREATE OR REPLACE VIEW analytics.dual_ticker_summary AS
SELECT 
    DATE(timestamp) as trading_date,
    symbol,
    COUNT(*) as bar_count,
    FIRST(open, timestamp) as day_open,
    MAX(high) as day_high,
    MIN(low) as day_low,
    LAST(close, timestamp) as day_close,
    SUM(volume) as total_volume,
    STDDEV(close) as price_volatility
FROM trading.market_data
WHERE symbol IN ('NVDA', 'MSFT')
GROUP BY DATE(timestamp), symbol
ORDER BY trading_date DESC, symbol;

-- Create correlation analysis view
CREATE OR REPLACE VIEW analytics.nvda_msft_correlation AS
WITH daily_returns AS (
    SELECT 
        DATE(timestamp) as trading_date,
        symbol,
        LAST(close, timestamp) as close_price,
        LAG(LAST(close, timestamp)) OVER (PARTITION BY symbol ORDER BY DATE(timestamp)) as prev_close
    FROM trading.market_data 
    WHERE symbol IN ('NVDA', 'MSFT')
    GROUP BY DATE(timestamp), symbol
),
returns_calc AS (
    SELECT 
        trading_date,
        symbol,
        close_price,
        CASE 
            WHEN prev_close IS NOT NULL AND prev_close > 0 
            THEN (close_price - prev_close) / prev_close 
            ELSE NULL 
        END as daily_return
    FROM daily_returns
)
SELECT 
    trading_date,
    CORR(
        CASE WHEN symbol = 'NVDA' THEN daily_return END,
        CASE WHEN symbol = 'MSFT' THEN daily_return END
    ) OVER (ORDER BY trading_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as rolling_30day_correlation
FROM returns_calc
GROUP BY trading_date
ORDER BY trading_date DESC;

-- Create monitoring functions
CREATE OR REPLACE FUNCTION monitoring.check_data_freshness()
RETURNS TABLE(
    symbol VARCHAR(10),
    last_update TIMESTAMPTZ,
    hours_since_update DECIMAL(5,2),
    status VARCHAR(10)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        md.symbol,
        MAX(md.timestamp) as last_update,
        EXTRACT(EPOCH FROM (NOW() - MAX(md.timestamp))) / 3600 as hours_since_update,
        CASE 
            WHEN MAX(md.timestamp) > NOW() - INTERVAL '1 day' THEN 'FRESH'
            WHEN MAX(md.timestamp) > NOW() - INTERVAL '3 days' THEN 'STALE'  
            ELSE 'OLD'
        END as status
    FROM trading.market_data md
    WHERE md.symbol IN ('NVDA', 'MSFT')
    GROUP BY md.symbol
    ORDER BY md.symbol;
END;
$$ LANGUAGE plpgsql;

-- Create data quality check function
CREATE OR REPLACE FUNCTION monitoring.validate_ohlc_data()
RETURNS TABLE(
    symbol VARCHAR(10),
    invalid_bars BIGINT,
    total_bars BIGINT,
    error_rate DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        md.symbol,
        COUNT(*) FILTER (
            WHERE NOT (
                md.high >= GREATEST(md.open, md.close) AND 
                md.low <= LEAST(md.open, md.close)
            )
        ) as invalid_bars,
        COUNT(*) as total_bars,
        (COUNT(*) FILTER (
            WHERE NOT (
                md.high >= GREATEST(md.open, md.close) AND 
                md.low <= LEAST(md.open, md.close)
            )
        )::DECIMAL / COUNT(*)::DECIMAL) as error_rate
    FROM trading.market_data md
    WHERE md.timestamp > NOW() - INTERVAL '30 days'
    GROUP BY md.symbol
    ORDER BY md.symbol;
END;
$$ LANGUAGE plpgsql;

-- Set up logical replication publication for replica
CREATE PUBLICATION trading_data_pub FOR ALL TABLES IN SCHEMA trading;

-- Create replication user for replica access
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'replicator') THEN
        CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'replica_secure_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE trading_data TO replicator;
GRANT USAGE ON SCHEMA trading TO replicator;
GRANT SELECT ON ALL TABLES IN SCHEMA trading TO replicator;
ALTER DEFAULT PRIVILEGES IN SCHEMA trading GRANT SELECT ON TABLES TO replicator;

-- Insert initial metadata
INSERT INTO trading.data_quality_reports (
    report_date, status, total_checks, passed_checks, failed_checks, pass_rate, environment, details
) VALUES (
    CURRENT_DATE, 'PASS', 1, 1, 0, 100.00, 'initialization',
    '{"message": "TimescaleDB initialized successfully", "timestamp": "' || NOW() || '"}'
) ON CONFLICT DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✅ TimescaleDB initialization complete';
    RAISE NOTICE '📊 Hypertables created: market_data, data_quality_reports, orders, positions';
    RAISE NOTICE '🔄 Logical replication publication created: trading_data_pub';
    RAISE NOTICE '👤 Replication user created: replicator';
    RAISE NOTICE '📈 Analytics views available: dual_ticker_summary, nvda_msft_correlation';
    RAISE NOTICE '🔍 Monitoring functions: check_data_freshness(), validate_ohlc_data()';
END $$;