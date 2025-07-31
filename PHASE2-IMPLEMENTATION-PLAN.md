# 🚀 **PHASE 2 IMPLEMENTATION PLAN**
**Polygon.io Live Data Integration**  
**Date**: July 31, 2025 - 12:30 PM  
**Total Duration**: 90 minutes  
**Parallel Execution**: While 200K training continues  

---

## 🎯 **IMPLEMENTATION STRATEGY**

### **✅ GUIDING PRINCIPLES**
- **P-1**: Exploit what already works (vault, TimescaleDB, REST fetchers)
- **P-2**: Incremental cut-over (REST fallback until WebSocket proven)
- **P-3**: Everything observable (health metrics to TimescaleDB)
- **P-4**: No double-writes (Redis → TimescaleDB batch writes)

### **🏗️ TARGET ARCHITECTURE**
```
Polygon WebSocket → Router → Redis Streams → TimescaleDB
                                    ↓
                            Feature Pipeline → Model → IB Gateway
```

---

## 📋 **PHASE 2A: INFRASTRUCTURE SETUP (40 minutes)**

### **Step 2A-1: Redis Container Deployment (10 minutes)**
**Owner**: Infrastructure  
**Dependencies**: None  
**Risk**: Low  

#### **Actions:**
1. **Extend docker-compose.timescale.yml**
```yaml
# Add to services section:
redis_cache:
  image: redis:7-alpine
  container_name: trading_redis
  restart: always
  ports:
    - "6379:6379"
  command: redis-server --maxmemory 200mb --maxmemory-policy allkeys-lru
  volumes:
    - redis_data:/data
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 30s
    timeout: 10s
    retries: 3

volumes:
  redis_data:
```

2. **Deploy Redis container**
```bash
cd /home/cristian/IntradayTrading/ITS
docker compose -f docker-compose.timescale.yml up -d redis_cache
```

3. **Verify Redis connectivity**
```bash
docker exec trading_redis redis-cli ping
# Expected: PONG
```

#### **Validation:**
- [ ] Redis container running
- [ ] Port 6379 accessible
- [ ] Memory limit 200MB enforced
- [ ] Health check passing

---

### **Step 2A-2: WebSocket Router Implementation (15 minutes)**
**Owner**: Data Engineering  
**Dependencies**: Redis container  
**Risk**: Medium  

#### **Actions:**
1. **Install WebSocket dependencies**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate
pip install websockets redis
```

2. **Create polygon_ws_router.py**
```python
#!/usr/bin/env python3
"""
Polygon WebSocket Router
Real-time tick data from Polygon.io to Redis Streams
"""

import os
import json
import asyncio
import websockets
import redis
import time
import logging
from datetime import datetime
from secrets_helper import SecretsHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolygonWebSocketRouter:
    def __init__(self):
        # Get API key from vault
        self.api_key = SecretsHelper.get_polygon_api_key()
        self.ws_url = f"wss://socket.polygon.io/stocks"
        
        # Redis connection
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
        
        # Metrics tracking
        self.message_count = 0
        self.last_metric_time = time.time()
        
    async def connect_and_stream(self):
        """Main WebSocket connection and streaming loop"""
        try:
            async with websockets.connect(
                self.ws_url, 
                ping_interval=30,
                ping_timeout=10
            ) as websocket:
                
                # Authenticate
                auth_msg = {
                    "action": "auth",
                    "params": self.api_key
                }
                await websocket.send(json.dumps(auth_msg))
                
                # Subscribe to NVDA and MSFT ticks and aggregates
                subscribe_msg = {
                    "action": "subscribe",
                    "params": "T.NVDA,T.MSFT,A.NVDA,A.MSFT"
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                logger.info("✅ Connected to Polygon WebSocket, subscribed to NVDA/MSFT")
                
                # Message processing loop
                async for message in websocket:
                    await self.process_message(message)
                    
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
            raise
    
    async def process_message(self, raw_message):
        """Process incoming WebSocket message"""
        try:
            ts_start = time.time()
            
            # Parse message
            messages = json.loads(raw_message)
            if not isinstance(messages, list):
                messages = [messages]
            
            # Process each message in the batch
            for msg in messages:
                msg_type = msg.get('ev')
                
                if msg_type == 'T':  # Trade tick
                    await self.handle_trade_tick(msg)
                elif msg_type == 'A':  # Aggregate (minute bar)
                    await self.handle_aggregate(msg)
                elif msg_type == 'status':
                    logger.info(f"📊 Status: {msg}")
            
            # Update metrics
            latency_ms = (time.time() - ts_start) * 1000
            self.update_metrics(latency_ms, len(messages))
            
        except Exception as e:
            logger.error(f"❌ Message processing error: {e}")
    
    async def handle_trade_tick(self, tick):
        """Handle individual trade tick"""
        try:
            # Add to Redis stream with TTL
            stream_data = {
                'symbol': tick.get('sym'),
                'price': tick.get('p'),
                'size': tick.get('s'),
                'timestamp': tick.get('t'),
                'exchange': tick.get('x', ''),
                'conditions': json.dumps(tick.get('c', [])),
                'type': 'trade'
            }
            
            # Add to ticks stream (high frequency, short retention)
            self.redis_client.xadd(
                'ticks', 
                stream_data, 
                maxlen=500000  # ~1 trading day
            )
            
        except Exception as e:
            logger.error(f"❌ Trade tick error: {e}")
    
    async def handle_aggregate(self, agg):
        """Handle minute aggregate"""
        try:
            # Add to aggregates stream (lower frequency, longer retention)
            stream_data = {
                'symbol': agg.get('sym'),
                'open': agg.get('o'),
                'high': agg.get('h'),
                'low': agg.get('l'),
                'close': agg.get('c'),
                'volume': agg.get('v'),
                'timestamp': agg.get('t'),
                'type': 'aggregate'
            }
            
            # Add to aggregates stream
            self.redis_client.xadd(
                'agg_minute', 
                stream_data, 
                maxlen=100000  # ~2 months of minute bars
            )
            
        except Exception as e:
            logger.error(f"❌ Aggregate error: {e}")
    
    def update_metrics(self, latency_ms, message_count):
        """Update performance metrics"""
        try:
            self.message_count += message_count
            current_time = time.time()
            
            # Update Redis metrics
            metrics = {
                'router.latency_ms': latency_ms,
                'router.messages_total': self.message_count,
                'router.last_update': current_time
            }
            
            self.redis_client.hset('metrics', mapping=metrics)
            
            # Log periodic stats
            if current_time - self.last_metric_time > 60:  # Every minute
                msg_per_sec = self.message_count / (current_time - self.last_metric_time)
                logger.info(f"📊 Router: {msg_per_sec:.1f} msg/s, {latency_ms:.1f}ms latency")
                self.last_metric_time = current_time
                self.message_count = 0
                
        except Exception as e:
            logger.error(f"❌ Metrics error: {e}")

async def main():
    """Main entry point"""
    router = PolygonWebSocketRouter()
    
    while True:
        try:
            await router.connect_and_stream()
        except Exception as e:
            logger.error(f"❌ Router crashed: {e}")
            logger.info("🔄 Restarting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
```

3. **Test WebSocket connection**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate
python polygon_ws_router.py
# Should connect and start streaming
```

#### **Validation:**
- [ ] WebSocket connects to Polygon
- [ ] Authentication successful
- [ ] NVDA/MSFT subscription active
- [ ] Messages flowing to Redis streams
- [ ] Latency metrics updating

---

### **Step 2A-3: TimescaleDB Loader Enhancement (10 minutes)**
**Owner**: Data Engineering  
**Dependencies**: Redis streams active  
**Risk**: Low  

#### **Actions:**
1. **Create enhanced loader: redis_to_timescale.py**
```python
#!/usr/bin/env python3
"""
Redis to TimescaleDB Loader
Batch processing from Redis streams to TimescaleDB
"""

import os
import sys
import json
import time
import redis
import psycopg2
import psycopg2.extras
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from secrets_helper import SecretsHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisToTimescaleLoader:
    def __init__(self):
        # Redis connection
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
        
        # TimescaleDB connection
        db_url = SecretsHelper.get_database_url()
        self.pg_conn = psycopg2.connect(db_url)
        self.pg_conn.autocommit = False
        
        # Processing state
        self.last_processed = {'agg_minute': '0-0'}
        
    def process_aggregates_batch(self):
        """Process batch of minute aggregates from Redis to TimescaleDB"""
        try:
            # Read from Redis stream
            streams = self.redis_client.xread(
                {'agg_minute': self.last_processed['agg_minute']}, 
                count=500, 
                block=5000  # 5 second timeout
            )
            
            if not streams:
                return 0
            
            # Process messages
            rows = []
            last_id = None
            
            for stream_name, messages in streams:
                for message_id, fields in messages:
                    last_id = message_id
                    
                    # Convert Redis message to database row
                    row = (
                        datetime.fromtimestamp(int(fields['timestamp']) / 1000),  # timestamp
                        fields['symbol'],  # symbol
                        float(fields['open']),  # open
                        float(fields['high']),  # high
                        float(fields['low']),  # low
                        float(fields['close']),  # close
                        int(fields['volume']),  # volume
                        'polygon_websocket',  # source
                        datetime.now()  # created_at
                    )
                    rows.append(row)
            
            if rows:
                # Batch insert to TimescaleDB
                with self.pg_conn.cursor() as cursor:
                    psycopg2.extras.execute_values(
                        cursor,
                        """
                        INSERT INTO market_data 
                        (timestamp, symbol, open, high, low, close, volume, source, created_at)
                        VALUES %s
                        ON CONFLICT (timestamp, symbol) DO NOTHING
                        """,
                        rows,
                        template=None,
                        page_size=100
                    )
                
                self.pg_conn.commit()
                self.last_processed['agg_minute'] = last_id
                
                logger.info(f"✅ Loaded {len(rows)} aggregates to TimescaleDB")
                
                # Update metrics
                self.redis_client.hset('metrics', mapping={
                    'loader.last_batch_size': len(rows),
                    'loader.last_update': time.time()
                })
            
            return len(rows)
            
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            self.pg_conn.rollback()
            return 0
    
    def run_continuous(self):
        """Run continuous processing loop"""
        logger.info("🔄 Starting continuous Redis → TimescaleDB loader")
        
        while True:
            try:
                processed = self.process_aggregates_batch()
                
                if processed == 0:
                    time.sleep(1)  # Brief pause if no data
                    
            except KeyboardInterrupt:
                logger.info("👋 Shutting down loader")
                break
            except Exception as e:
                logger.error(f"❌ Loader error: {e}")
                time.sleep(5)  # Pause on error
        
        self.pg_conn.close()

if __name__ == "__main__":
    loader = RedisToTimescaleLoader()
    loader.run_continuous()
```

2. **Test loader functionality**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate
python redis_to_timescale.py &
# Should start processing Redis streams
```

#### **Validation:**
- [ ] Loader connects to Redis and TimescaleDB
- [ ] Batch processing working
- [ ] Data appearing in market_data table
- [ ] Metrics updating in Redis

---

### **Step 2A-4: Health Metrics System (5 minutes)**
**Owner**: Operations  
**Dependencies**: Redis and TimescaleDB active  
**Risk**: Low  

#### **Actions:**
1. **Create sys_metrics table in TimescaleDB**
```sql
-- Connect to TimescaleDB
docker exec -it timescaledb_primary psql -U postgres -d trading_data

-- Create metrics table
CREATE TABLE IF NOT EXISTS sys_metrics (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    service VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable('sys_metrics', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_sys_metrics_service_metric 
ON sys_metrics (service, metric_name, timestamp DESC);
```

2. **Create metrics collector: collect_metrics.py**
```python
#!/usr/bin/env python3
"""
System Metrics Collector
Collects metrics from Redis and stores in TimescaleDB
"""

import time
import redis
import psycopg2
import logging
from datetime import datetime
from secrets_helper import SecretsHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        db_url = SecretsHelper.get_database_url()
        self.pg_conn = psycopg2.connect(db_url)
        
    def collect_and_store(self):
        """Collect metrics from Redis and store in TimescaleDB"""
        try:
            # Get all metrics from Redis
            metrics = self.redis_client.hgetall('metrics')
            
            if not metrics:
                return
            
            # Prepare batch insert
            rows = []
            timestamp = datetime.now()
            
            for key, value in metrics.items():
                if key.startswith(('router.', 'loader.')):
                    service = key.split('.')[0]
                    metric_name = key.split('.', 1)[1]
                    
                    try:
                        metric_value = float(value)
                        rows.append((timestamp, service, metric_name, metric_value, '{}'))
                    except ValueError:
                        continue
            
            if rows:
                with self.pg_conn.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO sys_metrics 
                        (timestamp, service, metric_name, metric_value, tags)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        rows
                    )
                self.pg_conn.commit()
                logger.info(f"✅ Stored {len(rows)} metrics")
                
        except Exception as e:
            logger.error(f"❌ Metrics collection error: {e}")
    
    def run_continuous(self):
        """Run continuous metrics collection"""
        while True:
            try:
                self.collect_and_store()
                time.sleep(30)  # Collect every 30 seconds
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Collector error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    collector = MetricsCollector()
    collector.run_continuous()
```

#### **Validation:**
- [ ] sys_metrics table created
- [ ] Metrics collector running
- [ ] Metrics appearing in TimescaleDB
- [ ] 30-second collection interval working

---

## 📋 **PHASE 2B: LIVE INTEGRATION (35 minutes)**

### **Step 2B-1: Feature Pipeline Redis Integration (15 minutes)**
**Owner**: Quantitative  
**Dependencies**: Redis streams active  
**Risk**: Medium  

#### **Actions:**
1. **Create live feature pipeline: live_feature_pipeline.py**
```python
#!/usr/bin/env python3
"""
Live Feature Pipeline
Real-time feature engineering from Redis streams
"""

import redis
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveFeaturePipeline:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.feature_cache = {}
        self.lookback_window = 50  # Same as training
        
    def get_latest_features(self, symbol: str) -> Optional[np.ndarray]:
        """Get latest engineered features for a symbol"""
        try:
            # Try Redis cache first (hot path)
            cache_key = f"features:{symbol}:latest"
            cached = self.redis_client.get(cache_key)
            
            if cached:
                features = np.frombuffer(cached, dtype=np.float32)
                logger.debug(f"✅ Cache hit for {symbol} features")
                return features
            
            # Cache miss - compute from recent data
            features = self._compute_features_from_stream(symbol)
            
            if features is not None:
                # Cache for 5 seconds
                self.redis_client.setex(
                    cache_key, 
                    5, 
                    features.tobytes()
                )
                logger.debug(f"✅ Computed and cached {symbol} features")
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature pipeline error for {symbol}: {e}")
            return None
    
    def _compute_features_from_stream(self, symbol: str) -> Optional[np.ndarray]:
        """Compute features from Redis stream data"""
        try:
            # Get recent aggregates from Redis
            stream_data = self.redis_client.xrevrange(
                'agg_minute', 
                count=self.lookback_window
            )
            
            if len(stream_data) < self.lookback_window:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(stream_data)} < {self.lookback_window}")
                return None
            
            # Convert to DataFrame
            rows = []
            for msg_id, fields in stream_data:
                if fields.get('symbol') == symbol:
                    rows.append({
                        'timestamp': datetime.fromtimestamp(int(fields['timestamp']) / 1000),
                        'open': float(fields['open']),
                        'high': float(fields['high']),
                        'low': float(fields['low']),
                        'close': float(fields['close']),
                        'volume': int(fields['volume'])
                    })
            
            if len(rows) < self.lookback_window:
                return None
            
            df = pd.DataFrame(rows).sort_values('timestamp')
            
            # Compute technical indicators (same as training)
            features = self._compute_technical_indicators(df)
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Stream computation error: {e}")
            return None
    
    def _compute_technical_indicators(self, df: pd.DataFrame) -> np.ndarray:
        """Compute technical indicators (matching training pipeline)"""
        try:
            features = []
            
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            
            # Volatility
            df['volatility'] = df['returns'].rolling(20).std()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # VWAP
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Time features
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            
            # Select final features (12 features to match training)
            feature_columns = [
                'returns', 'log_returns', 'sma_5', 'sma_20', 'ema_12',
                'volatility', 'rsi', 'volume_ratio', 'vwap',
                'hour', 'minute', 'close'
            ]
            
            # Get latest values
            latest_features = df[feature_columns].iloc[-1].values
            
            # Handle NaN values
            latest_features = np.nan_to_num(latest_features, nan=0.0)
            
            return latest_features
            
        except Exception as e:
            logger.error(f"❌ Technical indicators error: {e}")
            return np.zeros(12, dtype=np.float32)

# Test the pipeline
if __name__ == "__main__":
    pipeline = LiveFeaturePipeline()
    
    while True:
        try:
            for symbol in ['NVDA', 'MSFT']:
                features = pipeline.get_latest_features(symbol)
                if features is not None:
                    logger.info(f"📊 {symbol} features: {features[:3]}... (shape: {features.shape})")
                else:
                    logger.warning(f"⚠️ No features available for {symbol}")
            
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            break
```

#### **Validation:**
- [ ] Feature pipeline connects to Redis
- [ ] Technical indicators computed correctly
- [ ] Features cached with 5-second TTL
- [ ] Output matches training format (12 features)

---

### **Step 2B-2: Live Model Inference (10 minutes)**
**Owner**: Quantitative  
**Dependencies**: Feature pipeline, 200K model ready  
**Risk**: Medium  

#### **Actions:**
1. **Create live inference engine: live_inference.py**
```python
#!/usr/bin/env python3
"""
Live Model Inference
Real-time predictions using 200K trained model
"""

import os
import sys
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from live_feature_pipeline import LiveFeaturePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveInferenceEngine:
    def __init__(self):
        self.feature_pipeline = LiveFeaturePipeline()
        self.model = None
        self.vec_env = None
        self.lstm_states = None
        self.episode_starts = None
        
        # Load latest 200K model
        self._load_latest_model()
    
    def _load_latest_model(self):
        """Load the latest 200K trained model"""
        try:
            # Find latest checkpoint
            checkpoint_dir = Path("models/checkpoints")
            checkpoints = list(checkpoint_dir.glob("dual_ticker_200k_*.zip"))
            
            if not checkpoints:
                raise FileNotFoundError("No 200K checkpoints found")
            
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            vecnorm_path = "models/dual_ticker_enhanced_50k_vecnorm.pkl"
            
            logger.info(f"📊 Loading model: {latest_checkpoint.name}")
            
            # Create dummy environment for model loading
            from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
            
            # Minimal data for model loading
            dummy_data = np.random.randn(1000, 12).astype(np.float32)
            dummy_prices = pd.Series(np.random.randn(1000) * 100 + 500)
            dummy_days = pd.date_range('2024-01-01', periods=1000, freq='1min')
            
            def make_env():
                env = DualTickerTradingEnv(
                    nvda_data=dummy_data,
                    msft_data=dummy_data,
                    nvda_prices=dummy_prices,
                    msft_prices=dummy_prices,
                    trading_days=dummy_days,
                    initial_capital=100000
                )
                return env
            
            env = DummyVecEnv([make_env])
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False  # Inference mode
            
            # Load model
            self.model = RecurrentPPO.load(latest_checkpoint, env=env)
            self.vec_env = env
            
            # Initialize LSTM states
            self.lstm_states = None
            self.episode_starts = np.ones((1,), dtype=bool)
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            raise
    
    def predict_action(self, nvda_features: np.ndarray, msft_features: np.ndarray, 
                      current_positions: Dict[str, float]) -> Tuple[int, float]:
        """Predict trading action for current market state"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Construct observation (26 dims: 12 NVDA + 1 pos + 12 MSFT + 1 pos)
            observation = np.concatenate([
                nvda_features,  # 12 features
                [current_positions.get('NVDA', 0.0)],  # 1 position
                msft_features,  # 12 features
                [current_positions.get('MSFT', 0.0)]   # 1 position
            ]).reshape(1, -1)
            
            # Normalize observation
            observation = self.vec_env.normalize_obs(observation)
            
            # Get model prediction
            action, self.lstm_states = self.model.predict(
                observation,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True  # Use deterministic policy for live trading
            )
            
            # Update episode starts for next prediction
            self.episode_starts = np.zeros((1,), dtype=bool)
            
            # Get action probabilities for confidence
            obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(obs_tensor)
                action_probs = distribution.distribution.probs
                confidence = float(action_probs[0, action[0]])
            
            logger.info(f"🎯 Predicted action: {action[0]} (confidence: {confidence:.3f})")
            
            return int(action[0]), confidence
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return 4, 0.0  # Default to HOLD_BOTH with zero confidence
    
    def get_action_description(self, action: int) -> str:
        """Get human-readable action description"""
        action_map = {
            0: "SELL_BOTH",
            1: "SELL_NVDA_HOLD_MSFT", 
            2: "SELL_NVDA_BUY_MSFT",
            3: "HOLD_NVDA_SELL_MSFT",
            4: "HOLD_BOTH",
            5: "HOLD_NVDA_BUY_MSFT",
            6: "BUY_NVDA_SELL_MSFT",
            7: "BUY_NVDA_HOLD_MSFT",
            8: "BUY_BOTH"
        }
        return action_map.get(action, "UNKNOWN")

# Test inference
if __name__ == "__main__":
    import time
    import pandas as pd
    
    engine = LiveInferenceEngine()
    
    # Mock current positions
    positions = {'NVDA': 0.0, 'MSFT': 0.0}
    
    while True:
        try:
            # Get latest features
            nvda_features = engine.feature_pipeline.get_latest_features('NVDA')
            msft_features = engine.feature_pipeline.get_latest_features('MSFT')
            
            if nvda_features is not None and msft_features is not None:
                action, confidence = engine.predict_action(nvda_features, msft_features, positions)
                action_desc = engine.get_action_description(action)
                
                logger.info(f"🤖 Live Prediction: {action_desc} (confidence: {confidence:.3f})")
            else:
                logger.warning("⚠️ Insufficient feature data for prediction")
            
            time.sleep(30)  # Predict every 30 seconds
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"❌ Inference loop error: {e}")
            time.sleep(10)
```

#### **Validation:**
- [ ] Latest 200K model loads successfully
- [ ] LSTM states initialized correctly
- [ ] Predictions generated with confidence scores
- [ ] Action descriptions human-readable

---

### **Step 2B-3: Paper Trading Integration (10 minutes)**
**Owner**: Execution  
**Dependencies**: Live inference, IB Gateway  
**Risk**: High  

#### **Actions:**
1. **Create paper trading controller: paper_trading_controller.py**
```python
#!/usr/bin/env python3
"""
Paper Trading Controller
Execute model predictions via IB Gateway paper trading
"""

import time
import logging
from datetime import datetime
from typing import Dict, Optional
from live_inference import LiveInferenceEngine
from src.brokers.ib_gateway import IBGatewayClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperTradingController:
    def __init__(self):
        self.inference_engine = LiveInferenceEngine()
        self.ib_client = IBGatewayClient()
        self.positions = {'NVDA': 0.0, 'MSFT': 0.0}
        self.portfolio_value = 100000.0
        self.last_action_time = {}
        self.min_action_interval = 60  # Minimum 60 seconds between actions
        
    def connect_broker(self) -> bool:
        """Connect to IB Gateway"""
        try:
            success = self.ib_client.connect()
            if success:
                logger.info("✅ Connected to IB Gateway (Paper Trading)")
                return True
            else:
                logger.error("❌ Failed to connect to IB Gateway")
                return False
        except Exception as e:
            logger.error(f"❌ Broker connection error: {e}")
            return False
    
    def execute_trading_decision(self, action: int, confidence: float) -> bool:
        """Execute trading decision based on model prediction"""
        try:
            current_time = time.time()
            action_desc = self.inference_engine.get_action_description(action)
            
            # Check minimum interval between actions
            if action in self.last_action_time:
                time_since_last = current_time - self.last_action_time[action]
                if time_since_last < self.min_action_interval:
                    logger.debug(f"⏳ Action {action_desc} on cooldown ({time_since_last:.1f}s)")
                    return False
            
            # Check confidence threshold
            if confidence < 0.6:  # Only execute high-confidence predictions
                logger.debug(f"🤔 Low confidence {confidence:.3f} for {action_desc}, skipping")
                return False
            
            # Execute action
            executed = False
            
            if action == 0:  # SELL_BOTH
                executed = self._execute_sell_both()
            elif action == 1:  # SELL_NVDA_HOLD_MSFT
                executed = self._execute_sell_symbol('NVDA')
            elif action == 2:  # SELL_NVDA_BUY_MSFT
                executed = self._execute_sell_symbol('NVDA') and self._execute_buy_symbol('MSFT')
            elif action == 3:  # HOLD_NVDA_SELL_MSFT
                executed = self._execute_sell_symbol('MSFT')
            elif action == 4:  # HOLD_BOTH
                executed = True  # No action needed
            elif action == 5:  # HOLD_NVDA_BUY_MSFT
                executed = self._execute_buy_symbol('MSFT')
            elif action == 6:  # BUY_NVDA_SELL_MSFT
                executed = self._execute_buy_symbol('NVDA') and self._execute_sell_symbol('MSFT')
            elif action == 7:  # BUY_NVDA_HOLD_MSFT
                executed = self._execute_buy_symbol('NVDA')
            elif action == 8:  # BUY_BOTH
                executed = self._execute_buy_both()
            
            if executed:
                self.last_action_time[action] = current_time
                logger.info(f"✅ Executed: {action_desc} (confidence: {confidence:.3f})")
            
            return executed
            
        except Exception as e:
            logger.error(f"❌ Trading execution error: {e}")
            return False
    
    def _execute_buy_symbol(self, symbol: str) -> bool:
        """Execute buy order for symbol"""
        try:
            # Get current price
            price = self.ib_client.get_current_price(symbol)
            if price is None:
                return False
            
            # Calculate position size (simple: 10% of portfolio per symbol)
            position_value = self.portfolio_value * 0.1
            shares = int(position_value / price)
            
            if shares > 0:
                order_id = self.ib_client.place_market_order(symbol, shares, 'BUY')
                if order_id:
                    self.positions[symbol] += shares
                    logger.info(f"📈 BUY {shares} {symbol} @ ${price:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Buy execution error for {symbol}: {e}")
            return False
    
    def _execute_sell_symbol(self, symbol: str) -> bool:
        """Execute sell order for symbol"""
        try:
            current_position = self.positions.get(symbol, 0)
            if current_position <= 0:
                return True  # Nothing to sell
            
            price = self.ib_client.get_current_price(symbol)
            if price is None:
                return False
            
            order_id = self.ib_client.place_market_order(symbol, current_position, 'SELL')
            if order_id:
                self.positions[symbol] = 0
                logger.info(f"📉 SELL {current_position} {symbol} @ ${price:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Sell execution error for {symbol}: {e}")
            return False
    
    def _execute_buy_both(self) -> bool:
        """Execute buy orders for both symbols"""
        return self._execute_buy_symbol('NVDA') and self._execute_buy_symbol('MSFT')
    
    def _execute_sell_both(self) -> bool:
        """Execute sell orders for both symbols"""
        return self._execute_sell_symbol('NVDA') and self._execute_sell_symbol('MSFT')
    
    def update_portfolio_value(self):
        """Update current portfolio value"""
        try:
            total_value = 0.0
            
            for symbol, shares in self.positions.items():
                if shares > 0:
                    price = self.ib_client.get_current_price(symbol)
                    if price:
                        total_value += shares * price
            
            # Add cash (simplified)
            cash = self.portfolio_value - sum(
                self.positions[symbol] * self.ib_client.get_current_price(symbol, 0)
                for symbol in self.positions
            )
            
            self.portfolio_value = total_value + cash
            logger.info(f"💰 Portfolio: ${self.portfolio_value:.2f} | Positions: {self.positions}")
            
        except Exception as e:
            logger.error(f"❌ Portfolio update error: {e}")
    
    def run_live_trading(self):
        """Main live trading loop"""
        if not self.connect_broker():
            return
        
        logger.info("🚀 Starting live paper trading...")
        
        while True:
            try:
                # Get latest features
                nvda_features = self.inference_engine.feature_pipeline.get_latest_features('NVDA')
                msft_features = self.inference_engine.feature_pipeline.get_latest_features('MSFT')
                
                if nvda_features is not None and msft_features is not None:
                    # Get model prediction
                    action, confidence = self.inference_engine.predict_action(
                        nvda_features, msft_features, self.positions
                    )
                    
                    # Execute trading decision
                    self.execute_trading_decision(action, confidence)
                    
                    # Update portfolio
                    self.update_portfolio_value()
                else:
                    logger.warning("⚠️ Insufficient data for trading decision")
                
                time.sleep(30)  # Trade every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("👋 Shutting down live trading")
                break
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                time.sleep(10)
        
        self.ib_client.disconnect()

if __name__ == "__main__":
    controller = PaperTradingController()
    controller.run_live_trading()
```

#### **Validation:**
- [ ] IB Gateway connection successful
- [ ] Paper trading mode confirmed
- [ ] Order execution working
- [ ] Portfolio tracking accurate
- [ ] Risk controls active (confidence threshold, cooldown)

---

## 📋 **PHASE 2C: VALIDATION & MONITORING (15 minutes)**

### **Step 2C-1: Cross-Source Price Validation (10 minutes)**
**Owner**: Quality Assurance  
**Dependencies**: Polygon and IB Gateway active  
**Risk**: Low  

#### **Actions:**
1. **Create price validation: price_validator.py**
```python
#!/usr/bin/env python3
"""
Cross-Source Price Validation
Compare prices between Polygon and IB Gateway
"""

import time
import logging
from datetime import datetime
from typing import Dict, Optional
from live_feature_pipeline import LiveFeaturePipeline
from src.brokers.ib_gateway import IBGatewayClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceValidator:
    def __init__(self):
        self.feature_pipeline = LiveFeaturePipeline()
        self.ib_client = IBGatewayClient()
        self.price_differences = []
        self.max_difference_pct = 0.1  # 0.1% maximum acceptable difference
        
    def validate_prices(self) -> Dict[str, Dict]:
        """Validate prices between Polygon and IB Gateway"""
        results = {}
        
        for symbol in ['NVDA', 'MSFT']:
            try:
                # Get Polygon price (from latest tick)
                polygon_price = self._get_polygon_price(symbol)
                
                # Get IB Gateway price
                ib_price = self.ib_client.get_current_price(symbol)
                
                if polygon_price and ib_price:
                    difference_pct = abs(polygon_price - ib_price) / polygon_price * 100
                    
                    results[symbol] = {
                        'polygon_price': polygon_price,
                        'ib_price': ib_price,
                        'difference_pct': difference_pct,
                        'acceptable': difference_pct <= self.max_difference_pct,
                        'timestamp': datetime.now()
                    }
                    
                    if difference_pct > self.max_difference_pct:
                        logger.warning(f"⚠️ {symbol} price difference: {difference_pct:.3f}% "
                                     f"(Polygon: ${polygon_price:.2f}, IB: ${ib_price:.2f})")
                    else:
                        logger.debug(f"✅ {symbol} prices aligned: {difference_pct:.3f}% difference")
                
            except Exception as e:
                logger.error(f"❌ Price validation error for {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def _get_polygon_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Polygon data"""
        try:
            # Get latest tick from Redis
            stream_data = self.feature_pipeline.redis_client.xrevrange(
                'ticks', 
                count=10
            )
            
            for msg_id, fields in stream_data:
                if fields.get('symbol') == symbol and fields.get('type') == 'trade':
                    return float(fields['price'])
            
            # Fallback to aggregate data
            agg_data = self.feature_pipeline.redis_client.xrevrange(
                'agg_minute',
                count=5
            )
            
            for msg_id, fields in agg_data:
                if fields.get('symbol') == symbol:
                    return float(fields['close'])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Polygon price fetch error: {e}")
            return None
    
    def run_continuous_validation(self):
        """Run continuous price validation"""
        logger.info("🔍 Starting continuous price validation...")
        
        while True:
            try:
                results = self.validate_prices()
                
                # Log summary
                acceptable_count = sum(1 for r in results.values() 
                                     if isinstance(r, dict) and r.get('acceptable', False))
                total_count = len([r for r in results.values() if isinstance(r, dict) and 'error' not in r])
                
                if total_count > 0:
                    logger.info(f"📊 Price validation: {acceptable_count}/{total_count} within tolerance")
                
                time.sleep(60)  # Validate every minute
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Validation loop error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    validator = PriceValidator()
    validator.ib_client.connect()
    validator.run_continuous_validation()
```

#### **Validation:**
- [ ] Price comparison working
- [ ] Acceptable difference threshold enforced
- [ ] Alerts for price discrepancies
- [ ] Continuous monitoring active

---

### **Step 2C-2: Executive Dashboard Integration (5 minutes)**
**Owner**: Operations  
**Dependencies**: All systems active  
**Risk**: Low  

#### **Actions:**
1. **Create live dashboard: live_executive_dashboard.py**
```python
#!/usr/bin/env python3
"""
Live Executive Dashboard
Real-time monitoring of live trading system
"""

import time
import json
import redis
import psycopg2
from datetime import datetime, timedelta
from secrets_helper import SecretsHelper

class LiveExecutiveDashboard:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        db_url = SecretsHelper.get_database_url()
        self.pg_conn = psycopg2.connect(db_url)
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'websocket_router': self._check_websocket_status(),
            'data_loader': self._check_loader_status(),
            'feature_pipeline': self._check_feature_status(),
            'model_inference': self._check_inference_status(),
            'paper_trading': self._check_trading_status(),
            'price_validation': self._check_validation_status()
        }
        return status
    
    def _check_websocket_status(self) -> dict:
        """Check WebSocket router status"""
        try:
            metrics = self.redis_client.hgetall('metrics')
            last_update = float(metrics.get('router.last_update', 0))
            current_time = time.time()
            
            return {
                'status': 'healthy' if (current_time - last_update) < 60 else 'stale',
                'latency_ms': float(metrics.get('router.latency_ms', 0)),
                'messages_total': int(metrics.get('router.messages_total', 0)),
                'last_update': datetime.fromtimestamp(last_update).isoformat()
            }
        except:
            return {'status': 'error'}
    
    def _check_loader_status(self) -> dict:
        """Check data loader status"""
        try:
            metrics = self.redis_client.hgetall('metrics')
            return {
                'status': 'healthy',
                'last_batch_size': int(metrics.get('loader.last_batch_size', 0)),
                'last_update': metrics.get('loader.last_update', 'unknown')
            }
        except:
            return {'status': 'error'}
    
    def _check_feature_status(self) -> dict:
        """Check feature pipeline status"""
        try:
            # Check if features are being generated
            nvda_features = self.redis_client.get('features:NVDA:latest')
            msft_features = self.redis_client.get('features:MSFT:latest')
            
            return {
                'status': 'healthy' if (nvda_features and msft_features) else 'degraded',
                'nvda_available': nvda_features is not None,
                'msft_available': msft_features is not None
            }
        except:
            return {'status': 'error'}
    
    def _check_inference_status(self) -> dict:
        """Check model inference status"""
        # This would need to be implemented based on inference logging
        return {'status': 'unknown', 'note': 'Implement inference metrics'}
    
    def _check_trading_status(self) -> dict:
        """Check paper trading status"""
        # This would need to be implemented based on trading logs
        return {'status': 'unknown', 'note': 'Implement trading metrics'}
    
    def _check_validation_status(self) -> dict:
        """Check price validation status"""
        # This would need to be implemented based on validation logs
        return {'status': 'unknown', 'note': 'Implement validation metrics'}
    
    def print_dashboard(self):
        """Print formatted dashboard"""
        status = self.get_system_status()
        
        print("🎛️  LIVE TRADING EXECUTIVE DASHBOARD")
        print("=" * 60)
        print(f"⏰ Timestamp: {status['timestamp']}")
        print()
        
        for component, details in status.items():
            if component == 'timestamp':
                continue
                
            component_name = component.replace('_', ' ').title()
            status_emoji = {
                'healthy': '✅',
                'degraded': '⚠️',
                'stale': '🟡',
                'error': '❌',
                'unknown': '❓'
            }.get(details.get('status', 'unknown'), '❓')
            
            print(f"{status_emoji} {component_name}: {details.get('status', 'unknown').upper()}")
            
            # Show relevant metrics
            if component == 'websocket_router' and details.get('status') == 'healthy':
                print(f"   Latency: {details.get('latency_ms', 0):.1f}ms")
                print(f"   Messages: {details.get('messages_total', 0):,}")
            
        print()

if __name__ == "__main__":
    dashboard = LiveExecutiveDashboard()
    
    while True:
        try:
            dashboard.print_dashboard()
            time.sleep(30)  # Update every 30 seconds
        except KeyboardInterrupt:
            break
```

#### **Validation:**
- [ ] Dashboard shows all component status
- [ ] Real-time metrics displayed
- [ ] Health indicators accurate
- [ ] 30-second refresh working

---

## 🎯 **FINAL INTEGRATION & TESTING**

### **Step 2C-3: End-to-End System Test (5 minutes)**

#### **Actions:**
1. **Create system orchestrator: start_live_system.sh**
```bash
#!/bin/bash
# Live Trading System Orchestrator

echo "🚀 Starting Live Trading System"
echo "================================"

# Start Redis if not running
if ! docker ps | grep -q trading_redis; then
    echo "📊 Starting Redis..."
    docker compose -f docker-compose.timescale.yml up -d redis_cache
    sleep 5
fi

# Start all components in background
echo "🔄 Starting WebSocket router..."
python polygon_ws_router.py &
WS_PID=$!

echo "📥 Starting data loader..."
python redis_to_timescale.py &
LOADER_PID=$!

echo "📊 Starting metrics collector..."
python collect_metrics.py &
METRICS_PID=$!

echo "🔍 Starting price validator..."
python price_validator.py &
VALIDATOR_PID=$!

echo "🎛️ Starting executive dashboard..."
python live_executive_dashboard.py &
DASHBOARD_PID=$!

# Wait for startup
sleep 10

echo "✅ Live trading system started!"
echo "📋 Component PIDs:"
echo "   WebSocket Router: $WS_PID"
echo "   Data Loader: $LOADER_PID"
echo "   Metrics Collector: $METRICS_PID"
echo "   Price Validator: $VALIDATOR_PID"
echo "   Dashboard: $DASHBOARD_PID"

echo ""
echo "🎯 To start paper trading:"
echo "   python paper_trading_controller.py"

# Keep script running
wait
```

2. **Final system validation**
```bash
chmod +x start_live_system.sh
./start_live_system.sh
```

---

## ✅ **ACCEPTANCE CRITERIA CHECKLIST**

### **📊 METRICS TARGETS**
- [ ] Redis hit-ratio ≥ 0.95 during RTH
- [ ] Tick latency 95-percentile ≤ 150ms
- [ ] No missing minutes vs Polygon REST backfill
- [ ] Cross-source price diff < 0.1%

### **🔧 SYSTEM HEALTH**
- [ ] All Docker containers running
- [ ] WebSocket connection stable
- [ ] Data flowing Redis → TimescaleDB
- [ ] Features generating correctly
- [ ] Model predictions working
- [ ] Paper trading executing

### **📈 BUSINESS READINESS**
- [ ] Executive dashboard operational
- [ ] Metrics collection active
- [ ] Price validation passing
- [ ] Risk controls enforced
- [ ] Rollback procedures tested

---

## 🎯 **IMPLEMENTATION TIMELINE**

| Phase | Duration | Parallel Tasks | Dependencies |
|-------|----------|----------------|--------------|
| **2A-1** | 10 min | Redis deployment | None |
| **2A-2** | 15 min | WebSocket router | Redis ready |
| **2A-3** | 10 min | Data loader | Redis streams |
| **2A-4** | 5 min | Metrics system | TimescaleDB |
| **2B-1** | 15 min | Feature pipeline | Redis data |
| **2B-2** | 10 min | Model inference | 200K model ready |
| **2B-3** | 10 min | Paper trading | IB Gateway |
| **2C-1** | 10 min | Price validation | All sources |
| **2C-2** | 5 min | Dashboard | All systems |

**Total: 90 minutes**

---

## 🚨 **RISK MITIGATION**

### **🔄 ROLLBACK PLAN**
- **Toggle Flag**: `USE_POLYGON_WS=false` reverts to REST API
- **Data Integrity**: Staging table prevents corruption
- **Resource Limits**: <200MB RAM, no GPU impact
- **Training Protection**: 200K training continues unaffected

### **⚠️ MONITORING ALERTS**
- WebSocket disconnection → Slack alert
- Price difference >0.1% → Email alert  
- Redis memory >180MB → Warning
- Missing data gaps → Nightly report

---

**IMPLEMENTATION PLAN COMPLETE**  
**Ready for confirmation and execution** 🚀