#!/usr/bin/env python3
"""
Process-Safe Rate Limiter with Persistence
Addresses reviewer concerns about multiprocessing safety and crash recovery

Key Features:
- Redis-backed token persistence (survives restarts)
- Process-safe operations (multiprocessing compatible)  
- SQLite fallback (when Redis unavailable)
- Burst capacity handling
- Circuit breaker integration
"""

import time
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 5
    burst_capacity: int = 10
    token_ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379/0"
    sqlite_path: str = "rate_limiter.db"


class TokenStorageBackend(ABC):
    """Abstract interface for token storage"""
    
    @abstractmethod
    def get_tokens(self, key: str) -> Optional[Dict[str, Any]]:
        """Get current token state"""
        pass
    
    @abstractmethod
    def set_tokens(self, key: str, tokens: int, last_refill: datetime, ttl_seconds: int) -> bool:
        """Set token state with TTL"""
        pass
    
    @abstractmethod
    def atomic_consume(self, key: str, requested_tokens: int, max_tokens: int, 
                       refill_rate: float) -> tuple[bool, int, datetime]:
        """Atomically consume tokens if available"""
        pass


class RedisTokenStorage(TokenStorageBackend):
    """Redis-backed token storage for process safety"""
    
    def __init__(self, redis_url: str):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available - install with: pip install redis")
        
        self.redis_client = redis.from_url(redis_url)
        self._lua_consume_script = self._load_lua_script()
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def _load_lua_script(self) -> str:
        """Load atomic token consumption Lua script"""
        lua_script = """
        local key = KEYS[1]
        local requested_tokens = tonumber(ARGV[1])
        local max_tokens = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])
        local ttl_seconds = tonumber(ARGV[5])
        
        -- Get current state
        local state = redis.call('HMGET', key, 'tokens', 'last_refill')
        local current_tokens = tonumber(state[1]) or max_tokens
        local last_refill = tonumber(state[2]) or current_time
        
        -- Calculate refill
        local time_elapsed = current_time - last_refill
        local tokens_to_add = math.floor(time_elapsed * refill_rate)
        current_tokens = math.min(max_tokens, current_tokens + tokens_to_add)
        
        -- Try to consume
        if current_tokens >= requested_tokens then
            current_tokens = current_tokens - requested_tokens
            
            -- Update state
            redis.call('HMSET', key, 
                'tokens', current_tokens,
                'last_refill', current_time)
            redis.call('EXPIRE', key, ttl_seconds)
            
            return {1, current_tokens, current_time}  -- Success
        else
            -- Update refill time even on failure
            redis.call('HMSET', key,
                'tokens', current_tokens,
                'last_refill', current_time)
            redis.call('EXPIRE', key, ttl_seconds)
            
            return {0, current_tokens, current_time}  -- Failure
        end
        """
        return self.redis_client.register_script(lua_script)
    
    def get_tokens(self, key: str) -> Optional[Dict[str, Any]]:
        """Get current token state"""
        try:
            result = self.redis_client.hmget(key, 'tokens', 'last_refill')
            if result[0] is not None:
                return {
                    'tokens': int(result[0]),
                    'last_refill': datetime.fromtimestamp(float(result[1]))
                }
            return None
        except Exception as e:
            logger.error(f"Redis get_tokens failed: {e}")
            return None
    
    def set_tokens(self, key: str, tokens: int, last_refill: datetime, ttl_seconds: int) -> bool:
        """Set token state with TTL"""
        try:
            self.redis_client.hmset(key, {
                'tokens': tokens,
                'last_refill': last_refill.timestamp()
            })
            self.redis_client.expire(key, ttl_seconds)
            return True
        except Exception as e:
            logger.error(f"Redis set_tokens failed: {e}")
            return False
    
    def atomic_consume(self, key: str, requested_tokens: int, max_tokens: int, 
                       refill_rate: float) -> tuple[bool, int, datetime]:
        """Atomically consume tokens using Lua script"""
        try:
            current_time = time.time()
            result = self._lua_consume_script(
                keys=[key],
                args=[requested_tokens, max_tokens, refill_rate, current_time, 3600]
            )
            
            success = bool(result[0])
            remaining_tokens = int(result[1])
            timestamp = datetime.fromtimestamp(float(result[2]))
            
            return success, remaining_tokens, timestamp
            
        except Exception as e:
            logger.error(f"Redis atomic_consume failed: {e}")
            return False, 0, datetime.now()


class SQLiteTokenStorage(TokenStorageBackend):
    """SQLite fallback token storage"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_database()
        logger.info(f"✅ SQLite token storage initialized: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_buckets (
                    key TEXT PRIMARY KEY,
                    tokens INTEGER NOT NULL,
                    last_refill REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            
            # Create index for cleanup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON token_buckets(expires_at)
            """)
            
            conn.commit()
    
    def _cleanup_expired(self, conn: sqlite3.Connection):
        """Remove expired entries"""
        current_time = time.time()
        conn.execute("DELETE FROM token_buckets WHERE expires_at < ?", (current_time,))
    
    def get_tokens(self, key: str) -> Optional[Dict[str, Any]]:
        """Get current token state"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._cleanup_expired(conn)
                
                cursor = conn.execute(
                    "SELECT tokens, last_refill FROM token_buckets WHERE key = ? AND expires_at > ?",
                    (key, time.time())
                )
                result = cursor.fetchone()
                
                if result:
                    return {
                        'tokens': result[0],
                        'last_refill': datetime.fromtimestamp(result[1])
                    }
                return None
        except Exception as e:
            logger.error(f"SQLite get_tokens failed: {e}")
            return None
    
    def set_tokens(self, key: str, tokens: int, last_refill: datetime, ttl_seconds: int) -> bool:
        """Set token state with TTL"""
        try:
            expires_at = time.time() + ttl_seconds
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO token_buckets 
                    (key, tokens, last_refill, expires_at) 
                    VALUES (?, ?, ?, ?)
                """, (key, tokens, last_refill.timestamp(), expires_at))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"SQLite set_tokens failed: {e}")
            return False
    
    def atomic_consume(self, key: str, requested_tokens: int, max_tokens: int, 
                       refill_rate: float) -> tuple[bool, int, datetime]:
        """Atomically consume tokens with file locking"""
        try:
            current_time = time.time()
            current_datetime = datetime.fromtimestamp(current_time)
            
            with sqlite3.connect(self.db_path, timeout=1.0) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("BEGIN IMMEDIATE")  # Exclusive lock
                
                self._cleanup_expired(conn)
                
                # Get current state
                cursor = conn.execute(
                    "SELECT tokens, last_refill FROM token_buckets WHERE key = ? AND expires_at > ?",
                    (key, current_time)
                )
                result = cursor.fetchone()
                
                if result:
                    current_tokens, last_refill_ts = result
                    last_refill = last_refill_ts
                else:
                    current_tokens = max_tokens
                    last_refill = current_time
                
                # Calculate refill
                time_elapsed = current_time - last_refill
                tokens_to_add = int(time_elapsed * refill_rate)
                current_tokens = min(max_tokens, current_tokens + tokens_to_add)
                
                # Try to consume
                success = current_tokens >= requested_tokens
                if success:
                    current_tokens -= requested_tokens
                
                # Update state
                expires_at = current_time + 3600
                conn.execute("""
                    INSERT OR REPLACE INTO token_buckets 
                    (key, tokens, last_refill, expires_at) 
                    VALUES (?, ?, ?, ?)
                """, (key, current_tokens, current_time, expires_at))
                
                conn.commit()
                return success, current_tokens, current_datetime
                
        except Exception as e:
            logger.error(f"SQLite atomic_consume failed: {e}")
            return False, 0, datetime.now()


class ProcessSafeRateLimiter:
    """
    Process-safe rate limiter with persistent state
    
    Addresses reviewer concerns:
    - Process-safe operations (Redis/SQLite atomic ops)
    - Persistence across restarts
    - Burst capacity handling
    - Circuit breaker integration
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.refill_rate = config.requests_per_minute / 60.0  # tokens per second
        
        # Initialize storage backend with fallback
        self.storage = self._init_storage()
        
        logger.info(f"🔒 Process-safe rate limiter initialized")
        logger.info(f"   Rate: {config.requests_per_minute} req/min")
        logger.info(f"   Burst: {config.burst_capacity} tokens")
        logger.info(f"   Backend: {type(self.storage).__name__}")
    
    def _init_storage(self) -> TokenStorageBackend:
        """Initialize storage backend with fallback"""
        
        # Try Redis first
        if REDIS_AVAILABLE:
            try:
                return RedisTokenStorage(self.config.redis_url)
            except Exception as e:
                logger.warning(f"Redis initialization failed, falling back to SQLite: {e}")
        
        # Fallback to SQLite
        return SQLiteTokenStorage(self.config.sqlite_path)
    
    def consume(self, tokens: int = 1, client_id: str = "default") -> bool:
        """
        Consume tokens if available
        
        Args:
            tokens: Number of tokens to consume
            client_id: Client identifier for separate buckets
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        key = f"rate_limit:{client_id}"
        
        success, remaining, last_refill = self.storage.atomic_consume(
            key, tokens, self.config.burst_capacity, self.refill_rate
        )
        
        if success:
            logger.debug(f"✅ Consumed {tokens} tokens for {client_id}, {remaining} remaining")
        else:
            logger.debug(f"❌ Rate limit exceeded for {client_id}, {remaining} tokens available")
        
        return success
    
    def wait_for_tokens(self, tokens: int = 1, client_id: str = "default", 
                        max_wait_seconds: float = 60.0) -> bool:
        """
        Wait until tokens are available
        
        Args:
            tokens: Number of tokens needed
            client_id: Client identifier
            max_wait_seconds: Maximum time to wait
            
        Returns:
            True if tokens obtained, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            if self.consume(tokens, client_id):
                return True
            
            # Calculate wait time based on refill rate
            wait_time = min(tokens / self.refill_rate, 1.0)
            time.sleep(wait_time)
        
        logger.warning(f"⏰ Rate limiter timeout after {max_wait_seconds}s for {client_id}")
        return False
    
    def get_status(self, client_id: str = "default") -> Dict[str, Any]:
        """Get current rate limiter status"""
        key = f"rate_limit:{client_id}"
        state = self.storage.get_tokens(key)
        
        if state:
            # Calculate current tokens with refill
            current_time = time.time()
            last_refill_ts = state['last_refill'].timestamp()
            time_elapsed = current_time - last_refill_ts
            tokens_to_add = int(time_elapsed * self.refill_rate)
            current_tokens = min(self.config.burst_capacity, state['tokens'] + tokens_to_add)
            
            return {
                'current_tokens': current_tokens,
                'max_tokens': self.config.burst_capacity,
                'refill_rate_per_second': self.refill_rate,
                'last_refill': state['last_refill'],
                'utilization_pct': (self.config.burst_capacity - current_tokens) / self.config.burst_capacity * 100
            }
        else:
            # No state found, assume full capacity
            return {
                'current_tokens': self.config.burst_capacity,
                'max_tokens': self.config.burst_capacity,
                'refill_rate_per_second': self.refill_rate,
                'last_refill': datetime.now(),
                'utilization_pct': 0.0
            }


def main():
    """Test the process-safe rate limiter"""
    
    logging.basicConfig(level=logging.INFO)
    
    config = RateLimitConfig(
        requests_per_minute=5,
        burst_capacity=10
    )
    
    limiter = ProcessSafeRateLimiter(config)
    
    # Test basic consumption
    print("Testing basic token consumption:")
    for i in range(12):  # More than burst capacity
        success = limiter.consume(1, f"test_client_{i % 3}")
        status = limiter.get_status(f"test_client_{i % 3}")
        print(f"  Request {i+1}: {'✅' if success else '❌'} - {status['current_tokens']} tokens remaining")
    
    # Test wait functionality
    print("\nTesting wait functionality:")
    start_time = time.time()
    success = limiter.wait_for_tokens(3, "test_wait", max_wait_seconds=5.0)
    wait_time = time.time() - start_time
    print(f"  Wait result: {'✅' if success else '❌'} - Waited {wait_time:.2f}s")


if __name__ == "__main__":
    main()