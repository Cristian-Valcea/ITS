# DuckDB Concurrency Fix

## 🚨 **Problem**
DuckDB threw transaction errors during training-to-evaluation transitions:
```
write-write conflict: Cannot have two writers to the same database
```

**Root Cause**: Training phase kept DuckDB connections open in write mode while evaluation phase tried to open additional write connections to the same database file.

## 🛠️ **Solution Overview**

### **1. Connection Manager (`src/shared/duckdb_manager.py`)**
- **Thread-safe connection pooling** with proper locking
- **Read-only vs read-write mode** management
- **Automatic retry** on write conflicts with exponential backoff
- **Connection cleanup** between pipeline phases

### **2. Updated FeatureStore (`src/shared/feature_store.py`)**
- **Removed persistent connections** - no more `self.db`
- **Context-managed connections** for all operations
- **Read-only connections** for cache lookups
- **Write connections** only when needed for updates

### **3. Orchestrator Integration (`src/execution/orchestrator_agent.py`)**
- **Connection cleanup** after training completion
- **Write connection closure** before evaluation starts
- **Phase-aware connection management** in walk-forward validation

## 🔧 **Key Changes**

### **Connection Manager**
```python
# Thread-safe connection with automatic retry
with get_duckdb_connection(db_path, mode='rw', max_retries=3) as conn:
    conn.execute("INSERT INTO manifest ...")

# Read-only connection for cache lookups
with get_duckdb_connection(db_path, mode='r') as conn:
    result = conn.execute("SELECT * FROM manifest WHERE key = ?", [key])
```

### **Phase Management**
```python
# Training phase
with training_phase():
    trained_model = run_training(...)

# Evaluation phase (auto-closes write connections)
with evaluation_phase():
    results = run_evaluation(...)
```

### **Orchestrator Integration**
```python
# After training, before evaluation
self.logger.info("🔒 Closing DuckDB write connections before evaluation...")
close_write_duckdb_connections()

# Final cleanup after pipeline
self.logger.info("🧹 Cleaning up all DuckDB connections...")
close_all_duckdb_connections()
```

## 📊 **Before vs After**

### **Before (Problematic)**
```
Training Phase:
├── Opens DuckDB connection (write mode)
├── Keeps connection open during training
└── Training completes but connection still open

Evaluation Phase:
├── Tries to open DuckDB connection (write mode)
├── ❌ CONFLICT: Two writers to same database
└── Falls back to cache miss, recomputes features
```

### **After (Fixed)**
```
Training Phase:
├── Uses connection manager for DuckDB operations
├── Connections auto-closed after each operation
└── Training completes, all connections cleaned up

Transition:
├── 🔒 Explicitly close write connections
└── 🧹 Clean up any remaining connections

Evaluation Phase:
├── Uses read-only connections for cache lookups
├── Uses write connections only when needed
└── ✅ No conflicts, cache hits work correctly
```

## 🧪 **Testing**

Run the concurrency test:
```bash
python test_duckdb_concurrency_fix.py
```

This test simulates:
1. **Training phase** writing to DuckDB
2. **Connection cleanup** between phases
3. **Evaluation phase** reading from DuckDB
4. **FeatureStore operations** with proper connection management

## 🎯 **Benefits**

### **1. Eliminates Write-Write Conflicts**
- No more transaction errors during training-evaluation transitions
- Proper connection lifecycle management

### **2. Improves Cache Hit Rate**
- Cache lookups work correctly during evaluation
- No unnecessary feature recomputation

### **3. Better Resource Management**
- Connections are properly closed and cleaned up
- Reduced memory usage and file handle leaks

### **4. Thread Safety**
- Connection manager handles concurrent access safely
- Proper locking prevents race conditions

## 🔍 **Monitoring**

Check connection status:
```python
from src.shared.duckdb_manager import get_duckdb_connection_info
info = get_duckdb_connection_info()
print(f"Active connections: {info['active_connections']}")
```

## 🚀 **Usage Guidelines**

### **For Feature Operations**
```python
# Cache lookup (read-only)
with get_duckdb_connection(db_path, mode='r') as conn:
    result = conn.execute("SELECT path FROM manifest WHERE key = ?", [key])

# Cache update (write mode)
with get_duckdb_connection(db_path, mode='rw') as conn:
    conn.execute("INSERT INTO manifest (...) VALUES (...)")
```

### **For Pipeline Transitions**
```python
# Before starting evaluation
close_write_duckdb_connections()

# After completing pipeline
close_all_duckdb_connections()
```

### **For Phase Management**
```python
# Training with automatic cleanup
with training_phase():
    model = train_model(...)

# Evaluation with write connection cleanup
with evaluation_phase():
    results = evaluate_model(...)
```

## ✅ **Verification**

The fix ensures:
- ✅ **No write-write conflicts** during training-evaluation transitions
- ✅ **Proper cache hits** during evaluation phase
- ✅ **Clean connection lifecycle** management
- ✅ **Thread-safe operations** across all agents
- ✅ **Automatic retry** on transient conflicts
- ✅ **Resource cleanup** after pipeline completion

**Result**: Training and evaluation phases now transition smoothly without DuckDB transaction errors, and the feature cache works correctly throughout the entire pipeline. 🎉