# 📋 **PAPER TRADING SYSTEM - STARTUP RECAP**

## **🎯 CURRENT STATUS (End of Day - July 31, 2025)**

### **✅ COMPLETED TODAY:**
- **Phase 2B Training**: 251,200 steps completed successfully
- **Model**: 201K stable checkpoint ready (`models/200k_dual_ticker_training_0/`)
- **IBKR Connection**: Tested and working on port 7497 (paper trading mode)
- **Infrastructure**: All Docker services operational
- **Code Fixes**: Import errors resolved in execution components

---

## **🚀 TOMORROW'S STARTUP SEQUENCE**

### **STEP 1: INFRASTRUCTURE STARTUP (5 minutes)**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate

# Start Docker services
docker-compose up -d redis timescaledb pushgateway

# Verify services
docker ps
```

### **STEP 2: START IBKR WORKSTATION (2 minutes)**
1. Launch **IBKR Workstation**
2. **Login** to paper trading account
3. Ensure **API is enabled** (File → Global Configuration → API → Settings)
4. **Test connection**:
```bash
python test_ibkr_connection.py
```

### **STEP 3: START DATA PIPELINE (10 minutes)**
```bash
# 1. Start Polygon WebSocket (market data)
python scripts/launch_phase2b_services.py --service polygon

# 2. Start Redis → TimescaleDB ingest
python scripts/launch_phase2b_services.py --service ingest

# 3. Verify data flow
python -c "
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
print('Redis streams:', r.xinfo_stream('polygon:ticks')['length'])
"
```

### **STEP 4: START INFERENCE API (5 minutes)**
**Option A - Full Model (if working):**
```bash
python inference_api.py &
```

**Option B - Mock API (if model issues):**
```bash
python simple_inference_api.py &
```

**Test API:**
```bash
curl http://127.0.0.1:8000/health
```

### **STEP 5: START PAPER TRADING (2 minutes)**
```bash
# Extended hours for testing (9 AM - 11:59 PM)
TRADING_START="09:00:00" TRADING_END="23:59:00" python services/paper_loop.py \
    --symbols NVDA MSFT \
    --model-endpoint http://127.0.0.1:8000/inference \
    --portfolio-cap 100_000 \
    --risk-daily-max-loss 1_500 \
    --risk-intraday-dd 2.0 \
    --log-file logs/paper_$(date +%Y%m%d).jsonl &
```

### **STEP 6: MONITORING (3 minutes)**
```bash
# Check processes
ps aux | grep -E "paper_loop|inference|polygon"

# Monitor logs
tail -f logs/paper_$(date +%Y%m%d).jsonl

# Check Grafana dashboard (if configured)
# http://localhost:3000
```

---

## **📁 KEY FILES & LOCATIONS**

### **🤖 Models:**
- **Primary**: `models/200k_dual_ticker_training_0/` (201K steps, stable)
- **Backup**: `models/stable_baselines3_models/` (previous checkpoints)

### **🔧 Scripts:**
- **Connection Test**: `test_ibkr_connection.py`
- **Paper Trading**: `services/paper_loop.py`
- **Service Launcher**: `scripts/launch_phase2b_services.py`
- **Mock API**: `simple_inference_api.py` (lightweight fallback)

### **📊 Logs:**
- **Paper Trading**: `logs/paper_YYYYMMDD.jsonl`
- **Training**: `logs/200k_dual_ticker_training_0/`
- **System**: `logs/timescale/` and `logs/replica/`

### **⚙️ Config:**
- **Main**: `config/config.yaml`
- **Phase 2B**: `config/phase2b_config.yaml`
- **Docker**: `docker-compose.yml`

---

## **🚨 KNOWN ISSUES & SOLUTIONS**

### **Issue 1: Inference API Hanging**
- **Problem**: TensorFlow model loading causes API to freeze
- **Solution**: Use `simple_inference_api.py` (mock version)
- **Status**: Mock API tested and working

### **Issue 2: Trading Hours**
- **Problem**: Default hours (9:35 AM - 3:55 PM) too restrictive for testing
- **Solution**: Use environment variables to extend hours
- **Command**: `TRADING_START="09:00:00" TRADING_END="23:59:00"`

### **Issue 3: Model Path References**
- **Problem**: Some scripts reference old model paths
- **Solution**: Updated to point to `200k_dual_ticker_training_0/`
- **Status**: Fixed in key files

---

## **🎯 SUCCESS CRITERIA**

### **✅ System Ready When:**
1. **IBKR Connection**: `test_ibkr_connection.py` shows "CONNECTION SUCCESSFUL"
2. **Data Flow**: Redis stream `polygon:ticks` receiving messages
3. **API Response**: `curl http://127.0.0.1:8000/health` returns 200
4. **Paper Trading**: Process running and logging to `logs/paper_YYYYMMDD.jsonl`
5. **No Errors**: All processes stable for 5+ minutes

### **📈 Expected Behavior:**
- **Market Data**: Real-time ticks from Polygon.io → Redis
- **Inference**: API returning BUY/SELL/HOLD decisions
- **Risk Management**: $1,500 daily loss limit enforced
- **Execution**: Paper trades logged (simulation mode)
- **Monitoring**: Metrics flowing to Pushgateway

---

## **⏰ ESTIMATED STARTUP TIME: 25-30 MINUTES**

**Tomorrow morning, follow this sequence to resume paper trading from exactly where we left off today!** 🚀