# 🚀 **IMMEDIATE IBKR WSL SETUP - NEXT STEPS**

**Critical setup required for IBKR connection from WSL**

---

## 🎯 **CURRENT STATUS**

### **✅ WHAT'S WORKING**
- **Port connectivity**: `172.24.32.1:7497` is OPEN and accessible
- **Environment config**: `.env` file has correct `IBKR_HOST_IP=172.24.32.1`
- **Code implementation**: IBGatewayClient is complete and ready
- **WSL networking**: Can reach Windows host on correct IP

### **❌ WHAT'S BLOCKING**
- **Connection reset by peer**: IBKR is actively rejecting the connection
- **API configuration issue**: Critical settings missing in IBKR Workstation
- **Debug shows**: `[Errno 104] Connection reset by peer` - IBKR rejects immediately

---

## 🔧 **IMMEDIATE ACTION REQUIRED**

### **Step 1: Fix IBKR API Configuration (CRITICAL)**
1. **Open IBKR Workstation**
2. **Go to**: File → Global Configuration → API → Settings
3. **VERIFY THESE EXACT SETTINGS:**
   - ✅ **Enable ActiveX and Socket Clients**: CHECKED
   - ✅ **Socket port**: 7497
   - ✅ **Master API client ID**: 0 (or leave BLANK)
   - ✅ **Read-Only API**: UNCHECKED (must allow read/write)
   - ✅ **Trusted IPs**: 172.24.46.63 (your WSL IP)
   - ✅ **Allow connections from localhost**: CHECKED
4. **SECURITY SETTINGS:**
   - ✅ **Bypass Order Precautions for API orders**: CHECKED
5. **Click OK and RESTART TWS completely**
6. **Verify**: API status shows "Listening on port 7497"

### **Step 2: Test Connection**
```bash
# Activate environment
source venv/bin/activate

# Use debug script for detailed diagnostics
python debug_ibkr_connection.py

# Expected result after fixing API settings:
# ✅ Connection established!
# ✅ Server version: 176 (or similar)
# ✅ Account summary received
# ✅ Market data working

# If still failing, check for "Connection reset by peer" error
```

### **Step 3: Validate Full Integration**
```bash
# Test the main connection script
python test_ibkr_connection.py

# Should show:
# ✅ Connected to IB Gateway (mode: live)  # Not simulation!
# ✅ Account info retrieved
# ✅ Market data test completed
```

---

## 🎯 **VERIFICATION CHECKLIST**

### **IBKR Workstation Settings**
- [ ] **Paper Trading mode active** (shows "Paper" in title)
- [ ] **API enabled**: File → Global Config → API → Settings
- [ ] **Socket port 7497** configured
- [ ] **WSL IP added**: `172.24.46.63` in Trusted IPs
- [ ] **TWS restarted** after changes

### **Connection Tests**
- [ ] **Port test passes**: `nc -zv 172.24.32.1 7497`
- [ ] **Direct connection works**: `python test_ibkr_direct.py`
- [ ] **Integration test passes**: `python test_ibkr_connection.py`
- [ ] **Shows "live" mode**, not "simulation"

---

## 🚨 **TROUBLESHOOTING**

### **If Connection Still Times Out**
```bash
# Check Windows Firewall
# Ensure port 7497 is allowed for WSL subnet 172.24.0.0/16

# Try different client ID
python src/brokers/ib_gateway.py --host 172.24.32.1 --port 7497 --client-id 2 --test connect

# Check TWS logs for connection attempts
# Look for rejected connections from 172.24.46.63
```

### **If Still in Simulation Mode**
The code switches to simulation mode if:
1. `ib_insync` not available (✅ we have it)
2. IB credentials not found (this is normal for API connections)
3. Connection fails (this is our current issue)

**Solution**: Fix the connection, and it will automatically use live mode.

---

## 🎯 **EXPECTED OUTCOME**

After adding the WSL IP to IBKR Trusted IPs, you should see:

```bash
$ python test_ibkr_connection.py

🔌 IBKR WORKSTATION CONNECTION TEST
==================================================
📡 Testing connection to: 172.24.32.1:7497
🎭 Mode: LIVE  # ← This should change from SIMULATION
🆔 Client ID: 1

🔄 Attempting connection...
✅ CONNECTION SUCCESSFUL!

📊 Testing account information...
✅ Account info retrieved:
   account_id: DU123456  # ← Real paper account ID
   net_liquidation: 1000000.0  # ← Real paper account values
   available_funds: 1000000.0
   buying_power: 4000000.0
   currency: USD
   timestamp: 2025-08-05T12:00:00.000000
   mode: live  # ← Live mode confirmed

📈 Testing market data (AAPL)...
✅ Market data test completed

🔌 Disconnecting...
✅ Disconnected successfully
```

---

## 🚀 **NEXT STEPS AFTER CONNECTION WORKS**

Once the connection is established:

1. **Run simple paper trading demo**:
   ```bash
   python simple_ibkr_paper_trading.py
   ```

2. **Test live trader with fallback logic**:
   ```bash
   export AI_ENDPOINT_URL=disabled
   python live_trader.py --duration 5
   ```

3. **Monitor via Grafana dashboard**:
   ```bash
   # Ensure monitoring is running
   ./scripts/daily_startup.sh
   # Access: http://localhost:3000
   ```

---

**🎯 The key blocker is adding `172.24.46.63` to IBKR Trusted IPs. Once that's done, everything should work immediately!**

---

*Priority: CRITICAL - Required for paper trading*  
*ETA: 5 minutes to configure, immediate testing*