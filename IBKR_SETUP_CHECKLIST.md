# 🔧 **IBKR WORKSTATION SETUP CHECKLIST**

**Issue**: TCP connection succeeds but API handshake times out  
**WSL IP**: `172.24.46.63` (confirmed)  
**Windows Host**: `172.24.32.1:7497` (confirmed accessible)

---

## ✅ **STEP-BY-STEP VERIFICATION**

### **1. IBKR Workstation Configuration**

**In IBKR Workstation, go to:**
```
File → Global Configuration → API → Settings
```

**Check these settings:**
- [ ] ✅ **Enable ActiveX and Socket Clients** (MUST be checked)
- [ ] ✅ **Read-Only API**: UNCHECKED (we need trading permissions)  
- [ ] ✅ **Socket port**: `7497`
- [ ] ✅ **Master API client ID**: `0` (default)
- [ ] ✅ **Create API message log file**: Optional but helpful for debugging

### **2. Trusted IPs Configuration**

**In the same API Settings dialog:**
- [ ] ✅ **Trusted IPs**: Should contain `172.24.46.63`
- [ ] ✅ **Format**: One IP per line, no extra spaces
- [ ] ✅ **Also add**: `127.0.0.1` (for local testing)

**Example Trusted IPs list:**
```
127.0.0.1
172.24.46.63
```

### **3. Paper Trading Mode**

**Verify Paper Trading is active:**
- [ ] ✅ **Account selector**: Should show "Paper" in the account name
- [ ] ✅ **Window title**: Should contain "Paper Trading" 
- [ ] ✅ **Account balance**: Should show simulated funds (not real money)

### **4. Restart IBKR Workstation**

**After making changes:**
- [ ] ✅ **Click "OK"** to save API settings
- [ ] ✅ **Completely close** IBKR Workstation
- [ ] ✅ **Restart** IBKR Workstation
- [ ] ✅ **Login to Paper Trading** mode
- [ ] ✅ **Wait** for full startup (2-3 minutes)

---

## 🧪 **VERIFICATION TESTS**

### **Test 1: Manual Connection Check**

**In IBKR Workstation:**
1. Look for **API connection status** (usually bottom-right corner)
2. Should show something like: `API: Ready` or `API: Connected`
3. If you see `API: Disabled`, the settings didn't save properly

### **Test 2: Command Line Test**

**Run this after IBKR Workstation restart:**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate
python test_ibkr_debug.py
```

**Expected result:**
- Network connectivity: ✅ OK
- At least one client ID should connect successfully

### **Test 3: Check IBKR Logs**

**IBKR creates log files here (on Windows):**
```
C:\Users\[YourUsername]\Documents\IBLogs\
```

**Look for recent log files with API connection attempts**

---

## 🔍 **COMMON ISSUES & SOLUTIONS**

### **Issue 1: "API connection failed: TimeoutError()"**

**Likely causes:**
- Trusted IPs not configured correctly
- API not enabled properly
- IBKR Workstation not fully started

**Solutions:**
1. Double-check trusted IPs include `172.24.46.63`
2. Restart IBKR Workstation completely
3. Wait 2-3 minutes after startup before testing

### **Issue 2: "Connection refused"**

**Likely causes:**
- Wrong port (should be 7497 for paper trading)
- API not enabled
- Windows firewall blocking connection

**Solutions:**
1. Verify port 7497 in API settings
2. Check Windows firewall allows IBKR Workstation
3. Try disabling Windows firewall temporarily for testing

### **Issue 3: "Permission denied"**

**Likely causes:**
- Read-only API enabled
- Paper trading account restrictions

**Solutions:**
1. Uncheck "Read-Only API" in settings
2. Ensure account has paper trading permissions

---

## 📞 **NEXT STEPS**

**After verifying all settings:**

1. **Restart IBKR Workstation** with changes
2. **Wait 2-3 minutes** for full startup
3. **Run test**: `python test_ibkr_debug.py`
4. **If still failing**: Check Windows IBKR log files for errors
5. **Success**: We'll integrate real IBKR connection into the paper trading system

---

**🎯 Goal**: Get at least one successful connection with a client ID, then we can proceed with real paper trading integration!