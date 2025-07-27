# 🔐 **DAY 2 CREDENTIALS SETUP GUIDE**
**Required Secrets for Data Infrastructure**

---

## 🚨 **IMMEDIATE ACTION REQUIRED**

**Before Day 2 standup**, populate these credentials in GitHub repository settings to unblock development:

### **GitHub Repository Settings → Secrets and Variables → Actions**

#### **Required Secrets**
```bash
# Alpha Vantage API (Primary Data Feed)
ALPHA_VANTAGE_KEY=your_alpha_vantage_api_key_here

# Interactive Brokers Paper Trading
IB_USERNAME=your_ib_paper_username
IB_PASSWORD=your_ib_paper_password
IB_TRADING_MODE=paper

# Database (if using external TimescaleDB)
DB_PASSWORD=your_secure_db_password

# Optional: Backup Data Feeds
YAHOO_FINANCE_API_KEY=optional_for_premium_yahoo
```

---

## 📋 **CREDENTIAL ACQUISITION CHECKLIST**

### **✅ Alpha Vantage Setup** (Priority 1)
- [ ] **Sign up**: https://www.alphavantage.co/support/#api-key
- [ ] **Plan**: Premium Intraday (required for 1-minute bars)
- [ ] **Rate Limits**: 75 requests/minute, 500 requests/day (premium)
- [ ] **Test Key**: Verify with sample NVDA request
- [ ] **Add to GitHub**: Repository Settings → Secrets → `ALPHA_VANTAGE_KEY`

**Sample Test Command**:
```bash
curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&apikey=YOUR_KEY"
```

### **✅ Interactive Brokers Setup** (Priority 2)
- [ ] **Paper Account**: https://www.interactivebrokers.com/en/trading/free-trial.php
- [ ] **TWS Gateway**: Download and install
- [ ] **API Permissions**: Enable in account settings
- [ ] **Test Connection**: Verify localhost:7497 connectivity
- [ ] **Add to GitHub**: Repository Settings → Secrets → `IB_USERNAME`, `IB_PASSWORD`

**Connection Test**:
```python
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Should connect without error
```

### **✅ Database Credentials** (Priority 3)
- [ ] **Local Development**: Use default `testpass`
- [ ] **Production**: Generate secure password
- [ ] **Environment Variables**: Set `DB_PASSWORD` in deployment
- [ ] **Add to GitHub**: Repository Settings → Secrets → `DB_PASSWORD`

---

## 🔧 **LOCAL DEVELOPMENT SETUP**

### **Environment Variables File**
Create `.env` file in project root:
```bash
# Data Feeds
ALPHA_VANTAGE_KEY=your_key_here
YAHOO_FINANCE_FALLBACK=true

# Interactive Brokers
IB_HOST=localhost
IB_PORT=7497
IB_USERNAME=your_username
IB_PASSWORD=your_password
IB_TRADING_MODE=paper

# Database
DB_HOST=localhost
DB_PORT=5432
DB_PASSWORD=testpass
DOCKER_ENV=false

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
```

### **Docker Environment**
For containerized development:
```bash
# docker-compose.override.yml
version: '3.8'
services:
  app:
    environment:
      - ALPHA_VANTAGE_KEY=${ALPHA_VANTAGE_KEY}
      - IB_USERNAME=${IB_USERNAME}
      - IB_PASSWORD=${IB_PASSWORD}
      - DB_HOST=timescaledb
      - DOCKER_ENV=true
```

---

## 🧪 **CREDENTIAL VALIDATION TESTS**

### **Alpha Vantage Validation**
```bash
# Test script: scripts/test_alpha_vantage.py
python scripts/test_alpha_vantage.py
# Expected: ✅ NVDA data retrieved successfully
#          ✅ MSFT data retrieved successfully  
#          ✅ Rate limiting working correctly
```

### **IB Gateway Validation**
```bash
# Test script: scripts/test_ib_connection.py
python scripts/test_ib_connection.py
# Expected: ✅ Connected to IB Gateway
#          ✅ Paper trading account verified
#          ✅ Market data permissions confirmed
```

### **Database Connection**
```bash
# Test database connectivity
python -c "
from src.oms.position_tracker import PositionTracker
tracker = PositionTracker()
positions = tracker.get_all_positions()
print(f'✅ Database connected, {len(positions)} positions found')
"
```

---

## ⚠️ **SECURITY BEST PRACTICES**

### **Never Commit Secrets**
```bash
# Add to .gitignore (already included)
.env
*.key
credentials/
secrets/
```

### **Use Environment-Specific Configs**
```yaml
# config/prod.yaml
feeds:
  alpha_vantage:
    api_key: ${ALPHA_VANTAGE_KEY}  # From environment
    rate_limit: 75  # requests/minute
  
ib_gateway:
  username: ${IB_USERNAME}  # From environment
  password: ${IB_PASSWORD}  # From environment
  mode: ${IB_TRADING_MODE:-paper}
```

### **Rotate Keys Regularly**
- **Alpha Vantage**: Monthly rotation recommended
- **IB Credentials**: Change after each development cycle
- **Database Passwords**: Use strong, unique passwords

---

## 🚨 **TROUBLESHOOTING**

### **Alpha Vantage Issues**
```bash
# Rate limit exceeded
Error: "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute"
Solution: Upgrade to premium plan or implement request throttling

# Invalid API key
Error: "Invalid API call. Please retry or visit the documentation"
Solution: Verify API key in repository secrets, check for typos
```

### **IB Gateway Issues**
```bash
# Connection refused
Error: "ConnectionRefusedError: [Errno 111] Connection refused"
Solution: Start TWS Gateway, check port 7497, verify firewall settings

# Authentication failed
Error: "Authentication failed"
Solution: Verify paper trading account credentials, check account status
```

### **Database Issues**
```bash
# Connection timeout
Error: "psycopg2.OperationalError: could not connect to server"
Solution: Start TimescaleDB service, check port 5432, verify password
```

---

## 📞 **ESCALATION CONTACTS**

### **Credential Issues**
- **Alpha Vantage Support**: support@alphavantage.co
- **IB Support**: https://www.interactivebrokers.com/en/support/
- **Internal DevOps**: [Your team's DevOps contact]

### **Emergency Fallbacks**
- **Data Feed**: Switch to Yahoo Finance (no API key required)
- **Trading**: Use simulation mode (no IB credentials needed)
- **Database**: Use SQLite for local development

---

## ✅ **COMPLETION CHECKLIST**

Before Day 2 development begins:
- [ ] **Alpha Vantage API key**: Added to GitHub secrets and tested
- [ ] **IB Credentials**: Paper account created and credentials stored
- [ ] **Database Password**: Secure password generated and stored
- [ ] **Local .env**: Development environment configured
- [ ] **Validation Tests**: All credential tests passing
- [ ] **Team Notification**: Developers informed credentials are ready

---

**🎯 Success Criteria**: All developers can run `python scripts/validate_credentials.py` and see green checkmarks for all services.

---

*This setup must be completed before 09:00 Day 2 standup to prevent development delays.*