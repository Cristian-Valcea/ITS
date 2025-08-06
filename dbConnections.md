# 🗄️ **DATABASE CONNECTIONS GUIDE**

**TimescaleDB Docker Setup for IntradayJules Trading System**

---

## 📋 **DATABASE OVERVIEW**

### **Database Type**: TimescaleDB (PostgreSQL + Time-Series Extension)
### **Deployment**: Docker Containers
### **Primary Database**: `trading_data`
### **Main Table**: `minute_bars` (market data 2022-2024)

---

## 🐳 **DOCKER SETUP**

### **Container Information**
- **Primary Container**: `timescaledb_primary`
- **Image**: `timescale/timescaledb:2.14.2-pg14`
- **Port**: `5432:5432` (host:container)
- **Network**: `timescale_replication_network`

### **Docker Compose Files**
- **Main Config**: `docker-compose.timescale.yml`
- **Simple Config**: `docker-compose.timescale-simple.yml`
- **Grafana Integration**: `docker-compose.grafana.yml`

---

## 🚀 **STARTING THE DATABASE**

### **Method 1: Start Existing Container**
```bash
# Check container status
docker ps -a | grep timescale

# Start primary database
docker start timescaledb_primary

# Verify it's running
docker ps | grep timescale

# Check if database is ready
docker exec timescaledb_primary pg_isready -U postgres -d trading_data
```

### **Method 2: Docker Compose (Full Setup)**
```bash
# Start primary database only
docker compose -f docker-compose.timescale.yml up -d primary

# Start with Redis cache
docker compose -f docker-compose.timescale.yml up -d primary redis_cache

# Start with all tools (pgAdmin, backup service)
docker compose -f docker-compose.timescale.yml --profile tools up -d
```

### **Method 3: Restart All Services**
```bash
# Stop all TimescaleDB services
docker compose -f docker-compose.timescale.yml down

# Start fresh
docker compose -f docker-compose.timescale.yml up -d primary redis_cache
```

---

## 🔐 **CONNECTION DETAILS**

### **Database Credentials**
- **Host**: `localhost` (or `timescaledb_primary` from within Docker network)
- **Port**: `5432`
- **Database**: `trading_data`
- **Username**: `postgres`
- **Password**: Retrieved from `SecretsHelper` (encrypted vault)

### **Connection String Format**
```
postgresql://postgres:PASSWORD@localhost:5432/trading_data
```

### **Python Connection Example**
```python
# Using SecretsHelper (recommended)
from secrets_helper import SecretsHelper
db_url = SecretsHelper.get_database_url()

# Direct connection
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="trading_data",
    user="postgres",
    password=SecretsHelper.get_timescaledb_password()
)
```

---

## 📊 **DATABASE SCHEMA**

### **Main Table: `minute_bars`**
```sql
-- Market data table structure
CREATE TABLE minute_bars (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume BIGINT NOT NULL,
    PRIMARY KEY (timestamp, symbol)
);

-- TimescaleDB hypertable
SELECT create_hypertable('minute_bars', 'timestamp');
```

### **Data Coverage**
- **Symbols**: NVDA, MSFT (primary trading pairs)
- **Time Range**: 2022-01-03 to 2024-12-31
- **Frequency**: 1-minute bars
- **Records**: ~500K+ per symbol

### **Sample Queries**
```sql
-- Check recent data
SELECT COUNT(*) FROM minute_bars 
WHERE symbol IN ('NVDA', 'MSFT') 
AND timestamp > NOW() - INTERVAL '7 days';

-- Get latest prices
SELECT symbol, timestamp, close 
FROM minute_bars 
WHERE symbol IN ('NVDA', 'MSFT')
ORDER BY timestamp DESC 
LIMIT 10;

-- Market hours data only (9:30-16:00 ET)
SELECT * FROM minute_bars 
WHERE EXTRACT(HOUR FROM timestamp AT TIME ZONE 'America/New_York') BETWEEN 9 AND 16
AND symbol = 'MSFT'
ORDER BY timestamp DESC 
LIMIT 100;
```

---

## 🛠️ **MANAGEMENT TOOLS**

### **pgAdmin (Web Interface)**
```bash
# Start pgAdmin
docker compose -f docker-compose.timescale.yml --profile tools up -d pgadmin

# Access at: http://localhost:8080
# Email: admin@trading.local
# Password: admin_secure_password
```

### **Direct Database Access**
```bash
# Connect via docker exec
docker exec -it timescaledb_primary psql -U postgres -d trading_data

# Run SQL commands
docker exec timescaledb_primary psql -U postgres -d trading_data -c "SELECT COUNT(*) FROM minute_bars;"
```

### **Backup & Restore**
```bash
# Create backup
docker exec timescaledb_primary pg_dump -U postgres trading_data > backup_$(date +%Y%m%d).sql

# Restore backup
docker exec -i timescaledb_primary psql -U postgres trading_data < backup_20250805.sql

# Schema backup service
docker compose -f docker-compose.timescale.yml --profile tools up -d schema_backup
docker exec timescale_schema_backup /scripts/backup_schema.sh
```

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues**

#### **1. Container Won't Start**
```bash
# Check logs
docker logs timescaledb_primary

# Remove and recreate
docker rm timescaledb_primary
docker compose -f docker-compose.timescale.yml up -d primary
```

#### **2. Connection Refused**
```bash
# Check if container is running
docker ps | grep timescale

# Check port mapping
docker port timescaledb_primary

# Wait for startup (can take 30-60 seconds)
docker exec timescaledb_primary pg_isready -U postgres -d trading_data
```

#### **3. Permission Issues**
```bash
# Check data directory permissions
docker exec timescaledb_primary ls -la /var/lib/postgresql/data

# Reset permissions
docker exec timescaledb_primary chown -R postgres:postgres /var/lib/postgresql/data
```

#### **4. Memory Issues**
```bash
# Check container resources
docker stats timescaledb_primary

# Increase Docker memory limit (8GB recommended)
# Restart Docker Desktop if needed
```

---

## 📈 **PERFORMANCE OPTIMIZATION**

### **Database Configuration**
- **WAL Level**: `logical` (for replication)
- **Max Connections**: 100
- **Shared Buffers**: 256MB
- **Work Memory**: 4MB
- **Maintenance Work Memory**: 64MB

### **TimescaleDB Specific**
- **Chunk Time Interval**: 1 day
- **Compression**: Enabled for data older than 7 days
- **Retention Policy**: 2+ years of data

### **Monitoring Queries**
```sql
-- Check chunk compression
SELECT * FROM chunk_compression_stats('minute_bars');

-- Database size
SELECT pg_size_pretty(pg_database_size('trading_data'));

-- Table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables WHERE tablename LIKE '%minute_bars%';
```

---

## 🚨 **PRODUCTION CHECKLIST**

### **Before Trading Hours**
- [ ] TimescaleDB container running (`docker ps | grep timescale`)
- [ ] Database accepting connections (`pg_isready`)
- [ ] Recent market data available (last 7 days)
- [ ] Redis cache container running (if used)
- [ ] Backup completed within 24 hours

### **Health Check Commands**
```bash
# Quick status check
docker ps | grep timescale && docker exec timescaledb_primary pg_isready -U postgres -d trading_data

# Data freshness check
docker exec timescaledb_primary psql -U postgres -d trading_data -c "
SELECT symbol, MAX(timestamp) as latest_data 
FROM minute_bars 
WHERE symbol IN ('NVDA', 'MSFT') 
GROUP BY symbol;"
```

---

## 📞 **EMERGENCY PROCEDURES**

### **Database Down During Trading**
1. **Immediate**: Switch to backup data source or cached data
2. **Restart container**: `docker restart timescaledb_primary`
3. **If restart fails**: Restore from backup or start new container
4. **Contact**: Senior Developer immediately

### **Data Corruption**
1. **Stop trading immediately**
2. **Restore from latest backup**
3. **Verify data integrity before resuming**
4. **Document incident for post-mortem**

---

**Environment**: Docker containers on WSL2 Linux  
**Last Updated**: August 5, 2025  
**Maintenance**: Weekly restarts recommended, monthly backups mandatory