# REVISED INFRASTRUCTURE-FIRST PLAN
*Incorporating Production-Grade Refinements*

## IMMEDIATE DECISIONS (2-Hour Time-Box)

### **Cloud Provider Matrix** ⏱️ 2 Hours Research
| Provider | GPU Spot Pricing | Compliance | Org Credits | Recommendation |
|----------|------------------|------------|-------------|----------------|
| **AWS** | A10G: $0.75/hr spot | SOC2, HIPAA | $X available | ✅ **Primary choice** if credits exist |
| **GCP** | L4: $0.60/hr spot | ISO 27001 | $X available | ✅ **Alternative** for lower GPU cost |
| **Azure** | NC6s v3: $0.90/hr spot | FedRAMP | $X available | 🔄 **Backup** if enterprise discount |

**Decision Deadline**: Tomorrow EOD
**GPU Capacity Action**: Pre-book spot/reserved instances immediately after cloud choice

---

## WEEK 1: FOUNDATION + SKELETON CI/CD

### **Task 1.1: Enhanced Docker Containerization**
**Owner**: DevOps Lead  
**Duration**: 3-4 days

**Specific Deliverables**:
```dockerfile
# infrastructure/Dockerfile.institutional
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04  # Pin CUDA version
# SB3 + torch wheel cache layer (prevent repeated downloads)
# GPU driver compatibility validation
```

**Success Criteria**: ✅ **Week 1 Done Check**
- `docker compose up` reproduces **10-step local train on CPU**
- `pytest -q tests/smoke_gpu.py` passes in container
- Mixed driver version issues resolved

### **Task 1.2: Skeleton CI/CD Pipeline** 
**Owner**: DevOps Lead  
**Duration**: 1-2 days parallel

**Week 1 Scope** (Minimal but functional):
```yaml
# .github/workflows/institutional.yml
- Lint check (ruff, black)
- Docker build test
- Green checkmark on all PRs
```

**Full test matrix deferred to Week 3**
**Rationale**: Less merge friction for ongoing PRs

### **Task 1.3: Security & Secrets Framework**
**Owner**: DevOps Lead + You (approval)  
**Duration**: 1-2 days

**Deliverables**:
- AWS Secrets Manager / GCP Secret Manager integration
- `.env.template` file (no actual secrets)
- **CI Check**: Forbid hard-coded keys in Docker images
- Secret rotation procedures

**Rationale**: Auditors love to find keys in containers

---

## WEEK 2: DATA PIPELINE + METRICS FOUNDATION

### **Task 2.1: TimescaleDB + Data Model**
**Owner**: Data Engineer  
**Duration**: 5-7 days

**Enhanced Data Model** (ER Diagram):
```sql
-- Episodes table
CREATE TABLE episodes (
    run_id UUID,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    ep_reward DECIMAL,
    max_dd DECIMAL,
    final_pnl DECIMAL,
    -- ... other episode metrics
);

-- Raw ticks (S3 Parquet, partitioned by day/symbol)
-- Path: s3://bucket/ticks_raw/year=2025/month=07/day=23/symbol=NVDA/

-- Metrics hypertable (TimescaleDB)
CREATE TABLE metrics_minute (
    timestamp TIMESTAMPTZ,
    metric_name TEXT,
    value DECIMAL,
    run_id UUID,
    -- Timescale hypertable partitioning
);
```

**Success Criteria**: ✅ **Week 2 Done Check**
- TimescaleDB receives **>5 metrics/second** from mock run
- S3 cold storage policy active (>90 days)
- Grafana connects and shows basic charts

**Rationale**: Prevents later churn when dashboards ask for fields that never landed

### **Task 2.2: Metrics Ingestion Pipeline**
**Owner**: Data Engineer  
**Duration**: 3-4 days parallel

**Deliverables**:
- Real-time metrics streaming from training runs
- Batch upload for historical data
- Data validation and quality checks
- Partition management (auto-cleanup old data)

---

## WEEK 3: FULL CI/CD + DEPLOYMENT AUTOMATION

### **Task 3.1: Complete CI/CD Pipeline**
**Owner**: DevOps Lead  
**Duration**: 3-4 days

**Full Pipeline Scope**:
```yaml
# Complete .github/workflows/institutional.yml
stages:
  - lint: ruff, black, mypy
  - test: pytest full suite
  - build: Docker multi-arch
  - deploy: Auto-push to registry
  - qa: Spin up QA stack
```

**Success Criteria**: ✅ **Week 3 Done Check**
- **"Merge → deploy"** auto-pushes latest Docker tag to registry
- QA stack spins up automatically
- Rollback mechanism tested

### **Task 3.2: Blue/Green Deployment Test**
**Owner**: DevOps Lead  
**Duration**: 0.5 days (reserved)

**Deliverables**:
- **Blue/green flip test** (traffic switching)
- **Terraform destroy/apply test** (infrastructure recreation)
- Rollback procedures validated

**Rationale**: Proves infra is *re-creatable*, not just "worked on my laptop"

---

## WEEK 4: LOAD TESTING + PHASE 2A PREP

### **Task 4.1: Infrastructure Load Testing**
**Owner**: DevOps Lead + Data Engineer  
**Duration**: 2-3 days

**Success Criteria**: ✅ **Week 4 Done Check**
- Load test hits **2× production TPS** for 60 minutes
- **<5% dropped samples** under load
- GPU memory and compute utilization profiled
- Database performance characterized under load

### **Task 4.2: Phase 2A Design Work** 
**Owner**: Claude (AI Lead Quant)  
**Duration**: 2-3 days parallel

**Deliverables**:
- Transaction cost engine mathematical specification
- Integration points with TimescaleDB metrics
- Configuration schema for cost parameters
- Performance impact analysis

---

## ENHANCED SUCCESS CRITERIA PER WEEK

### ✅ **Week 1 Done Checklist**
- [ ] Cloud provider selected and GPU capacity reserved
- [ ] `docker compose up` reproduces 10-step local train on CPU
- [ ] `pytest -q tests/smoke_gpu.py` passes in container  
- [ ] Skeleton CI/CD shows green checkmarks on PRs
- [ ] Secrets management framework deployed (no hardcoded keys)

### ✅ **Week 2 Done Checklist**  
- [ ] TimescaleDB receives >5 metrics/second from mock run
- [ ] S3 cold storage policy active and tested
- [ ] Grafana dashboards show real-time training metrics
- [ ] Data model supports all planned Phase 2A metrics
- [ ] Partition management and cleanup working

### ✅ **Week 3 Done Checklist**
- [ ] "Merge → deploy" pipeline working end-to-end
- [ ] Blue/green deployment flip tested successfully
- [ ] Terraform destroy/apply recreates full stack
- [ ] All team members can deploy from their branches
- [ ] Monitoring and alerting active

### ✅ **Week 4 Done Checklist**
- [ ] Load test sustains 2× production TPS for 60 minutes
- [ ] <5% sample drop rate under maximum load
- [ ] GPU utilization profiled and optimized
- [ ] Phase 2A transaction cost engine designed
- [ ] Team ready for Phase 2A implementation

---

## RISK MITIGATION

### **Early Warning System**
- **Week slips**: Call risk early if any week's checklist <80% complete
- **GPU shortage**: Pre-booked capacity prevents "0 GPUs available" 
- **Integration issues**: Docker smoke tests catch driver mismatches early
- **Data model gaps**: ER diagram prevents dashboard field requests later

### **Contingency Plans**
- **Cloud provider fallback**: Secondary choice pre-researched
- **GPU capacity**: Mix of spot + reserved instances
- **Timeline buffer**: Week 4 has flex time if Weeks 1-3 slip
- **Rollback capability**: Blue/green + terraform tested before production

---

## IMMEDIATE ACTIONS (Next 24 Hours)

### **For You**:
1. ⏱️ **2-hour time-box**: Complete cloud provider matrix research
2. 📞 **GPU capacity**: Reserve spot/reserved instances immediately after cloud choice
3. 👥 **Team assignments**: Confirm DevOps Lead and Data Engineer availability
4. 💰 **Budget approval**: Infrastructure costs + GPU reservation costs

### **For Team Members**:
1. 🐳 **DevOps Lead**: Begin CUDA base image research and Docker setup
2. 🗄️ **Data Engineer**: Start TimescaleDB architecture research
3. 🔐 **Security review**: Secrets management requirements analysis

### **For Claude**:
1. 📊 **Data model design**: Create detailed ER diagram for metrics database
2. 🔧 **Docker specifications**: Detailed requirements for containerization
3. 📈 **50K analysis**: Continue training results deep dive

---

## PRODUCTION-GRADE REFINEMENTS INTEGRATED

### **Your Feedback Incorporated**:
- ✅ **Cloud decision detail**: 2-hour time-boxed matrix with GPU pricing
- ✅ **Container baseline**: CUDA base image pinning + wheel cache layers
- ✅ **Data model**: Detailed ER diagram with episodes/ticks/metrics tables
- ✅ **Security/secrets**: Explicit secrets management + CI checks
- ✅ **CI/CD timing**: Skeleton pipeline Week 1, full pipeline Week 3
- ✅ **Success criteria per week**: Concrete Done checklists with measurable targets
- ✅ **Roll-back story**: Blue/green flip + terraform destroy/apply tests
- ✅ **GPU capacity**: Pre-booking requirement to avoid shortages

### **Benefits of These Refinements**:
- **Prevents analysis-paralysis**: 2-hour cloud decision time-box
- **Avoids driver hell**: CUDA version pinning and smoke tests
- **Prevents data model churn**: ER diagram upfront
- **Satisfies auditors**: Proper secrets management
- **Reduces merge friction**: Green CI checks from day one
- **Enables early risk calls**: Weekly measurable success criteria
- **Proves reproducibility**: Infrastructure recreation tests
- **Prevents capacity surprises**: Pre-booked GPU instances

**Ready to proceed with this enhanced Infrastructure-First approach?**