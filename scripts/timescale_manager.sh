#!/bin/bash
# TimescaleDB Replica Management Script
# Comprehensive management for TimescaleDB primary/replica setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.timescale.yml"
ENV_FILE=".env.timescale"

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"  
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_success "Requirements check passed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create directories
    mkdir -p data/timescale/{primary,replica,wal}
    mkdir -p logs/{timescale,replica}
    mkdir -p sql/{init,schema,backups,pgadmin}
    
    # Set permissions (only if directories are writable)
    if [ -w data/timescale ]; then
        chmod -R 755 data/timescale 2>/dev/null || true
    fi
    if [ -w logs ]; then
        chmod -R 755 logs 2>/dev/null || true
    fi
    
    # Copy environment template if .env doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f "$ENV_FILE" ]; then
            log_info "Copying environment template..."
            cp "$ENV_FILE" ".env"
            log_warning "Please edit .env file with your actual passwords!"
        fi
    fi
    
    log_success "Environment setup complete"
}

start_primary() {
    log_info "Starting TimescaleDB primary..."
    
    docker-compose -f "$COMPOSE_FILE" up -d primary
    
    log_info "Waiting for primary to be healthy..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose -f "$COMPOSE_FILE" ps primary | grep -q "healthy"; then
            log_success "Primary is healthy"
            return 0
        fi
        sleep 2
        ((timeout-=2))
    done
    
    log_error "Primary failed to become healthy within 60 seconds"
    docker-compose -f "$COMPOSE_FILE" logs primary
    exit 1
}

start_replica() {
    log_info "Starting TimescaleDB replica..."
    
    # Ensure primary is running
    if ! docker-compose -f "$COMPOSE_FILE" ps primary | grep -q "healthy"; then
        log_error "Primary must be healthy before starting replica"
        exit 1
    fi
    
    docker-compose -f "$COMPOSE_FILE" --profile replica up -d replica
    
    log_info "Waiting for replica to be healthy..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if docker-compose -f "$COMPOSE_FILE" ps replica | grep -q "healthy"; then
            log_success "Replica is healthy"
            return 0
        fi
        sleep 3
        ((timeout-=3))
    done
    
    log_error "Replica failed to become healthy within 120 seconds"
    docker-compose -f "$COMPOSE_FILE" logs replica
    exit 1
}

start_tools() {
    log_info "Starting management tools (pgAdmin, schema backup)..."
    
    docker-compose -f "$COMPOSE_FILE" --profile tools up -d pgadmin schema_backup
    
    log_success "Management tools started"
    log_info "pgAdmin available at: http://localhost:8080"
}

stop_services() {
    log_info "Stopping TimescaleDB services..."
    
    docker-compose -f "$COMPOSE_FILE" --profile replica --profile tools down
    
    log_success "Services stopped"
}

backup_schema() {
    log_info "Running schema backup..."
    
    if ! docker-compose -f "$COMPOSE_FILE" ps schema_backup | grep -q "running"; then
        log_error "Schema backup service is not running. Start tools first."
        exit 1
    fi
    
    docker exec timescale_schema_backup /scripts/backup_schema.sh
    
    # Copy to local schema directory
    docker cp timescale_schema_backup:/backups/schema_latest.sql ./sql/schema/
    
    log_success "Schema backup completed and copied to sql/schema/"
}

check_replication() {
    log_info "Checking replication status..."
    
    # Check primary status
    log_info "Primary replication status:"
    docker exec timescaledb_primary psql -U postgres -d trading_data -c "
        SELECT 
            client_addr,
            application_name,
            state,
            sync_state,
            replay_lag
        FROM pg_stat_replication;
    " || log_warning "Could not check primary replication status"
    
    # Check replica status
    log_info "Replica replication status:"
    docker exec timescaledb_replica psql -U postgres -d trading_data -c "
        SELECT 
            CASE 
                WHEN pg_is_in_recovery() 
                THEN 'Replica (in recovery)' 
                ELSE 'Primary (not in recovery)' 
            END as status;
    " || log_warning "Could not check replica status"
    
    # Check lag
    docker exec timescaledb_replica psql -U postgres -d trading_data -c "
        SELECT 
            EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as replay_lag_seconds;
    " || log_warning "Could not check replica lag"
}

test_connection() {
    local host="$1"
    local port="$2"
    local name="$3"
    
    log_info "Testing connection to $name ($host:$port)..."
    
    if docker exec timescaledb_primary pg_isready -h "$host" -p "$port" -U postgres; then
        log_success "$name connection successful"
        
        # Test database access
        docker exec timescaledb_primary psql -h "$host" -p "$port" -U postgres -d trading_data -c "SELECT version();" > /dev/null
        log_success "$name database access successful"
    else
        log_error "$name connection failed"
        return 1
    fi
}

test_connections() {
    log_info "Testing database connections..."
    
    test_connection "primary" "5432" "Primary"
    test_connection "replica" "5432" "Replica"
    
    log_success "All connection tests passed"
}

show_status() {
    log_info "TimescaleDB Service Status:"
    echo ""
    
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Container Health Status:"
    
    # Check each container
    containers=("timescaledb_primary" "timescaledb_replica" "timescale_pgadmin" "timescale_schema_backup")
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container"; then
            status=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$container" | awk '{print $2}')
            log_success "$container: $status"
        else
            log_warning "$container: Not running"
        fi
    done
    
    echo ""
    log_info "Access URLs:"
    echo "  📊 pgAdmin: http://localhost:8080"
    echo "  🔗 Primary DB: postgresql://postgres:password@localhost:5432/trading_data"
    echo "  🔗 Replica DB: postgresql://postgres:password@localhost:5433/trading_data"
}

show_logs() {
    local service="$1"
    
    if [ -z "$service" ]; then
        log_info "Available services: primary, replica, pgadmin, schema_backup"
        return 1
    fi
    
    docker-compose -f "$COMPOSE_FILE" logs -f "$service"
}

clean_data() {
    log_warning "This will DELETE ALL TimescaleDB data! Are you sure? (yes/NO)"
    read -r confirmation
    
    if [ "$confirmation" = "yes" ]; then
        log_info "Stopping services..."
        stop_services
        
        log_info "Removing data directories..."
        sudo rm -rf data/timescale/primary/*
        sudo rm -rf data/timescale/replica/*  
        sudo rm -rf data/timescale/wal/*
        sudo rm -rf logs/timescale/*
        sudo rm -rf logs/replica/*
        
        log_warning "All TimescaleDB data has been deleted!"
    else
        log_info "Operation cancelled"
    fi
}

promote_replica() {
    log_warning "This will promote replica to primary! Are you sure? (yes/NO)"
    read -r confirmation
    
    if [ "$confirmation" = "yes" ]; then
        log_info "Promoting replica to primary..."
        docker exec timescaledb_replica touch /tmp/promote_replica
        log_success "Replica promotion triggered"
        
        sleep 5
        check_replication
    else
        log_info "Operation cancelled"
    fi
}

restart_replica() {
    log_info "Restarting replica with slot cleanup (faster than full restart)..."
    
    # Stop replica
    log_info "Stopping replica container..."
    docker-compose -f "$COMPOSE_FILE" --profile replica stop replica
    
    # Drop replication slot on primary
    log_info "Dropping replication slot on primary..."
    docker exec timescaledb_primary psql -U postgres -d trading_data -c "
        SELECT pg_drop_replication_slot(slot_name) 
        FROM pg_replication_slots 
        WHERE slot_name LIKE 'replica_%';" || log_warning "No slots to drop"
    
    # Remove replica data
    log_info "Cleaning replica data..."
    docker volume rm timescale_replica_data 2>/dev/null || true
    
    # Restart replica
    log_info "Starting fresh replica..."
    start_replica
    
    log_success "Replica restart complete"
}

# Main script logic
case "$1" in
    "setup")
        check_requirements
        setup_environment
        log_success "Setup complete. Run './scripts/timescale_manager.sh start' to begin."
        ;;
    "start")
        check_requirements
        setup_environment
        start_primary
        ;;
    "start-replica")
        start_replica
        ;;
    "start-tools")
        start_tools
        ;;
    "start-all")
        check_requirements
        setup_environment
        start_primary
        start_replica
        start_tools
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_primary
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "backup")
        backup_schema
        ;;
    "test")
        test_connections
        ;;
    "replication")
        check_replication
        ;;
    "promote")
        promote_replica
        ;;
    "restart-replica")
        restart_replica
        ;;
    "clean")
        clean_data
        ;;
    *)
        echo "TimescaleDB Replica Manager"
        echo ""
        echo "Usage: $0 {command} [options]"
        echo ""
        echo "Commands:"
        echo "  setup          - Initialize directories and environment"
        echo "  start          - Start primary database only"
        echo "  start-replica  - Start replica database"
        echo "  start-tools    - Start pgAdmin and backup tools"
        echo "  start-all      - Start all services"
        echo "  stop           - Stop all services"
        echo "  restart        - Restart primary database"
        echo "  status         - Show service status"
        echo "  logs [service] - Show logs for service"
        echo "  backup         - Run schema backup"
        echo "  test           - Test database connections"
        echo "  replication    - Check replication status"
        echo "  promote        - Promote replica to primary"
        echo "  restart-replica - Drop slot & re-sync replica (faster than full restart)"
        echo "  clean          - Delete all data (DANGEROUS)"
        echo ""
        echo "Examples:"
        echo "  $0 setup                 # First time setup"
        echo "  $0 start-all            # Start everything"
        echo "  $0 logs primary         # Show primary logs"
        echo "  $0 backup               # Backup schema"
        exit 1
        ;;
esac