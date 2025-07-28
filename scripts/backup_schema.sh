#!/bin/bash
# Schema backup script for CI integration
# Usage: docker exec timescale_schema_backup /scripts/backup_schema.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
PRIMARY_HOST="primary"
DATABASE="trading_data"
USER="postgres"

echo "🚀 Starting schema backup at $TIMESTAMP"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Wait for primary to be ready
echo "⏳ Waiting for primary database to be ready..."
until pg_isready -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE"; do
    echo "   Primary not ready, waiting..."
    sleep 2
done

echo "✅ Primary database is ready"

# Backup schema only (structure without data)
echo "📄 Backing up database schema..."
pg_dump -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE" \
    --schema-only \
    --no-owner \
    --no-privileges \
    --clean \
    --if-exists \
    --verbose \
    > "$BACKUP_DIR/schema_${TIMESTAMP}.sql"

# Create latest schema symlink
ln -sf "schema_${TIMESTAMP}.sql" "$BACKUP_DIR/schema_latest.sql"

echo "✅ Schema backup completed: schema_${TIMESTAMP}.sql"

# Backup TimescaleDB specific objects
echo "📊 Backing up TimescaleDB hypertables metadata..."
pg_dump -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE" \
    --data-only \
    --table="_timescaledb_catalog.*" \
    --inserts \
    > "$BACKUP_DIR/timescale_metadata_${TIMESTAMP}.sql" 2>/dev/null || true

echo "✅ TimescaleDB metadata backup completed"

# Create schema documentation
echo "📚 Generating schema documentation..."
psql -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE" -c "
SELECT 
    schemaname,
    tablename,
    tableowner,
    hasindexes,
    hasrules,
    hastriggers
FROM pg_tables 
WHERE schemaname IN ('trading', 'analytics', 'monitoring')
ORDER BY schemaname, tablename;
" > "$BACKUP_DIR/tables_${TIMESTAMP}.txt"

# Backup hypertable information
psql -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE" -c "
SELECT 
    hypertable_schema,
    hypertable_name,
    num_dimensions,
    num_chunks,
    compression_enabled,
    tablespaces
FROM timescaledb_information.hypertables;
" > "$BACKUP_DIR/hypertables_${TIMESTAMP}.txt"

# Backup continuous aggregates (if any)
psql -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE" -c "
SELECT 
    hypertable_schema,
    hypertable_name,
    view_schema,
    view_name,
    view_definition
FROM timescaledb_information.continuous_aggregates;
" > "$BACKUP_DIR/continuous_aggregates_${TIMESTAMP}.txt" 2>/dev/null || echo "No continuous aggregates found"

# Create version info
echo "📋 Recording version information..."
echo "# Schema Backup Information" > "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"
echo "Backup Date: $(date)" >> "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"
echo "Database: $DATABASE" >> "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"
echo "Host: $PRIMARY_HOST" >> "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"

# Get PostgreSQL version
PG_VERSION=$(psql -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE" -t -c "SELECT version();")
echo "PostgreSQL Version: $PG_VERSION" >> "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"

# Get TimescaleDB version  
TS_VERSION=$(psql -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE" -t -c "SELECT extversion FROM pg_extension WHERE extname='timescaledb';")
echo "TimescaleDB Version: $TS_VERSION" >> "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"

# Count records in key tables
echo "" >> "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"
echo "# Table Record Counts" >> "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"

TABLES=("trading.market_data" "trading.data_quality_reports" "trading.orders" "trading.positions")
for table in "${TABLES[@]}"; do
    COUNT=$(psql -h "$PRIMARY_HOST" -p 5432 -U "$USER" -d "$DATABASE" -t -c "SELECT COUNT(*) FROM $table;" 2>/dev/null || echo "0")
    echo "$table: $(echo $COUNT | tr -d ' ')" >> "$BACKUP_DIR/backup_info_${TIMESTAMP}.txt"
done

# Clean up old backups (keep last 10)
echo "🧹 Cleaning up old backups..."
cd "$BACKUP_DIR"
ls -t schema_*.sql | tail -n +11 | xargs -r rm -f
ls -t timescale_metadata_*.sql | tail -n +11 | xargs -r rm -f
ls -t tables_*.txt | tail -n +11 | xargs -r rm -f
ls -t hypertables_*.txt | tail -n +11 | xargs -r rm -f
ls -t continuous_aggregates_*.txt | tail -n +11 | xargs -r rm -f
ls -t backup_info_*.txt | tail -n +11 | xargs -r rm -f

echo "✅ Schema backup process completed successfully!"
echo "📁 Backup files created:"
echo "   - schema_${TIMESTAMP}.sql (main schema)"
echo "   - schema_latest.sql (symlink to latest)"
echo "   - timescale_metadata_${TIMESTAMP}.sql (TimescaleDB metadata)"
echo "   - tables_${TIMESTAMP}.txt (table documentation)"
echo "   - hypertables_${TIMESTAMP}.txt (hypertable info)"
echo "   - backup_info_${TIMESTAMP}.txt (version and stats)"

# Generate CI-friendly output
echo ""
echo "🔄 For CI integration, add this to your workflow:"
echo "docker exec timescale_schema_backup /scripts/backup_schema.sh"
echo "docker cp timescale_schema_backup:/backups/schema_latest.sql ./sql/schema/"