"""
Redis Backup and Recovery System
Ensures risk state persistence with AOF + hourly backups to S3/Postgres
"""

import os
import time
import logging
import subprocess
import threading
import json
import gzip
from datetime import datetime, timezone
from typing import Dict, Optional, List
import schedule
import redis
import psycopg2
from psycopg2.extras import Json
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

class RedisBackupManager:
    """
    Comprehensive Redis backup system with multiple layers:
    1. AOF (Append-Only File) for real-time persistence
    2. Hourly compressed backups to local storage
    3. Daily backups to PostgreSQL
    4. Weekly backups to S3 (if configured)
    """
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 backup_dir: str = "/tmp/redis_backups",
                 postgres_url: Optional[str] = None,
                 s3_bucket: Optional[str] = None):
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.backup_dir = backup_dir
        self.postgres_url = postgres_url
        self.s3_bucket = s3_bucket
        
        self.logger = logging.getLogger("RedisBackupManager")
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # PostgreSQL connection (if configured)
        self.postgres_conn = None
        if postgres_url:
            try:
                self.postgres_conn = psycopg2.connect(postgres_url)
                self._create_backup_tables()
            except Exception as e:
                self.logger.error(f"PostgreSQL connection failed: {e}")
        
        # S3 client (if configured)
        self.s3_client = None
        if s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                # Test connection
                self.s3_client.head_bucket(Bucket=s3_bucket)
            except (NoCredentialsError, ClientError) as e:
                self.logger.error(f"S3 connection failed: {e}")
                self.s3_client = None
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup scheduling
        self.scheduler_active = False
        self.scheduler_thread = None
        
        self.logger.info("Redis Backup Manager initialized")
    
    def enable_aof(self) -> bool:
        """Enable Redis AOF (Append-Only File) persistence"""
        try:
            # Enable AOF
            self.redis_client.config_set("appendonly", "yes")
            self.redis_client.config_set("appendfsync", "everysec")  # Sync every second
            
            # Configure AOF rewrite
            self.redis_client.config_set("auto-aof-rewrite-percentage", "100")
            self.redis_client.config_set("auto-aof-rewrite-min-size", "64mb")
            
            # Save configuration
            self.redis_client.config_rewrite()
            
            self.logger.info("Redis AOF enabled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable AOF: {e}")
            return False
    
    def create_snapshot_backup(self, backup_name: Optional[str] = None) -> Optional[str]:
        """Create compressed snapshot backup"""
        if not backup_name:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"redis_backup_{timestamp}"
        
        try:
            # Trigger Redis BGSAVE
            self.redis_client.bgsave()
            
            # Wait for background save to complete
            while self.redis_client.lastsave() == self.redis_client.lastsave():
                time.sleep(0.1)
                # Add timeout to prevent infinite loop
                timeout_counter = 0
                if timeout_counter > 100:  # 10 seconds
                    raise TimeoutError("BGSAVE timeout")
                timeout_counter += 1
            
            # Get Redis data directory
            redis_config = self.redis_client.config_get("dir")
            redis_dir = redis_config.get("dir", "/var/lib/redis")
            dump_file = os.path.join(redis_dir, "dump.rdb")
            
            if not os.path.exists(dump_file):
                raise FileNotFoundError(f"Redis dump file not found: {dump_file}")
            
            # Create compressed backup
            backup_file = os.path.join(self.backup_dir, f"{backup_name}.rdb.gz")
            
            with open(dump_file, 'rb') as f_in:
                with gzip.open(backup_file, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Add metadata
            metadata = {
                "backup_name": backup_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "redis_host": self.redis_host,
                "redis_port": self.redis_port,
                "file_size": os.path.getsize(backup_file),
                "key_count": self.redis_client.dbsize()
            }
            
            metadata_file = os.path.join(self.backup_dir, f"{backup_name}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Snapshot backup created: {backup_file}")
            return backup_file
            
        except Exception as e:
            self.logger.error(f"Snapshot backup failed: {e}")
            return None
    
    def backup_to_postgres(self) -> bool:
        """Backup Redis data to PostgreSQL"""
        if not self.postgres_conn:
            self.logger.warning("PostgreSQL not configured, skipping backup")
            return False
        
        try:
            # Get all Redis keys
            all_keys = self.redis_client.keys("*")
            
            if not all_keys:
                self.logger.info("No Redis keys to backup")
                return True
            
            # Prepare data for backup
            backup_data = {}
            
            for key in all_keys:
                try:
                    key_type = self.redis_client.type(key)
                    
                    if key_type == "string":
                        backup_data[key] = {
                            "type": "string",
                            "value": self.redis_client.get(key),
                            "ttl": self.redis_client.ttl(key)
                        }
                    elif key_type == "hash":
                        backup_data[key] = {
                            "type": "hash", 
                            "value": self.redis_client.hgetall(key),
                            "ttl": self.redis_client.ttl(key)
                        }
                    elif key_type == "list":
                        backup_data[key] = {
                            "type": "list",
                            "value": self.redis_client.lrange(key, 0, -1),
                            "ttl": self.redis_client.ttl(key)
                        }
                    elif key_type == "set":
                        backup_data[key] = {
                            "type": "set",
                            "value": list(self.redis_client.smembers(key)),
                            "ttl": self.redis_client.ttl(key)
                        }
                    elif key_type == "zset":
                        backup_data[key] = {
                            "type": "zset",
                            "value": self.redis_client.zrange(key, 0, -1, withscores=True),
                            "ttl": self.redis_client.ttl(key)
                        }
                
                except Exception as e:
                    self.logger.error(f"Error backing up key {key}: {e}")
            
            # Insert into PostgreSQL
            with self.postgres_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO redis_backups (backup_timestamp, redis_host, redis_port, key_count, data)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    datetime.now(timezone.utc),
                    self.redis_host,
                    self.redis_port,
                    len(backup_data),
                    Json(backup_data)
                ))
            
            self.postgres_conn.commit()
            
            self.logger.info(f"PostgreSQL backup completed: {len(backup_data)} keys")
            return True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL backup failed: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
            return False
    
    def backup_to_s3(self, local_backup_file: str) -> bool:
        """Upload backup file to S3"""
        if not self.s3_client or not self.s3_bucket:
            self.logger.warning("S3 not configured, skipping upload")
            return False
        
        try:
            # Generate S3 key
            filename = os.path.basename(local_backup_file)
            s3_key = f"redis-backups/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
            
            # Upload file
            self.s3_client.upload_file(
                local_backup_file,
                self.s3_bucket, 
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'StorageClass': 'STANDARD_IA'  # Infrequent access for cost savings
                }
            )
            
            self.logger.info(f"S3 backup uploaded: s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"S3 backup failed: {e}")
            return False
    
    def restore_from_snapshot(self, backup_file: str) -> bool:
        """Restore Redis from snapshot backup"""
        try:
            if not os.path.exists(backup_file):
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Stop Redis (this would need proper service management in production)
            self.logger.warning("Manual Redis restart required for restore")
            
            # Get Redis data directory
            redis_config = self.redis_client.config_get("dir")
            redis_dir = redis_config.get("dir", "/var/lib/redis")
            dump_file = os.path.join(redis_dir, "dump.rdb")
            
            # Backup current dump file
            if os.path.exists(dump_file):
                backup_current = f"{dump_file}.backup.{int(time.time())}"
                os.rename(dump_file, backup_current)
                self.logger.info(f"Current dump backed up to: {backup_current}")
            
            # Restore from backup
            if backup_file.endswith('.gz'):
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(dump_file, 'wb') as f_out:
                        f_out.writelines(f_in)
            else:
                import shutil
                shutil.copy2(backup_file, dump_file)
            
            self.logger.info(f"Restored from backup: {backup_file}")
            self.logger.warning("Redis restart required to load restored data")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def restore_from_postgres(self, backup_timestamp: Optional[datetime] = None) -> bool:
        """Restore Redis data from PostgreSQL backup"""
        if not self.postgres_conn:
            self.logger.error("PostgreSQL not configured")
            return False
        
        try:
            with self.postgres_conn.cursor() as cursor:
                if backup_timestamp:
                    cursor.execute("""
                        SELECT data FROM redis_backups 
                        WHERE backup_timestamp = %s
                        ORDER BY backup_timestamp DESC LIMIT 1
                    """, (backup_timestamp,))
                else:
                    cursor.execute("""
                        SELECT data FROM redis_backups 
                        ORDER BY backup_timestamp DESC LIMIT 1
                    """)
                
                result = cursor.fetchone()
                if not result:
                    self.logger.error("No backup found in PostgreSQL")
                    return False
                
                backup_data = result[0]
            
            # Restore data to Redis
            restored_keys = 0
            
            for key, key_data in backup_data.items():
                try:
                    key_type = key_data["type"]
                    value = key_data["value"]
                    ttl = key_data.get("ttl", -1)  # -1 means no expiration
                    
                    if key_type == "string":
                        self.redis_client.set(key, value)
                    elif key_type == "hash":
                        self.redis_client.hset(key, mapping=value)
                    elif key_type == "list":
                        self.redis_client.delete(key)  # Clear first
                        if value:
                            self.redis_client.lpush(key, *reversed(value))
                    elif key_type == "set":
                        self.redis_client.delete(key)
                        if value:
                            self.redis_client.sadd(key, *value)
                    elif key_type == "zset":
                        self.redis_client.delete(key)
                        if value:
                            # Convert list of tuples to flat list for zadd
                            zadd_data = {}
                            for member, score in value:
                                zadd_data[member] = score
                            self.redis_client.zadd(key, zadd_data)
                    
                    # Set TTL if specified
                    if ttl > 0:
                        self.redis_client.expire(key, ttl)
                    
                    restored_keys += 1
                    
                except Exception as e:
                    self.logger.error(f"Error restoring key {key}: {e}")
            
            self.logger.info(f"Restored {restored_keys} keys from PostgreSQL")
            return True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL restore failed: {e}")
            return False
    
    def start_scheduled_backups(self):
        """Start scheduled backup operations"""
        if self.scheduler_active:
            return
        
        # Schedule different types of backups
        schedule.every().hour.do(self._hourly_backup)
        schedule.every().day.at("02:00").do(self._daily_backup)
        schedule.every().sunday.at("03:00").do(self._weekly_backup)
        
        # Cleanup old backups
        schedule.every().day.at("04:00").do(self._cleanup_old_backups)
        
        self.scheduler_active = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Scheduled backups started")
    
    def stop_scheduled_backups(self):
        """Stop scheduled backup operations"""
        schedule.clear()
        self.scheduler_active = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        self.logger.info("Scheduled backups stopped")
    
    def _run_scheduler(self):
        """Run backup scheduler"""
        while self.scheduler_active:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _hourly_backup(self):
        """Hourly snapshot backup"""
        self.logger.info("Running hourly backup")
        backup_file = self.create_snapshot_backup()
        
        if backup_file:
            self.logger.info(f"Hourly backup completed: {backup_file}")
        else:
            self.logger.error("Hourly backup failed")
    
    def _daily_backup(self):
        """Daily PostgreSQL backup"""
        self.logger.info("Running daily PostgreSQL backup")
        success = self.backup_to_postgres()
        
        if success:
            self.logger.info("Daily PostgreSQL backup completed")
        else:
            self.logger.error("Daily PostgreSQL backup failed")
    
    def _weekly_backup(self):
        """Weekly S3 backup"""
        self.logger.info("Running weekly S3 backup")
        
        # Create snapshot and upload to S3
        backup_file = self.create_snapshot_backup()
        
        if backup_file:
            success = self.backup_to_s3(backup_file)
            if success:
                self.logger.info("Weekly S3 backup completed")
            else:
                self.logger.error("Weekly S3 backup failed")
        else:
            self.logger.error("Weekly backup failed - could not create snapshot")
    
    def _cleanup_old_backups(self):
        """Clean up old local backup files"""
        try:
            current_time = time.time()
            retention_days = 7  # Keep 7 days of local backups
            retention_seconds = retention_days * 24 * 60 * 60
            
            removed_count = 0
            
            for filename in os.listdir(self.backup_dir):
                file_path = os.path.join(self.backup_dir, filename)
                
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > retention_seconds:
                        os.remove(file_path)
                        removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old backup files")
        
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    def _create_backup_tables(self):
        """Create PostgreSQL tables for backup storage"""
        try:
            with self.postgres_conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS redis_backups (
                        id SERIAL PRIMARY KEY,
                        backup_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        redis_host VARCHAR(255) NOT NULL,
                        redis_port INTEGER NOT NULL,
                        key_count INTEGER NOT NULL,
                        data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create index for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_redis_backups_timestamp 
                    ON redis_backups(backup_timestamp DESC)
                """)
            
            self.postgres_conn.commit()
            self.logger.info("PostgreSQL backup tables created")
            
        except Exception as e:
            self.logger.error(f"Failed to create backup tables: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
    
    def get_backup_status(self) -> Dict:
        """Get comprehensive backup status"""
        status = {
            "redis_connected": False,
            "aof_enabled": False,
            "postgres_configured": self.postgres_conn is not None,
            "s3_configured": self.s3_client is not None and self.s3_bucket is not None,
            "scheduled_backups_active": self.scheduler_active,
            "local_backups": [],
            "last_backup_times": {}
        }
        
        try:
            # Test Redis connection
            self.redis_client.ping()
            status["redis_connected"] = True
            
            # Check AOF status
            aof_config = self.redis_client.config_get("appendonly")
            status["aof_enabled"] = aof_config.get("appendonly") == "yes"
            
        except Exception as e:
            self.logger.error(f"Error checking Redis status: {e}")
        
        try:
            # List local backups
            if os.path.exists(self.backup_dir):
                backups = []
                for filename in os.listdir(self.backup_dir):
                    if filename.endswith('.rdb.gz'):
                        file_path = os.path.join(self.backup_dir, filename)
                        backups.append({
                            "filename": filename,
                            "size": os.path.getsize(file_path),
                            "modified": os.path.getmtime(file_path)
                        })
                
                status["local_backups"] = sorted(backups, key=lambda x: x["modified"], reverse=True)
        
        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")
        
        return status

# Example usage
def setup_redis_backup_system(postgres_url: Optional[str] = None, 
                              s3_bucket: Optional[str] = None) -> RedisBackupManager:
    """Set up complete Redis backup system"""
    
    backup_manager = RedisBackupManager(
        postgres_url=postgres_url,
        s3_bucket=s3_bucket
    )
    
    # Enable AOF
    backup_manager.enable_aof()
    
    # Start scheduled backups
    backup_manager.start_scheduled_backups()
    
    logging.getLogger("RedisBackup").info("Redis backup system initialized")
    
    return backup_manager