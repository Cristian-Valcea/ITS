#!/usr/bin/env python3
"""
Stairways V3 Cleanup Script
Removes old log files and non-functional partial models, keeping only the latest working Stairways V3 components.
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
import json

class StairwaysV3Cleaner:
    """Clean up old training artifacts while preserving working Stairways V3 model"""
    
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.project_root = Path("/home/cristian/IntradayTrading/ITS")
        
        # The working Stairways V3 model (from your recap and working_ai_trader.py)
        self.working_model_path = "train_runs/v3_gold_standard_400k_20250802_202736"
        
        # Essential files to keep
        self.essential_files = [
            "recap2025-08-05.md",  # Your comprehensive recap
            "working_ai_trader.py",  # Working AI trader
            "verify_ibkr_setup.py",  # IBKR verification
            "src/brokers/ib_gateway.py",  # Enhanced IBKR gateway
        ]
        
        # Essential directories to keep
        self.essential_dirs = [
            "src",  # Core source code
            "config",  # Configuration files
            "data/processed",  # Processed data (if exists)
            self.working_model_path,  # Working Stairways V3 model
        ]
        
        self.cleanup_stats = {
            "log_files_removed": 0,
            "old_models_removed": 0,
            "space_freed_mb": 0,
            "files_kept": 0,
            "dirs_kept": 0
        }
    
    def analyze_disk_usage(self):
        """Analyze current disk usage"""
        print("🔍 ANALYZING DISK USAGE")
        print("=" * 50)
        
        # Check train_runs directory
        train_runs_path = self.project_root / "train_runs"
        if train_runs_path.exists():
            train_runs_size = self._get_dir_size(train_runs_path)
            print(f"📁 train_runs/: {train_runs_size:.1f} MB")
            
            # List all model directories
            model_dirs = [d for d in train_runs_path.iterdir() if d.is_dir()]
            print(f"   Total model directories: {len(model_dirs)}")
            
            # Show working model
            working_model_full_path = self.project_root / self.working_model_path
            if working_model_full_path.exists():
                working_size = self._get_dir_size(working_model_full_path)
                print(f"   ✅ Working model (KEEP): {working_size:.1f} MB")
            
        # Check log files
        log_files = list(self.project_root.glob("*.log"))
        log_size = sum(f.stat().st_size for f in log_files) / (1024 * 1024)
        print(f"📄 Root log files: {len(log_files)} files, {log_size:.1f} MB")
        
        # Check other log files
        all_logs = list(self.project_root.rglob("*.log"))
        all_log_size = sum(f.stat().st_size for f in all_logs if f.exists()) / (1024 * 1024)
        print(f"📄 All log files: {len(all_logs)} files, {all_log_size:.1f} MB")
        
        print()
    
    def identify_cleanup_targets(self):
        """Identify files and directories to clean up"""
        print("🎯 IDENTIFYING CLEANUP TARGETS")
        print("=" * 50)
        
        cleanup_targets = {
            "old_model_dirs": [],
            "log_files": [],
            "temp_files": [],
            "backup_dirs": []
        }
        
        # Find old model directories (keep only working model)
        train_runs_path = self.project_root / "train_runs"
        if train_runs_path.exists():
            for model_dir in train_runs_path.iterdir():
                if model_dir.is_dir() and model_dir.name != Path(self.working_model_path).name:
                    size_mb = self._get_dir_size(model_dir)
                    cleanup_targets["old_model_dirs"].append({
                        "path": model_dir,
                        "size_mb": size_mb,
                        "name": model_dir.name
                    })
        
        # Find log files
        for log_file in self.project_root.rglob("*.log"):
            if log_file.exists() and not self._is_essential_file(log_file):
                size_mb = log_file.stat().st_size / (1024 * 1024)
                cleanup_targets["log_files"].append({
                    "path": log_file,
                    "size_mb": size_mb,
                    "name": str(log_file.relative_to(self.project_root))
                })
        
        # Find temporary and backup files
        temp_patterns = ["*.tmp", "*.temp", "*~", "*.bak"]
        for pattern in temp_patterns:
            for temp_file in self.project_root.rglob(pattern):
                if temp_file.exists():
                    size_mb = temp_file.stat().st_size / (1024 * 1024)
                    cleanup_targets["temp_files"].append({
                        "path": temp_file,
                        "size_mb": size_mb,
                        "name": str(temp_file.relative_to(self.project_root))
                    })
        
        # Find backup directories
        backup_patterns = ["backup_*", "*_backup", "archive*", "archives*"]
        for pattern in backup_patterns:
            for backup_dir in self.project_root.glob(pattern):
                if backup_dir.is_dir():
                    size_mb = self._get_dir_size(backup_dir)
                    cleanup_targets["backup_dirs"].append({
                        "path": backup_dir,
                        "size_mb": size_mb,
                        "name": backup_dir.name
                    })
        
        # Show summary
        total_old_models = len(cleanup_targets["old_model_dirs"])
        total_old_model_size = sum(item["size_mb"] for item in cleanup_targets["old_model_dirs"])
        
        total_logs = len(cleanup_targets["log_files"])
        total_log_size = sum(item["size_mb"] for item in cleanup_targets["log_files"])
        
        total_temp = len(cleanup_targets["temp_files"])
        total_temp_size = sum(item["size_mb"] for item in cleanup_targets["temp_files"])
        
        total_backup = len(cleanup_targets["backup_dirs"])
        total_backup_size = sum(item["size_mb"] for item in cleanup_targets["backup_dirs"])
        
        print(f"🗂️  Old model directories: {total_old_models} ({total_old_model_size:.1f} MB)")
        print(f"📄 Log files: {total_logs} ({total_log_size:.1f} MB)")
        print(f"🗑️  Temp files: {total_temp} ({total_temp_size:.1f} MB)")
        print(f"📦 Backup directories: {total_backup} ({total_backup_size:.1f} MB)")
        
        total_size = total_old_model_size + total_log_size + total_temp_size + total_backup_size
        print(f"💾 Total space to free: {total_size:.1f} MB")
        print()
        
        return cleanup_targets
    
    def show_what_will_be_kept(self):
        """Show what will be preserved"""
        print("✅ WHAT WILL BE KEPT")
        print("=" * 50)
        
        # Working model
        working_model_full_path = self.project_root / self.working_model_path
        if working_model_full_path.exists():
            size_mb = self._get_dir_size(working_model_full_path)
            print(f"🤖 Working Stairways V3 Model: {size_mb:.1f} MB")
            print(f"   Path: {self.working_model_path}")
            
            # Show key files in working model
            key_files = [
                "chunk7_final_358400steps.zip",  # The actual working model
                "TRAINING_SUMMARY.md",
                "validation_results.yaml"
            ]
            
            for key_file in key_files:
                file_path = working_model_full_path / key_file
                if file_path.exists():
                    file_size = file_path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ {key_file}: {file_size:.1f} MB")
        
        # Essential files
        print(f"\n📄 Essential Files:")
        for essential_file in self.essential_files:
            file_path = self.project_root / essential_file
            if file_path.exists():
                file_size = file_path.stat().st_size / 1024  # KB
                print(f"   ✅ {essential_file}: {file_size:.1f} KB")
        
        # Essential directories
        print(f"\n📁 Essential Directories:")
        for essential_dir in self.essential_dirs:
            dir_path = self.project_root / essential_dir
            if dir_path.exists():
                dir_size = self._get_dir_size(dir_path)
                print(f"   ✅ {essential_dir}/: {dir_size:.1f} MB")
        
        print()
    
    def perform_cleanup(self, cleanup_targets):
        """Perform the actual cleanup"""
        if self.dry_run:
            print("🧪 DRY RUN MODE - No files will be deleted")
        else:
            print("🗑️  PERFORMING CLEANUP")
        
        print("=" * 50)
        
        total_freed = 0
        
        # Clean up old model directories
        for item in cleanup_targets["old_model_dirs"]:
            path = item["path"]
            size_mb = item["size_mb"]
            
            print(f"🗂️  Removing old model: {item['name']} ({size_mb:.1f} MB)")
            
            if not self.dry_run:
                try:
                    shutil.rmtree(path)
                    self.cleanup_stats["old_models_removed"] += 1
                    total_freed += size_mb
                except Exception as e:
                    print(f"   ❌ Error removing {path}: {e}")
        
        # Clean up log files
        for item in cleanup_targets["log_files"]:
            path = item["path"]
            size_mb = item["size_mb"]
            
            print(f"📄 Removing log: {item['name']} ({size_mb:.1f} MB)")
            
            if not self.dry_run:
                try:
                    path.unlink()
                    self.cleanup_stats["log_files_removed"] += 1
                    total_freed += size_mb
                except Exception as e:
                    print(f"   ❌ Error removing {path}: {e}")
        
        # Clean up temp files
        for item in cleanup_targets["temp_files"]:
            path = item["path"]
            size_mb = item["size_mb"]
            
            print(f"🗑️  Removing temp: {item['name']} ({size_mb:.1f} MB)")
            
            if not self.dry_run:
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    total_freed += size_mb
                except Exception as e:
                    print(f"   ❌ Error removing {path}: {e}")
        
        # Clean up backup directories
        for item in cleanup_targets["backup_dirs"]:
            path = item["path"]
            size_mb = item["size_mb"]
            
            print(f"📦 Removing backup: {item['name']} ({size_mb:.1f} MB)")
            
            if not self.dry_run:
                try:
                    shutil.rmtree(path)
                    total_freed += size_mb
                except Exception as e:
                    print(f"   ❌ Error removing {path}: {e}")
        
        self.cleanup_stats["space_freed_mb"] = total_freed
        print(f"\n💾 Total space freed: {total_freed:.1f} MB")
        print()
    
    def create_cleanup_summary(self):
        """Create a summary of the cleanup operation"""
        summary_file = self.project_root / f"cleanup_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            "cleanup_date": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "working_model_preserved": self.working_model_path,
            "essential_files_preserved": self.essential_files,
            "essential_dirs_preserved": self.essential_dirs,
            "cleanup_stats": self.cleanup_stats
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Cleanup summary saved to: {summary_file.name}")
        return summary_file
    
    def _get_dir_size(self, path):
        """Get directory size in MB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, PermissionError):
            pass
        return total_size / (1024 * 1024)
    
    def _is_essential_file(self, file_path):
        """Check if a file is essential and should be kept"""
        relative_path = str(file_path.relative_to(self.project_root))
        
        # Keep files in essential directories
        for essential_dir in self.essential_dirs:
            if relative_path.startswith(essential_dir):
                return True
        
        # Keep essential files
        if relative_path in self.essential_files:
            return True
        
        # Keep recent important logs (last 7 days)
        if file_path.name in ["training.log", "validation.log"]:
            try:
                file_age_days = (datetime.now().timestamp() - file_path.stat().st_mtime) / (24 * 3600)
                if file_age_days < 7:
                    return True
            except:
                pass
        
        return False

def main():
    """Main cleanup function"""
    print("🧹 STAIRWAYS V3 CLEANUP TOOL")
    print("=" * 50)
    print("This tool will clean up old log files and non-functional models")
    print("while preserving your working Stairways V3 model and essential files.")
    print()
    
    # Parse command line arguments
    dry_run = "--execute" not in sys.argv
    
    if dry_run:
        print("🧪 Running in DRY RUN mode (no files will be deleted)")
        print("   Use --execute flag to perform actual cleanup")
    else:
        print("⚠️  EXECUTE mode - files will be permanently deleted!")
        response = input("Are you sure you want to proceed? (type 'yes' to confirm): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled.")
            return
    
    print()
    
    # Create cleaner instance
    cleaner = StairwaysV3Cleaner(dry_run=dry_run)
    
    try:
        # Step 1: Analyze current disk usage
        cleaner.analyze_disk_usage()
        
        # Step 2: Identify cleanup targets
        cleanup_targets = cleaner.identify_cleanup_targets()
        
        # Step 3: Show what will be kept
        cleaner.show_what_will_be_kept()
        
        # Step 4: Perform cleanup
        cleaner.perform_cleanup(cleanup_targets)
        
        # Step 5: Create summary
        summary_file = cleaner.create_cleanup_summary()
        
        print("✅ CLEANUP COMPLETE!")
        print("=" * 50)
        print(f"📊 Statistics:")
        print(f"   Old models removed: {cleaner.cleanup_stats['old_models_removed']}")
        print(f"   Log files removed: {cleaner.cleanup_stats['log_files_removed']}")
        print(f"   Space freed: {cleaner.cleanup_stats['space_freed_mb']:.1f} MB")
        
        if dry_run:
            print(f"\n🧪 This was a dry run. To execute cleanup, run:")
            print(f"   python {sys.argv[0]} --execute")
        else:
            print(f"\n🎉 Your Stairways V3 system is now clean and optimized!")
            print(f"   Working model preserved: {cleaner.working_model_path}")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())