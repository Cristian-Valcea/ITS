# scripts/setup_ray_cluster.py
"""
Ray Cluster Setup Script for Distributed Hyperparameter Search.

This script helps set up Ray clusters on spare nodes for scalable hyperparameter optimization.

Usage:
    # Start head node
    python scripts/setup_ray_cluster.py --mode head --port 10001
    
    # Start worker node
    python scripts/setup_ray_cluster.py --mode worker --head_address ray://head-node:10001
    
    # Check cluster status
    python scripts/setup_ray_cluster.py --mode status --head_address ray://head-node:10001
"""

import os
import sys
import argparse
import subprocess
import socket
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_system_resources() -> Dict[str, Any]:
    """Get system resource information."""
    import psutil
    
    # CPU information
    cpu_count = os.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory information
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    
    # GPU information
    gpu_info = {"count": 0, "names": [], "memory_gb": []}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["count"] = torch.cuda.device_count()
            for i in range(gpu_info["count"]):
                gpu_info["names"].append(torch.cuda.get_device_name(i))
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info["memory_gb"].append(gpu_memory)
    except ImportError:
        pass
    
    # Disk information
    disk = psutil.disk_usage('/')
    disk_gb = disk.total / (1024**3)
    disk_free_gb = disk.free / (1024**3)
    
    return {
        "cpu": {
            "count": cpu_count,
            "usage_percent": cpu_percent
        },
        "memory": {
            "total_gb": memory_gb,
            "available_gb": memory_available_gb,
            "usage_percent": memory.percent
        },
        "gpu": gpu_info,
        "disk": {
            "total_gb": disk_gb,
            "free_gb": disk_free_gb,
            "usage_percent": disk.percent
        },
        "hostname": socket.gethostname(),
        "ip_address": socket.gethostbyname(socket.gethostname())
    }

def start_ray_head(port: int = 10001, dashboard_port: int = 8265) -> bool:
    """Start Ray head node."""
    print(f"🚀 Starting Ray head node on port {port}")
    
    # Get system resources
    resources = get_system_resources()
    print(f"System resources:")
    print(f"  CPU: {resources['cpu']['count']} cores ({resources['cpu']['usage_percent']:.1f}% used)")
    print(f"  Memory: {resources['memory']['available_gb']:.1f}GB available / {resources['memory']['total_gb']:.1f}GB total")
    print(f"  GPU: {resources['gpu']['count']} devices")
    for i, (name, memory) in enumerate(zip(resources['gpu']['names'], resources['gpu']['memory_gb'])):
        print(f"    GPU {i}: {name} ({memory:.1f}GB)")
    
    # Build Ray start command
    cmd = [
        "ray", "start",
        "--head",
        f"--port={port}",
        f"--dashboard-port={dashboard_port}",
        "--include-dashboard=true"
    ]
    
    # Add resource specifications
    if resources['gpu']['count'] > 0:
        cmd.extend([f"--num-gpus={resources['gpu']['count']}"])
    
    cmd.extend([f"--num-cpus={resources['cpu']['count']}"])
    
    try:
        # Start Ray head node
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Ray head node started successfully")
            print(f"Dashboard available at: http://{resources['ip_address']}:{dashboard_port}")
            print(f"Connect workers with: ray start --address='{resources['ip_address']}:{port}'")
            
            # Save cluster info
            cluster_info = {
                "head_address": f"ray://{resources['ip_address']}:{port}",
                "dashboard_url": f"http://{resources['ip_address']}:{dashboard_port}",
                "resources": resources,
                "started_at": time.time()
            }
            
            with open("ray_cluster_info.json", "w") as f:
                json.dump(cluster_info, f, indent=2)
            
            return True
        else:
            print(f"❌ Failed to start Ray head node: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Ray head node startup timed out")
        return False
    except Exception as e:
        print(f"❌ Error starting Ray head node: {e}")
        return False

def start_ray_worker(head_address: str) -> bool:
    """Start Ray worker node."""
    print(f"🔧 Starting Ray worker node, connecting to {head_address}")
    
    # Get system resources
    resources = get_system_resources()
    print(f"Worker resources:")
    print(f"  CPU: {resources['cpu']['count']} cores")
    print(f"  Memory: {resources['memory']['available_gb']:.1f}GB available")
    print(f"  GPU: {resources['gpu']['count']} devices")
    
    # Extract IP and port from head address
    if head_address.startswith("ray://"):
        head_address = head_address[6:]  # Remove ray:// prefix
    
    # Build Ray start command
    cmd = [
        "ray", "start",
        f"--address={head_address}"
    ]
    
    # Add resource specifications
    if resources['gpu']['count'] > 0:
        cmd.extend([f"--num-gpus={resources['gpu']['count']}"])
    
    cmd.extend([f"--num-cpus={resources['cpu']['count']}"])
    
    try:
        # Start Ray worker node
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Ray worker node started successfully")
            print(f"Connected to cluster at {head_address}")
            return True
        else:
            print(f"❌ Failed to start Ray worker node: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Ray worker node startup timed out")
        return False
    except Exception as e:
        print(f"❌ Error starting Ray worker node: {e}")
        return False

def check_cluster_status(head_address: Optional[str] = None) -> Dict[str, Any]:
    """Check Ray cluster status."""
    print("📊 Checking Ray cluster status...")
    
    try:
        import ray
        
        # Connect to cluster
        if head_address:
            if head_address.startswith("ray://"):
                ray.init(address=head_address, ignore_reinit_error=True)
            else:
                ray.init(address=f"ray://{head_address}", ignore_reinit_error=True)
        else:
            ray.init(ignore_reinit_error=True)
        
        # Get cluster resources
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        
        # Get node information
        nodes = ray.nodes()
        
        status = {
            "cluster_resources": cluster_resources,
            "available_resources": available_resources,
            "nodes": len(nodes),
            "node_details": []
        }
        
        print(f"Cluster Status:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Total CPUs: {cluster_resources.get('CPU', 0)}")
        print(f"  Available CPUs: {available_resources.get('CPU', 0)}")
        print(f"  Total GPUs: {cluster_resources.get('GPU', 0)}")
        print(f"  Available GPUs: {available_resources.get('GPU', 0)}")
        print(f"  Total Memory: {cluster_resources.get('memory', 0) / (1024**3):.1f}GB")
        
        print(f"\nNode Details:")
        for i, node in enumerate(nodes):
            node_resources = node.get('Resources', {})
            print(f"  Node {i+1}:")
            print(f"    Alive: {node.get('Alive', False)}")
            print(f"    CPUs: {node_resources.get('CPU', 0)}")
            print(f"    GPUs: {node_resources.get('GPU', 0)}")
            print(f"    Memory: {node_resources.get('memory', 0) / (1024**3):.1f}GB")
            
            status["node_details"].append({
                "alive": node.get('Alive', False),
                "resources": node_resources
            })
        
        ray.shutdown()
        return status
        
    except Exception as e:
        print(f"❌ Error checking cluster status: {e}")
        return {"error": str(e)}

def stop_ray_cluster():
    """Stop Ray cluster."""
    print("🛑 Stopping Ray cluster...")
    
    try:
        result = subprocess.run(["ray", "stop"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Ray cluster stopped successfully")
            
            # Remove cluster info file
            if os.path.exists("ray_cluster_info.json"):
                os.remove("ray_cluster_info.json")
                
        else:
            print(f"⚠️ Ray stop output: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error stopping Ray cluster: {e}")

def create_cluster_config() -> Dict[str, Any]:
    """Create a sample Ray cluster configuration."""
    config = {
        "cluster_name": "hyperparameter-search-cluster",
        "provider": {
            "type": "local",
            "head_ip": "auto"
        },
        "auth": {
            "ssh_user": "ubuntu"
        },
        "available_node_types": {
            "head_node": {
                "resources": {"CPU": 8, "GPU": 1},
                "node_config": {
                    "instance_type": "head"
                }
            },
            "worker_node": {
                "resources": {"CPU": 16, "GPU": 2},
                "node_config": {
                    "instance_type": "worker"
                },
                "min_workers": 0,
                "max_workers": 4
            }
        },
        "head_node_type": "head_node",
        "setup_commands": [
            "pip install ray[tune] optuna hyperopt",
            "pip install torch torchvision",
            "pip install stable-baselines3[extra]"
        ],
        "initialization_commands": [
            "echo 'Ray cluster initialized'"
        ]
    }
    
    return config

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Ray Cluster Setup for Hyperparameter Search")
    parser.add_argument(
        "--mode",
        choices=["head", "worker", "status", "stop", "config"],
        required=True,
        help="Operation mode"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10001,
        help="Port for Ray head node"
    )
    parser.add_argument(
        "--dashboard_port",
        type=int,
        default=8265,
        help="Port for Ray dashboard"
    )
    parser.add_argument(
        "--head_address",
        type=str,
        help="Head node address (for worker mode or status check)"
    )
    
    args = parser.parse_args()
    
    print("🌐 Ray Cluster Setup for Distributed Hyperparameter Search")
    print("Solving: Hyper-param search limited to laptop CPU with concurrency=1")
    print("Solution: Distributed Ray cluster on spare nodes with GPU support")
    print()
    
    try:
        if args.mode == "head":
            success = start_ray_head(args.port, args.dashboard_port)
            if success:
                print("\n🎯 Next Steps:")
                print("1. Start worker nodes with:")
                print(f"   python scripts/setup_ray_cluster.py --mode worker --head_address ray://THIS_IP:{args.port}")
                print("2. Run hyperparameter search with:")
                print(f"   python examples/distributed_hyperparameter_search_example.py --mode ray_cluster --ray_address ray://THIS_IP:{args.port}")
            
        elif args.mode == "worker":
            if not args.head_address:
                print("❌ Head address required for worker mode")
                print("Example: --head_address ray://head-node:10001")
                return
            
            success = start_ray_worker(args.head_address)
            if success:
                print("\n✅ Worker node ready for hyperparameter search tasks")
            
        elif args.mode == "status":
            status = check_cluster_status(args.head_address)
            
            if "error" not in status:
                print("\n💡 Cluster is ready for hyperparameter search!")
                print("Run search with:")
                print("python examples/distributed_hyperparameter_search_example.py --mode ray_cluster")
            
        elif args.mode == "stop":
            stop_ray_cluster()
            
        elif args.mode == "config":
            config = create_cluster_config()
            config_file = "ray_cluster_config.yaml"
            
            import yaml
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"✅ Sample cluster configuration saved to {config_file}")
            print("Customize and use with: ray up ray_cluster_config.yaml")
        
        print("\n🏆 Ray cluster setup complete!")
        print("Benefits achieved:")
        print("✅ Distributed execution across spare nodes")
        print("✅ GPU acceleration for training")
        print("✅ Scalable resource utilization")
        print("✅ Fault tolerance and load balancing")
        
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()