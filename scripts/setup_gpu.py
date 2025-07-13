#!/usr/bin/env python3
"""
GPU Setup Script for IntradayJules

Installs GPU-optimized versions of PyTorch and TensorFlow
for NVIDIA RTX 3060 with CUDA 12.1+
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False

def check_gpu():
    """Check if NVIDIA GPU is available."""
    print("🔍 Checking GPU availability...")
    
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print(result.stdout.split('\n')[0])  # First line with driver info
            return True
        else:
            print("❌ NVIDIA GPU not detected")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found. Please install NVIDIA drivers.")
        return False

def main():
    """Main setup function."""
    print("=" * 80)
    print("🚀 GPU SETUP FOR INTRADAYJULES")
    print("=" * 80)
    print("Setting up GPU acceleration for:")
    print("• NVIDIA RTX 3060 Laptop GPU")
    print("• CUDA 12.1+ Runtime")
    print("• PyTorch with CUDA support")
    print("• TensorFlow with GPU support")
    print("-" * 80)
    
    # Check if we're in the right directory
    if not Path("requirements-gpu.txt").exists():
        print("❌ Error: requirements-gpu.txt not found")
        print("Please run this script from the IntradayJules root directory")
        return False
    
    # Check GPU availability
    if not check_gpu():
        print("\n⚠️ Warning: GPU not detected. Installation will continue but GPU may not work.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    print("\n📦 INSTALLATION STEPS:")
    
    # Step 1: Uninstall CPU versions
    print("\n1️⃣ Removing CPU-only versions...")
    cpu_packages = [
        "torch torchvision torchaudio",
        "tensorflow"
    ]
    
    for package in cpu_packages:
        run_command(f"pip uninstall {package} -y", f"Uninstalling {package}")
    
    # Step 2: Install PyTorch with CUDA
    print("\n2️⃣ Installing PyTorch with CUDA 12.1 support...")
    pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA"):
        print("❌ PyTorch installation failed")
        return False
    
    # Step 3: Install TensorFlow with GPU
    print("\n3️⃣ Installing TensorFlow with GPU support...")
    if not run_command("pip install tensorflow[and-cuda]", "Installing TensorFlow with GPU"):
        print("❌ TensorFlow installation failed")
        return False
    
    # Step 4: Install GPU monitoring tools
    print("\n4️⃣ Installing GPU monitoring tools...")
    gpu_tools = [
        "nvidia-ml-py3",
        "gpustat"
    ]
    
    for tool in gpu_tools:
        run_command(f"pip install {tool}", f"Installing {tool}")
    
    # Step 5: Verify installation
    print("\n5️⃣ Verifying GPU setup...")
    
    verification_script = '''
import torch
import tensorflow as tf

print("🔍 VERIFICATION RESULTS:")
print("-" * 40)

# PyTorch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ PyTorch CUDA not available")

print()

# TensorFlow
print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPU Available: {len(gpu_devices) > 0}")
if gpu_devices:
    print(f"GPU Devices: {len(gpu_devices)}")
    for i, device in enumerate(gpu_devices):
        print(f"  Device {i}: {device}")
else:
    print("❌ TensorFlow GPU not available")

print()
print("✅ Verification complete!")
'''
    
    with open("temp_verify.py", "w") as f:
        f.write(verification_script)
    
    if run_command("python temp_verify.py", "Verifying GPU setup"):
        os.remove("temp_verify.py")
        print("\n🎉 GPU setup completed successfully!")
        print("\n🚀 NEXT STEPS:")
        print("1. Test training with GPU:")
        print("   python -m src.main --config config/main_config_orchestrator_production.yaml")
        print("2. Monitor GPU usage:")
        print("   gpustat -i 1")
        print("3. Check GPU memory:")
        print("   nvidia-smi")
        return True
    else:
        if os.path.exists("temp_verify.py"):
            os.remove("temp_verify.py")
        print("\n❌ GPU setup verification failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)