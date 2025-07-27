# 🚀 GPU Setup Guide for IntradayJules

## Overview
This guide helps you set up GPU acceleration for your **NVIDIA RTX 3060 Laptop GPU** with **CUDA 12.1+** for both training and real-time trading.

## 🔍 Current Status
- ✅ **GPU Detected**: NVIDIA GeForce RTX 3060 Laptop GPU
- ✅ **CUDA Runtime**: 12.9 (compatible with 12.1+)
- ✅ **Driver Version**: 576.80
- ✅ **GPU Memory**: 6144 MB (6 GB)
- ❌ **PyTorch**: CPU-only version (needs GPU version)
- ❌ **TensorFlow**: CPU-only version (needs GPU version)

## 🚀 Quick Setup (Recommended)

### Option 1: Automated Setup Script
```bash
# Run the automated GPU setup script
python scripts/setup_gpu.py
```

### Option 2: Manual Installation
```bash
# 1. Uninstall CPU versions
pip uninstall torch torchvision torchaudio tensorflow -y

# 2. Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# 4. Install GPU monitoring tools
pip install nvidia-ml-py3 gpustat

# 5. Verify installation
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## 📊 Performance Expectations

### Training Performance Boost
| Component | CPU (Current) | GPU (Expected) | Speedup |
|-----------|---------------|----------------|---------|
| **DQN Training** | ~170 steps/sec | ~800-1200 steps/sec | **5-7x faster** |
| **Buffer Size** | 200k (limited) | 500k+ (more memory) | **2.5x larger** |
| **Batch Size** | 256 (limited) | 512+ (parallel) | **2x larger** |
| **Training Time** | 4 hours | 45-60 minutes | **4-5x faster** |

### Real-Time Trading Benefits
- **Faster inference**: <1ms vs 5-10ms per decision
- **Parallel processing**: Multiple symbols simultaneously
- **Lower latency**: Critical for high-frequency strategies
- **Better resource utilization**: CPU free for other tasks

## 🔧 Configuration Files

### GPU-Optimized Training Config
Use the GPU-optimized configuration for maximum performance:
```bash
python -m src.main --config config/main_config_orchestrator_gpu.yaml
```

**Key GPU optimizations:**
- `buffer_size: 500000` (increased from 200k)
- `batch_size: 512` (increased from 256)
- `total_timesteps: 500000` (increased from 300k)
- `device: auto` (auto-detects GPU)
- `mixed_precision: true` (faster training)

## 🎯 Training Commands

### Production Training with GPU
```bash
# Full production training (6 hours, 500k timesteps)
python -m src.main --config config/main_config_orchestrator_gpu.yaml

# Quick test training (30 minutes, 50k timesteps)
python -m src.main --config config/main_config_orchestrator_test.yaml
```

### Monitor GPU Usage
```bash
# Real-time GPU monitoring
gpustat -i 1

# NVIDIA system monitoring
nvidia-smi

# Watch GPU memory usage during training
watch -n 1 nvidia-smi
```

## 🔍 Verification Steps

### 1. Check GPU Detection
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 2. Test TensorFlow GPU
```python
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpu_devices) > 0}")
print(f"GPU Devices: {gpu_devices}")
```

### 3. Verify Training Uses GPU
Look for these log messages during training:
```
🚀 GPU Auto-detected: NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB)
🚀 GPU Training Mode: NVIDIA GeForce RTX 3060 Laptop GPU
🔧 Device set to: cuda
```

## ⚡ Performance Optimization Tips

### 1. GPU Memory Management
- **Monitor memory usage**: Keep below 80% to avoid OOM errors
- **Adjust batch size**: Increase until you hit memory limits
- **Use mixed precision**: Reduces memory usage by ~40%

### 2. Training Optimization
- **Larger buffer sizes**: More diverse experience replay
- **Higher batch sizes**: Better gradient estimates
- **Parallel environments**: Multiple trading environments

### 3. Real-Time Trading
- **Model caching**: Keep models in GPU memory
- **Batch inference**: Process multiple decisions together
- **Asynchronous processing**: Non-blocking GPU operations

## 🚨 Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"
```bash
# Reduce batch size in config
batch_size: 256  # Instead of 512

# Or reduce buffer size
buffer_size: 300000  # Instead of 500000
```

#### 2. "No CUDA-capable device"
```bash
# Check GPU is detected
nvidia-smi

# Verify CUDA installation
nvcc --version
```

#### 3. "PyTorch not compiled with CUDA"
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Performance Issues

#### 1. Slow Training Despite GPU
- Check GPU utilization: `nvidia-smi`
- Increase batch size if GPU usage <80%
- Verify data loading isn't bottleneck

#### 2. Memory Leaks
- Monitor GPU memory: `watch nvidia-smi`
- Restart training if memory keeps growing
- Use `torch.cuda.empty_cache()` periodically

## 📈 Expected Results

### Before GPU (Current)
- **Training time**: 4 hours for 300k timesteps
- **Average reward**: -138k (needs improvement)
- **Episodes**: 12 (early stopping)
- **Speed**: ~170 timesteps/second

### After GPU (Expected)
- **Training time**: 45-60 minutes for 500k timesteps
- **Average reward**: Positive (with improved hyperparameters)
- **Episodes**: 50+ (more exploration)
- **Speed**: ~800-1200 timesteps/second

## 🎯 Next Steps

1. **Install GPU packages** (see Quick Setup above)
2. **Test GPU training** with test config
3. **Run production training** with GPU config
4. **Monitor performance** and adjust batch sizes
5. **Deploy for real-time trading** with GPU inference

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your CUDA installation
3. Ensure sufficient GPU memory
4. Monitor GPU utilization during training

**Your RTX 3060 is perfect for IntradayJules - expect 5-7x faster training and sub-millisecond inference!** 🚀