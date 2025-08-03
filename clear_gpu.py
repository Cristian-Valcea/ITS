#!/usr/bin/env python3
"""
Clear GPU Memory
"""

import torch
import gc

print("🧹 Clearing GPU memory...")

if torch.cuda.is_available():
    print(f"GPU memory before: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory after: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
else:
    print("CUDA not available")

gc.collect()
print("✅ Memory cleanup completed")