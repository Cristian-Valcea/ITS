# Model Registry

This directory contains versioned models with their metadata and performance metrics.

## Directory Structure
```
registry/
├── v1.0.0-baseline/          # Your current model
│   ├── model/                # Model files
│   ├── metadata.json         # Training config & metrics
│   ├── performance.json      # Episode summaries & analysis
│   └── README.md            # Model description
├── v1.1.0-improved/         # Future versions
└── latest -> v1.0.0-baseline # Symlink to latest stable
```

## Versioning Convention
- **v1.0.0**: Major architecture changes
- **v1.1.0**: Hyperparameter improvements
- **v1.0.1**: Bug fixes or minor tweaks

## Model Promotion Process
1. Train new model
2. Evaluate against baseline
3. If improved, promote to registry
4. Update `latest` symlink
5. Commit to git with tags