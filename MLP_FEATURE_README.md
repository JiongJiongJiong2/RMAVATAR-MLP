# MLP Feature Field for RMAvatar

## Overview

This modification replaces the original HexPlane (Tri-plane) feature extraction with a pure MLP + Positional Encoding approach for canonical space feature extraction.

## Changes Made

### 1. New File: `scene/mlp_feature.py`
- Implements `MLPFeatureField` class with NeRF-style positional encoding
- Implements `MLPFeatureFieldLite` class (lightweight version)
- Features:
  - Positional encoding for spatial coordinates (xyz) and time (t)
  - Skip connections (like NeRF)
  - Xavier weight initialization

### 2. Modified: `scene/deformation.py`
- Added import for `MLPFeatureField` and `MLPFeatureFieldLite`
- Modified `Deformation.__init__` to support feature field type switching via `args.feature_type`
- Supports both `'hexplane'` (original) and `'mlp'` (new) modes

### 3. Modified: `model/rmavatar_model.py`
- Added `_merge_config_args` helper method to merge `config.model` into args namespace
- This allows deformation network to access MLP configuration parameters

### 4. New Config: `configs/peoplesnapshot_mlp.yaml`
- MLP-specific configuration file
- Key parameters:
  - `feature_type: mlp` - enables MLP feature field
  - `mlp_feat_dim: 64` - output feature dimension
  - `mlp_pos_pe: 6` - positional encoding frequency for xyz
  - `mlp_time_pe: 4` - positional encoding frequency for time
  - `mlp_hidden_dim: 256` - MLP hidden layer dimension
  - `mlp_num_layers: 4` - number of MLP layers
  - `mlp_skips: [2]` - skip connection layers
  - `mlp_lite: False` - use lightweight version

## Usage

### Using MLP Feature Field
```bash
python train_rmavatar.py --dat_dir /path/to/data --configs peoplesnapshot_mlp.yaml --deform_on 1
```

### Using Original HexPlane (Default)
```bash
python train_rmavatar.py --dat_dir /path/to/data --configs peoplesnapshot.yaml --deform_on 1
```

## Technical Details

### HexPlane vs MLP Comparison

| Aspect | HexPlane | MLP |
|--------|----------|-----|
| Storage | Grid planes (explicit) | MLP weights (implicit) |
| Resolution | Limited by grid size | Continuous (infinite resolution) |
| Memory | Higher (stores grids) | Lower (only MLP weights) |
| Interpolation | Bilinear grid sample | Direct MLP forward |
| Extrapolation | Border padding | Smooth extrapolation |

### MLP Architecture

```
Input: (x, y, z, t) normalized to [-1, 1]
    ↓
Positional Encoding:
  - pos: 3 + 3 * 2 * 6 = 39 dims
  - time: 1 + 1 * 2 * 4 = 9 dims
  - total: 48 dims
    ↓
MLP Layers (with skip connection at layer 2):
  Layer 0: Linear(48, 256) + ReLU
  Layer 1: Linear(256, 256) + ReLU
  Layer 2: Linear(48+256, 256) + ReLU (skip)
  Layer 3: Linear(256, 256) + ReLU
    ↓
Output: Linear(256, 64) → features
```

## Author

[Your Name] - Graduation Thesis Research

## License

Same as original RMAvatar project.