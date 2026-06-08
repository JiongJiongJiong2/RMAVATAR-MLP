# RMAvatar-MLP: MLP-based Feature Field for Canonical Space Representation

This repository is a modified version of [RMAvatar](https://github.com/RMAvatar/RMAvatar), where we replace the original HexPlane (Tri-plane) feature extraction module with a pure **MLP + Positional Encoding** approach for canonical space feature representation.

The research was a part of our Final Year Project of our undergraduate.

## Key Modification

### Original Approach: HexPlane (Tri-plane)
The original RMAvatar uses a HexPlane field that stores features on 6 learnable 2D grid planes (xy, yz, xz, xt, yt, zt). Features are extracted via bilinear interpolation (`grid_sample`) on these planes, then aggregated to produce deformation features.

### Our Approach: MLP + Positional Encoding
We replace the explicit grid-based feature storage with an implicit MLP network that takes spatial-temporal coordinates as input and outputs deformation features directly. This approach:

- **Reduces memory footprint**: No need to store large grid plane parameters
- **Provides continuous representation**: MLP produces smooth, continuous features without discretization artifacts
- **Enables better generalization**: Implicit representation can extrapolate beyond training data more smoothly
- **Simplifies the architecture**: Single unified network instead of multiple grid planes

## Architecture

```
Input: (x, y, z, t) normalized to [-1, 1]
    │
    ▼
Positional Encoding:
  - Spatial (xyz): 3 + 3 × 2 × 6 = 39 dimensions
  - Temporal (t):  1 + 1 × 2 × 4 = 9 dimensions
  - Total input:   48 dimensions
    │
    ▼
MLP with Skip Connections:
  Layer 0: Linear(48, 256) + ReLU
  Layer 1: Linear(256, 256) + ReLU
  Layer 2: Linear(48+256, 256) + ReLU  ← skip connection from input
  Layer 3: Linear(256, 256) + ReLU
    │
    ▼
Output: Linear(256, 64) → feature vector
```

## Comparison

| Aspect | HexPlane (Original) | MLP + PE (Ours) |
|--------|---------------------|------------------|
| Storage | Explicit grid planes | Implicit MLP weights |
| Resolution | Limited by grid size | Continuous (infinite resolution) |
| Memory | Higher (stores multiple grids) | Lower (only MLP parameters) |
| Interpolation | Bilinear grid sampling | Direct MLP forward pass |
| Extrapolation | Border padding | Smooth extrapolation |
| Feature smoothness | Discrete (grid-dependent) | Continuous (network-dependent) |

## Files Modified

| File | Description |
|------|-------------|
| `scene/mlp_feature.py` | **New file**: MLP feature field implementation with positional encoding |
| `scene/deformation.py` | Modified to support switching between HexPlane and MLP via configuration |
| `model/rmavatar_model.py` | Added config merging to pass MLP parameters to deformation network |
| `configs/peoplesnapshot_mlp.yaml` | **New file**: MLP-specific configuration |

## Usage

### Training with MLP Feature Field
```bash
python train_rmavatar.py --dat_dir /path/to/data --configs peoplesnapshot_mlp.yaml --deform_on 1
```

### Training with Original HexPlane (Default)
```bash
python train_rmavatar.py --dat_dir /path/to/data --configs peoplesnapshot.yaml --deform_on 1
```

## Configuration Parameters

The following parameters can be configured in `configs/peoplesnapshot_mlp.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_type` | `'mlp'` | Feature extraction type: `'hexplane'` or `'mlp'` |
| `mlp_feat_dim` | 64 | Output feature dimension |
| `mlp_pos_pe` | 6 | Positional encoding frequency bands for xyz |
| `mlp_time_pe` | 4 | Positional encoding frequency bands for time |
| `mlp_hidden_dim` | 256 | MLP hidden layer dimension |
| `mlp_num_layers` | 4 | Number of MLP layers |
| `mlp_skips` | [2] | Layers with skip connections (NeRF-style) |
| `mlp_lite` | False | Use lightweight MLP variant |

## Implementation Details

### Positional Encoding
We adopt the NeRF-style positional encoding to map low-dimensional coordinates to a higher-dimensional space:

$$\gamma(p, L) = \left[p, \sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)\right]$$

where $p$ is the input coordinate and $L$ is the number of frequency bands.

### Skip Connections
Following the NeRF architecture, we inject the input encoding back into intermediate layers (default: layer 2) to improve gradient flow and help the network learn both low-frequency and high-frequency details.

### Compatibility
The `MLPFeatureField` class provides the same interface as `HexPlaneField`, including:
- `forward(pts, timestamps)` → feature extraction
- `get_aabb` / `set_aabb()` → bounding box management
- `feat_dim` property → output dimension

This ensures seamless integration with the existing deformation pipeline without modifying downstream components.

## Acknowledgements

This work is built upon [RMAvatar](https://github.com/RMAvatar/RMAvatar). We thank the authors for their excellent work and open-source contribution.

## License

This project follows the same license as the original RMAvatar repository.
