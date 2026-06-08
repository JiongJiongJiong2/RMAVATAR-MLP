# MLP-based Feature Field for Canonical Space
# Replaces HexPlane with pure MLP + Positional Encoding
# Contributer(s): [Your Name]
# All rights reserved. Prometheus 2022-2024.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normalize_aabb(pts, aabb):
    """Normalize points to [-1, 1] range based on AABB"""
    return (pts - aabb[1]) * (2.0 / (aabb[0] - aabb[1])) - 1.0


def positional_encoding(x, L=6, include_input=True):
    """
    Positional encoding as in NeRF.
    x: input tensor [..., D]
    L: number of frequency bands
    Returns: [..., D * (2L + 1)] if include_input else [..., D * 2L]
    """
    device = x.device
    freq_bands = 2.0 ** torch.arange(L, device=device).float() * torch.pi
    
    out = []
    if include_input:
        out.append(x)
    
    for freq in freq_bands:
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))
    
    return torch.cat(out, dim=-1)


class MLPFeatureField(nn.Module):
    """
    MLP-based feature field with positional encoding.
    Replaces HexPlaneField for canonical feature extraction.
    
    Args:
        bounds: AABB bounds for normalization
        feat_dim: output feature dimension
        pos_pe: positional encoding frequency bands for xyz
        time_pe: positional encoding frequency bands for time
        hidden_dim: MLP hidden layer dimension
        num_layers: number of MLP layers
        skips: skip connection layers (like NeRF)
    """
    def __init__(
        self, 
        bounds=1.6,
        feat_dim=64,
        pos_pe=6,
        time_pe=4,
        hidden_dim=256,
        num_layers=4,
        skips=[2],
    ):
        super().__init__()
        
        # AABB bounds
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        
        self.feat_dim = feat_dim
        self.pos_pe = pos_pe
        self.time_pe = time_pe
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skips = skips
        
        # Calculate input dimension
        # pos: 3 + 3 * 2 * pos_pe (include input + sin/cos)
        # time: 1 + 1 * 2 * time_pe
        self.pos_dim = 3 + 3 * 2 * pos_pe
        self.time_dim = 1 + 1 * 2 * time_pe
        self.input_dim = self.pos_dim + self.time_dim
        
        # Build MLP with skip connections
        self.layers = nn.ModuleList()
        in_dim = self.input_dim
        for i in range(num_layers):
            if i in skips:
                self.layers.append(nn.Linear(in_dim + self.input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, feat_dim)
        
        # Initialize weights
        self._init_weights()
        
        print(f"[MLPFeatureField] Initialized with feat_dim={feat_dim}, "
              f"pos_pe={pos_pe}, time_pe={time_pe}, hidden_dim={hidden_dim}, "
              f"num_layers={num_layers}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self, xyz_max, xyz_min):
        """Set AABB from mesh bounds"""
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ], dtype=torch.float32).cuda()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print(f"[MLPFeatureField] Set aabb={self.aabb}")
    
    def forward(self, pts: torch.Tensor, timestamps: torch.Tensor = None):
        """
        Forward pass to extract features.
        
        Args:
            pts: [..., 3] point coordinates
            timestamps: [..., 1] time values (optional)
        
        Returns:
            features: [..., feat_dim]
        """
        # Normalize to [-1, 1]
        pts = normalize_aabb(pts, self.aabb)
        
        # Positional encoding for spatial coordinates
        pts_emb = positional_encoding(pts, self.pos_pe, include_input=True)
        
        # Handle timestamps
        if timestamps is not None:
            t_emb = positional_encoding(timestamps, self.time_pe, include_input=True)
            h = torch.cat([pts_emb, t_emb], dim=-1)
        else:
            # If no timestamp, use zeros
            t_emb = torch.zeros(*pts.shape[:-1], self.time_dim, device=pts.device)
            h = torch.cat([pts_emb, t_emb], dim=-1)
        
        # Reshape for MLP
        original_shape = h.shape[:-1]
        h = h.reshape(-1, h.shape[-1])
        
        # MLP forward with skip connections
        input_h = h
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, input_h], dim=-1)
            h = F.relu(layer(h))
        
        # Output
        out = self.output_layer(h)
        out = out.reshape(*original_shape, self.feat_dim)
        
        return out
    
    def get_density(self, pts: torch.Tensor, timestamps: torch.Tensor = None):
        """Compatibility interface with HexPlaneField"""
        return self.forward(pts, timestamps)


class MLPFeatureFieldLite(nn.Module):
    """
    Lightweight version of MLP feature field.
    Fewer parameters, suitable for smaller scenes.
    """
    def __init__(
        self,
        bounds=1.6,
        feat_dim=64,
        pos_pe=4,
        time_pe=2,
        hidden_dim=128,
        num_layers=3,
    ):
        super().__init__()
        
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        
        self.feat_dim = feat_dim
        self.pos_pe = pos_pe
        self.time_pe = time_pe
        
        # Input dimensions
        pos_dim = 3 + 3 * 2 * pos_pe
        time_dim = 1 + 1 * 2 * time_pe
        input_dim = pos_dim + time_dim
        
        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        
        print(f"[MLPFeatureFieldLite] Initialized with feat_dim={feat_dim}")
    
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32).cuda()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
    
    def forward(self, pts, timestamps=None):
        pts = normalize_aabb(pts, self.aabb)
        pts_emb = positional_encoding(pts, self.pos_pe, include_input=True)
        
        if timestamps is not None:
            t_emb = positional_encoding(timestamps, self.time_pe, include_input=True)
            h = torch.cat([pts_emb, t_emb], dim=-1)
        else:
            t_emb = torch.zeros(*pts.shape[:-1], 1 + 2 * self.time_pe, device=pts.device)
            h = torch.cat([pts_emb, t_emb], dim=-1)
        
        return self.mlp(h)
    
    def get_density(self, pts, timestamps=None):
        return self.forward(pts, timestamps)