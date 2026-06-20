# Lightweight ATL implementations: Adapter & LoRA
# =============================================
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLAdapter(nn.Module):
    """Adapter-style ATL: Down(r) -> Act -> Up(d_model) with residual.
    Zero-init Up so the module starts as identity.
    """
    def __init__(self, d_model: int, r: int = 8, p: float = 0.1):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(r, d_model, bias=False)
        nn.init.zeros_(self.up.weight)  # critical for stable start
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.up(self.act(self.down(x))))


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear: W' = W + alpha/r * (B @ A).
    - freeze base weight by default; only A/B are trainable.
    - supports 2D input (..., in_features)
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        # copy base
        self.weight = nn.Parameter(base.weight.data, requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data, requires_grad=False)
        # LoRA factors
        self.A = nn.Linear(self.in_features, r, bias=False)
        self.B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)  # start from 0 -> identity
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.B(self.A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out
