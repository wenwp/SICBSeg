# Utilities to load SAM image encoder and inject ATL/LoRA into last N blocks
# =============================================
from typing import List, Tuple
import os
import sys
import torch
import torch.nn as nn

_SEGMENT_ANYTHING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'segment-anything')
)
if os.path.isdir(_SEGMENT_ANYTHING_DIR) and _SEGMENT_ANYTHING_DIR not in sys.path:
    sys.path.insert(0, _SEGMENT_ANYTHING_DIR)

try:
    from segment_anything import sam_model_registry
except Exception:
    sam_model_registry = None

from atl_modules import ATLAdapter, LoRALinear


def load_sam_image_encoder(sam_type: str, sam_checkpoint: str, device: torch.device) -> nn.Module:
    if sam_model_registry is None:
        raise ImportError("segment_anything not installed. git clone https://github.com/facebookresearch/segment-anything")
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()
    # freeze everything
    for p in sam.parameters():
        p.requires_grad = False
    return sam.image_encoder  # ViT


def _get_vit_blocks(vit: nn.Module) -> List[nn.Module]:
    # SAM ViT has vit.blocks as ModuleList
    if hasattr(vit, 'blocks'):
        return list(vit.blocks)
    # fallback: try encoder.layer for generic ViT
    if hasattr(vit, 'encoder') and hasattr(vit.encoder, 'layers'):
        return list(vit.encoder.layers)
    raise AttributeError('Unsupported ViT structure: cannot find blocks')


def inject_atl_into_vit(vit: nn.Module, n_last_blocks: int, atl_type: str = 'adapter',
                        r: int = 8, alpha: float = 16.0, p: float = 0.1,
                        target_submodules: Tuple[str, ...] = ('attn', 'ffn')) -> List[nn.Module]:
    """Insert ATL modules into the last N transformer blocks.
    - atl_type: 'adapter' or 'lora'
    - For 'adapter': attach small adapter after attn and/or ffn residual outputs.
    - For 'lora': replace Linear layers in attn QKV/Proj with LoRA-wrapped linears.
    Returns a list of trainable modules added/replaced (for optimizer).
    """
    blocks = _get_vit_blocks(vit)
    n = len(blocks)
    start = max(0, n - n_last_blocks)
    trainables: List[nn.Module] = []

    # 获取设备信息
    device = next(vit.parameters()).device
    
    for i in range(start, n):
        blk = blocks[i]
        # heuristics: try to access submodules by common names
        # SAM ViT block fields: norm1, attn, norm2, mlp
        if atl_type == 'adapter':
            d_model = None
            # after attention output
            if 'attn' in target_submodules and hasattr(blk, 'attn'):
                # Attach adapter that accepts token embeddings: (B, N, C)
                # We add a small Sequential wrapper to keep shape.
                if hasattr(blk, 'norm1'):
                    d_model = blk.norm1.normalized_shape[0]
                elif hasattr(blk, 'attn') and hasattr(blk.attn, 'num_heads'):
                    # fallback guess
                    d_model = getattr(blk, 'dim', None)
                assert d_model is not None, 'Cannot infer d_model for adapter'
                adapter_attn = ATLAdapter(d_model, r=r, p=p).to(device)  # 移到正确的设备
                # register as a submodule and insert via forward hook-like pattern:
                # Here we monkey-patch by wrapping blk.attn.forward
                _wrap_with_adapter_after_attn(blk, adapter_attn)
                trainables.append(adapter_attn)
            # after FFN/MLP output
            if 'ffn' in target_submodules and hasattr(blk, 'mlp'):
                if d_model is None:
                    if hasattr(blk, 'norm2'):
                        d_model = blk.norm2.normalized_shape[0]
                adapter_ffn = ATLAdapter(d_model, r=r, p=p).to(device)  # 移到正确的设备
                _wrap_with_adapter_after_mlp(blk, adapter_ffn)
                trainables.append(adapter_ffn)

        elif atl_type == 'lora':
            # Replace Linear layers in attention (qkv/proj) with LoRALinear
            if hasattr(blk, 'attn'):
                attn = blk.attn
                for name, module in list(attn.named_modules()):
                    if isinstance(module, nn.Linear):
                        lora = LoRALinear(module, r=r, alpha=alpha, dropout=p).to(device)  # 移到正确的设备
                        _replace_module(attn, name, lora)
                        trainables.append(lora)
            # Optionally also MLP linears
            if hasattr(blk, 'mlp'):
                mlp = blk.mlp
                for name, module in list(mlp.named_modules()):
                    if isinstance(module, nn.Linear):
                        lora = LoRALinear(module, r=r, alpha=alpha, dropout=p).to(device)  # 移到正确的设备
                        _replace_module(mlp, name, lora)
                        trainables.append(lora)
        else:
            raise ValueError('atl_type must be adapter or lora')

    return trainables


def _replace_module(parent: nn.Module, path: str, new: nn.Module):
    parts = path.split('.')
    obj = parent
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new)


def _wrap_with_adapter_after_attn(block: nn.Module, adapter: nn.Module):
    attn = block.attn
    orig_forward = attn.forward

    # 显式注册为子模块（确保参数能被 state_dict() 捕获）
    # add_module 优于 setattr：跨 PyTorch 版本和 torch.compile 都更安全
    attn.add_module('atl_adapter', adapter)

    def forward(*args, **kwargs):
        x = orig_forward(*args, **kwargs)
        return attn.atl_adapter(x)
    attn.forward = forward


def _wrap_with_adapter_after_mlp(block: nn.Module, adapter: nn.Module):
    mlp = block.mlp
    orig_forward = mlp.forward

    mlp.add_module('atl_adapter', adapter)

    def forward(*args, **kwargs):
        x = orig_forward(*args, **kwargs)
        return mlp.atl_adapter(x)
    mlp.forward = forward
