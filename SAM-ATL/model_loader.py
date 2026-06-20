
from typing import Tuple, Any, Dict
import os
import torch

from sam_atl_inject import load_sam_image_encoder, inject_atl_into_vit
from model_decoder import SAMMaskDecoderHead


def _pick(cfg_in_ckpt: Dict[str, Any], yaml_section: Dict[str, Any], *keys, default=None):
    """优先取 ckpt.cfg 中的字段，找不到再回退到 yaml 段。

    ckpt.cfg 由 train.py 通过 vars(cfg) 写入，字段是扁平命名（如 atl_type、
    last_blocks、num_learnable_prompts），yaml 是分组结构，二者都要兼容。
    """
    for k in keys:
        if cfg_in_ckpt and k in cfg_in_ckpt and cfg_in_ckpt[k] is not None:
            return cfg_in_ckpt[k]
    if yaml_section is not None:
        for k in keys:
            if k in yaml_section and yaml_section[k] is not None:
                return yaml_section[k]
    return default


def resolve_model_config(ckpt: Dict[str, Any], yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """合并 ckpt.cfg 与 yaml 配置，得到最终的模型重建参数。

    Args:
        ckpt:     torch.load 后的字典（含 'cfg', 'enc', 'head' 等）
        yaml_cfg: 完整 yaml 字典（calibrate / infer 用）

    Returns:
        一个扁平 dict，包含 sam_type / atl_type / last_blocks / atl_rank /
        lora_alpha / atl_dropout / num_learnable_prompts / use_iou_head。
    """
    cik = ckpt.get('cfg', {}) or {}
    if ckpt.get('sam_type') is not None and 'sam_type' not in cik:
        cik = {**cik, 'sam_type': ckpt.get('sam_type')}
    atl_yaml = yaml_cfg.get('atl', {}) or {}
    dec_yaml = yaml_cfg.get('decoder', {}) or {}
    model_yaml = yaml_cfg.get('model', {}) or {}

    return {
        'sam_type':              _pick(cik, model_yaml, 'sam_type'),
        'atl_type':              _pick(cik, atl_yaml, 'atl_type', 'type'),
        'last_blocks':           _pick(cik, atl_yaml, 'last_blocks'),
        'atl_rank':              _pick(cik, atl_yaml, 'atl_rank', 'rank'),
        'lora_alpha':            _pick(cik, atl_yaml, 'lora_alpha', default=16.0),
        'atl_dropout':           _pick(cik, atl_yaml, 'atl_dropout', 'dropout', default=0.1),
        'num_learnable_prompts': _pick(cik, dec_yaml, 'num_learnable_prompts', default=1),
        'use_iou_head':          _pick(cik, dec_yaml, 'use_iou_head', default=False),
    }


def build_and_load_model(
    ckpt_path: str,
    sam_checkpoint: str,
    yaml_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, SAMMaskDecoderHead, Dict[str, Any]]:
    """加载 ckpt、重建 encoder + head、灌入权重。

    Returns:
        (encoder, head, resolved_cfg)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    resolved = resolve_model_config(ckpt, yaml_cfg)

    enc = load_sam_image_encoder(resolved['sam_type'], sam_checkpoint, device)
    inject_atl_into_vit(
        enc,
        n_last_blocks=resolved['last_blocks'],
        atl_type=resolved['atl_type'],
        r=resolved['atl_rank'],
        alpha=resolved['lora_alpha'],
        p=resolved['atl_dropout'],
    )

    missing, unexpected = enc.load_state_dict(ckpt['enc'], strict=False)
    # 只要 ATL 配置一致，missing 应该只剩冻结的 SAM 主干（也在 ckpt 里），
    # unexpected 应该为 0；实际打印交给调用方。

    head = SAMMaskDecoderHead(
        transformer_dim=256,
        num_learnable_prompts=resolved['num_learnable_prompts'],
        use_iou_head=resolved['use_iou_head'],
    ).to(device)
    # 旧 ckpt 里的 head 没有 no_mask_embed / pe_layer 等新字段；strict=False 允许加载。
    head.load_state_dict(ckpt['head'], strict=False)

    enc.eval()
    head.eval()
    return enc, head, {'resolved': resolved, 'missing': missing, 'unexpected': unexpected, 'ckpt': ckpt}


def head_logits(output):
    """统一解包 head 的返回值（use_iou_head=True 时是 tuple）。"""
    if isinstance(output, tuple):
        return output[0]
    return output
