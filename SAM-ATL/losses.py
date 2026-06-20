# BCE+Dice, Focal, and OHEM utilities for binary segmentation
# =============================================
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def _pw_tensor(pos_weight, device):
    """把标量/None 的 pos_weight 转成 [1] 的 tensor，None 时返回 None。"""
    if pos_weight is None:
        return None
    if isinstance(pos_weight, torch.Tensor):
        return pos_weight.to(device)
    return torch.tensor([float(pos_weight)], device=device)


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: (B,1,H,W), target: (B,H,W) in {0,1,255}
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.new_zeros(())
        probs = torch.sigmoid(logits)
        t = target.float().unsqueeze(1)
        probs = probs[valid.unsqueeze(1)]
        t = t[valid.unsqueeze(1)]
        num = 2 * (probs * t).sum() + self.smooth
        den = (probs * probs).sum() + (t * t).sum() + self.smooth
        return 1 - num / den


def bce_ignore(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0):
    valid = target != IGNORE_INDEX
    if valid.sum() == 0:
        return logits.new_zeros(())
    y = target[valid].float()
    x = logits.squeeze(1)[valid]
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=_pw_tensor(pos_weight, logits.device))


def ohem_topk(loss_map: torch.Tensor, frac: float = 0.3):
    # loss_map: (B,1,H,W) element-wise BCE loss (no reduction, ignore 区域已置 0)
    flat = loss_map.view(-1)
    k = max(1, int(frac * flat.numel()))
    topk = torch.topk(flat, k).values.mean()
    return topk


def elementwise_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0):
    valid = target != IGNORE_INDEX
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    y = safe_target.float().unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=_pw_tensor(pos_weight, logits.device), reduction='none'
    )
    loss = torch.where(valid.unsqueeze(1), loss, torch.zeros_like(loss))
    return loss
