
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import os
import sys

_SEGMENT_ANYTHING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'segment-anything')
)
if os.path.isdir(_SEGMENT_ANYTHING_DIR) and _SEGMENT_ANYTHING_DIR not in sys.path:
    sys.path.insert(0, _SEGMENT_ANYTHING_DIR)

from segment_anything.modeling import MaskDecoder, TwoWayTransformer
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom


class SAMMaskDecoderHead(nn.Module):
    """
    使用 SAM 原生 mask decoder 进行二分类分割，无需外部 prompt。
    使用可学习的 prompt embeddings 作为查询。
    """
    def __init__(
        self,
        transformer_dim: int = 256,
        num_learnable_prompts: int = 1,
        use_iou_head: bool = False,
    ):
        """
        Args:
            transformer_dim: transformer 通道维度（SAM 默认 256）
            num_learnable_prompts: 可学习 prompt tokens 的数量
            use_iou_head: 是否同时返回 IoU 预测
        """
        super().__init__()

        self.transformer_dim = transformer_dim
        self.num_learnable_prompts = num_learnable_prompts
        self.use_iou_head = use_iou_head

        # SAM 原生 mask decoder
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=transformer_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # 可学习的稀疏 prompt embeddings（替代点/框输入）
        # Shape: [num_prompts, transformer_dim]
        self.learnable_prompts = nn.Parameter(
            torch.randn(num_learnable_prompts, transformer_dim)
        )

        # SAM 风格的 2D sin-cos 位置编码（num_pos_feats * 2 = transformer_dim）
        # PositionEmbeddingRandom 内部含一个 [2, num_pos_feats] 的高斯频率矩阵 buffer，
        # 跟随 state_dict 持久化，训练/推理一致。
        assert transformer_dim % 2 == 0, "transformer_dim must be divisible by 2"
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=transformer_dim // 2)

        # 模拟 SAM prompt_encoder.no_mask_embed：当没有 mask 输入时的稠密先验。
        # SAM 中是 nn.Embedding(1, embed_dim)；这里用同形状的 Parameter，
        # forward 时广播成 [B, C, H, W]。
        self.no_mask_embed = nn.Embedding(1, transformer_dim)

    def _get_image_pe(self, feat_hw: Tuple[int, int]) -> torch.Tensor:
        """生成与 image_embeddings 相同空间维度的位置编码，[1, C, H, W]"""
        # PositionEmbeddingRandom.forward(size) 返回 [C, H, W]
        pe = self.pe_layer(feat_hw)
        return pe.unsqueeze(0)

    def forward(
        self,
        image_embeddings: torch.Tensor,   # [B, C, H, W] 来自 SAM encoder
        original_size: Tuple[int, int],   # 上采样目标 (H, W)
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            image_embeddings: SAM image encoder 的输出 [B, C, H, W]
            original_size: 上采样到的输出尺寸 (H, W)
        Returns:
            logits: [B, 1, H, W]
            或 (logits, iou_predictions) when use_iou_head=True
        """
        B, C, H, W = image_embeddings.shape

        # 1. 位置编码 [1, C, H, W]
        image_pe = self._get_image_pe((H, W)).to(
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )

        # 2. 稀疏 prompt（可学习），扩展到 batch：[B, P, C]
        sparse_prompt_embeddings = self.learnable_prompts.unsqueeze(0)

        # 3. 稠密 prompt 来自 no_mask_embed，[B, C, H, W]
        dense_prompt_embeddings = self.no_mask_embed.weight.reshape(1, C, 1, 1).expand(1, C, H, W)

        # 4. 一次前向跑整个 batch（SAM 的 MaskDecoder 原生支持 batch）
        # Meta's MaskDecoder expands image embeddings per token batch, so decode
        # one image at a time and concatenate to support training batches > 1.
        masks_per_image = []
        iou_per_image = []
        for b in range(B):
            masks_b, iou_b = self.mask_decoder(
                image_embeddings=image_embeddings[b:b + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False,
            )
            masks_per_image.append(masks_b)
            iou_per_image.append(iou_b)

        masks = torch.cat(masks_per_image, dim=0)
        iou_predictions = torch.cat(iou_per_image, dim=0)
        # masks: [B, 1, H*4, W*4], iou_predictions: [B, 1]

        # 5. 上采样到目标尺寸
        logits = F.interpolate(
            masks,
            size=original_size,
            mode='bilinear',
            align_corners=False,
        )

        if self.use_iou_head:
            return logits, iou_predictions
        return logits


class SimpleBinHead(nn.Module):
    """保留原有的 SimpleBinHead 作为备份"""
    def __init__(self, enc_out: int, mid: int = 256):
        super().__init__()
        self.lateral = nn.Conv2d(enc_out, mid, 1)
        self.up1 = nn.ConvTranspose2d(mid, mid, 2, 2)
        self.up2 = nn.ConvTranspose2d(mid, mid, 2, 2)
        self.pred = nn.Conv2d(mid, 1, 1)

    def forward(self, feat: torch.Tensor, out_hw):
        h, w = out_hw
        x = F.gelu(self.lateral(feat))
        x = F.gelu(self.up1(x))
        x = F.gelu(self.up2(x))
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return self.pred(x)
