# Class-collapsed dataset: map multi-class mask to binary destroyed label
# Supports single-folder images with suffix pre/post; we use *post.* only.
# =============================================
import os
import random
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IGNORE_INDEX = 255

class DestroyedBinaryDataset(Dataset):

    exts = ('.jpg','.jpeg','.png','.tif','.tiff')
    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 1024,
                 destroyed_idx: int = 3,  # for 4-class setup: 3 is destroyed
                 is_train: bool = True,
                 destroyed_value_is_255: bool = False,
                 require_post_suffix: bool = False,
                 augmentation: Optional[Dict[str, Any]] = None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.is_train = is_train
        self.destroyed_idx = destroyed_idx
        self.destroyed_value_is_255 = destroyed_value_is_255
        self.require_post_suffix = require_post_suffix


        aug = augmentation or {}
        self.aug_enabled = is_train and aug.get('enabled', False)
        self.aug_hflip = aug.get('hflip', True)
        self.aug_vflip = aug.get('vflip', True)
        self.aug_rotate90 = aug.get('rotate90', True)
        cj = aug.get('color_jitter', {})
        self.aug_color_jitter = cj.get('enabled', True) if isinstance(cj, dict) else bool(cj)
        self.aug_brightness = cj.get('brightness', 0.25) if isinstance(cj, dict) else 0.25
        self.aug_contrast = cj.get('contrast', 0.25) if isinstance(cj, dict) else 0.25
        self.aug_saturation = cj.get('saturation', 0.1) if isinstance(cj, dict) else 0.1
        
        files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(self.exts)])
        self.items: List[Tuple[str,str]] = []
        
        for name in files:
            base, ext = os.path.splitext(name)
            

            if self.require_post_suffix and not base.endswith('post'):
                continue
            
            img_path = os.path.join(images_dir, name)

            label = None

            cand = os.path.join(labels_dir, name)
            if os.path.exists(cand):
                label = cand
            else:

                cand2 = os.path.join(labels_dir, base + '_target' + ext)
                if os.path.exists(cand2):
                    label = cand2
                else:

                    for try_ext in self.exts:
                        cand3 = os.path.join(labels_dir, base + try_ext)
                        if os.path.exists(cand3):
                            label = cand3
                            break
            
            if label is not None:
                self.items.append((img_path, label))
        
        if len(self.items) == 0:
            error_msg = f'No valid image-label pairs found in {images_dir}'
            if self.require_post_suffix:
                error_msg += '\n提示: require_post_suffix=True，但未找到以"post"结尾的图像文件'
                error_msg += '\n如果你的图像不使用pre/post命名，请设置 require_post_suffix: false'
            raise ValueError(error_msg)

    def __len__(self):
        return len(self.items)

    def _augment(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """训练阶段数据增强：几何变换同步施加于图像和掩码，颜色变换仅施加于图像。"""

        # 1. 随机水平翻转
        if self.aug_hflip and random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2. 随机垂直翻转
        if self.aug_vflip and random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # 3. 随机 90° 倍数旋转
        # expand=True：旋转后图像宽高自动交换（如 600×800 → 800×600），
        # 不产生任何 fill 区域，后续 resize 步骤统一缩放，完全安全。
        if self.aug_rotate90:
            k = random.randint(0, 3)
            if k > 0:
                angle = k * 90
                img = TF.rotate(img, angle,
                                interpolation=TF.InterpolationMode.BILINEAR,
                                expand=True)
                mask = TF.rotate(mask, angle,
                                 interpolation=TF.InterpolationMode.NEAREST,
                                 expand=True)

        # 4. 颜色抖动（仅图像，掩码不变）
        if self.aug_color_jitter and random.random() < 0.8:
            bf = random.uniform(max(0.0, 1.0 - self.aug_brightness),
                                1.0 + self.aug_brightness)
            cf = random.uniform(max(0.0, 1.0 - self.aug_contrast),
                                1.0 + self.aug_contrast)
            sf = random.uniform(max(0.0, 1.0 - self.aug_saturation),
                                1.0 + self.aug_saturation)
            img = TF.adjust_brightness(img, bf)
            img = TF.adjust_contrast(img, cf)
            img = TF.adjust_saturation(img, sf)

        return img, mask

    def _transform(self, img: Image.Image, mask: Image.Image):
        # 强制转换为RGB（处理灰度图、RGBA等情况）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if mask.mode != 'L':
            mask = mask.convert('L')

        # 训练阶段：在 resize 之前施加随机数据增强
        if self.aug_enabled:
            img, mask = self._augment(img, mask)

        w, h = img.size
        scale = self.img_size / max(w, h)
        nw, nh = int(w*scale), int(h*scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)
        pad_w, pad_h = self.img_size-nw, self.img_size-nh
        pl, pt = pad_w//2, pad_h//2
        canvas = Image.new('RGB', (self.img_size, self.img_size), (0,0,0))
        # destroyed_value_is_255=True 时，标签用 255 表示 destroyed，与 IGNORE_INDEX=255 冲突。
        # 使用哨兵值 128 初始化 padding 区域，在 __getitem__ 中再映射回 IGNORE_INDEX，
        # 以区分 padding（128）和真实 destroyed（255）。
        canvas_m_fill = 128 if self.destroyed_value_is_255 else IGNORE_INDEX
        canvas_m = Image.new('L', (self.img_size, self.img_size), canvas_m_fill)
        canvas.paste(img, (pl, pt))
        canvas_m.paste(mask, (pl, pt))
        
        # 转换为numpy array
        canvas_array = np.array(canvas, dtype=np.uint8)
        
        # Ensure the image tensor has three RGB channels.
        if len(canvas_array.shape) == 2:
            # 如果是2D数组（灰度），扩展为3通道
            canvas_array = np.stack([canvas_array, canvas_array, canvas_array], axis=-1)
        elif len(canvas_array.shape) == 3:
            if canvas_array.shape[2] == 1:
                # [H, W, 1] -> [H, W, 3]
                canvas_array = np.repeat(canvas_array, 3, axis=2)
            elif canvas_array.shape[2] == 4:
                # RGBA -> RGB
                canvas_array = canvas_array[:, :, :3]
            elif canvas_array.shape[2] != 3:
                # 其他情况，强制转为3通道
                raise ValueError(f"Unexpected image shape: {canvas_array.shape}")
        else:
            raise ValueError(f"Unexpected image dimensions: {canvas_array.shape}")
        
        # 现在canvas_array应该是 [H, W, 3]
        assert canvas_array.shape == (self.img_size, self.img_size, 3), \
            f"Expected shape ({self.img_size}, {self.img_size}, 3), got {canvas_array.shape}"
        
        # 转换为tensor: [H, W, 3] -> [3, H, W]
        x = torch.from_numpy(canvas_array).permute(2, 0, 1).float() / 255.
        
        # 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        
        y = torch.from_numpy(np.array(canvas_m)).long()
        return x, y

    def __getitem__(self, i):
        img_path, lab_path = self.items[i]
        img = Image.open(img_path)
        m = Image.open(lab_path)
        x, y = self._transform(img, m)
        
        # collapse to binary (1 if destroyed)
        if self.destroyed_value_is_255:
            # 模式1: 标签中255表示destroyed, 0表示background
            # _transform 中 padding 区域用哨兵值128初始化（避免与255=destroyed冲突），
            # 此处将128重新映射为 IGNORE_INDEX，确保 padding 区域不参与损失计算。
            y_bin = torch.where(y == 255, torch.ones_like(y), torch.zeros_like(y))
            y_bin = torch.where(y == 128, torch.full_like(y_bin, IGNORE_INDEX), y_bin)
        else:
            # 模式2: 使用destroyed_idx索引，255是ignore
            y_bin = torch.where(y == self.destroyed_idx, torch.ones_like(y), torch.zeros_like(y))
            y_bin = torch.where(y == IGNORE_INDEX, torch.full_like(y, IGNORE_INDEX), y_bin)
        
        return x, y_bin, os.path.basename(img_path)
