

import argparse
import os
import yaml
import time
import csv
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm
import math

from sam_atl_inject import load_sam_image_encoder, inject_atl_into_vit
from dataset_bin import DestroyedBinaryDataset
from losses import BinaryDiceLoss, bce_ignore, elementwise_bce, ohem_topk
from model_decoder import SAMMaskDecoderHead
from BinaryEvaluator import BinaryEvaluator
import numpy as np


class EarlyStopping:
    """早停机制：当监控指标不再改善时停止训练"""
    def __init__(self, patience=15, min_delta=0.001, mode='min', verbose=True):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善幅度
            mode: 'min' 表示指标越小越好，'max' 表示越大越好
            verbose: 是否打印详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, epoch, val_metric):
        score = val_metric
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"  [EarlyStopping] 初始化 best_score={score:.6f}")
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            if self.verbose:
                direction = "⬇️" if self.mode == 'min' else "⬆️"
                print(f"  [EarlyStopping] {direction} 改善: {self.best_score:.6f} → {score:.6f} "
                      f"(epoch {self.best_epoch} → {epoch})")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] 无改善 ({self.counter}/{self.patience}), "
                      f"当前={score:.6f}, 最佳={self.best_score:.6f} @ epoch {self.best_epoch}")
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"🛑 Early Stopping 触发！")
                print(f"最佳 {self.mode} score: {self.best_score:.6f} @ epoch {self.best_epoch}")
                print(f"已经 {self.counter} 个 epoch 无改善，停止训练")
                print(f"{'='*60}\n")
            
        return self.early_stop


def load_config(config_path):
    """从yaml文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def _resolve_input_path(path, config_dir):
    if path is None or os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(config_dir, path)),
        os.path.abspath(os.path.join(os.path.dirname(config_dir), path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[1]


def _resolve_output_path(path, config_dir):
    if path is None or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(config_dir, path))


def normalize_config_paths(config, config_path):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    data_cfg = config.get('data', {})
    for key in ('train_images_dir', 'train_labels_dir', 'val_images_dir', 'val_labels_dir'):
        if key in data_cfg:
            data_cfg[key] = _resolve_input_path(data_cfg[key], config_dir)

    sam_cfg = config.get('sam', {})
    if 'checkpoint' in sam_cfg:
        sam_cfg['checkpoint'] = _resolve_input_path(sam_cfg['checkpoint'], config_dir)

    out_cfg = config.get('output', {})
    if 'dir' in out_cfg:
        out_cfg['dir'] = _resolve_output_path(out_cfg['dir'], config_dir)
    return config


def config_to_args(config):
    """将yaml配置转换为argparse风格的对象"""
    class Args:
        pass
    
    args = Args()
    
    # 数据配置
    args.train_images_dir = config['data']['train_images_dir']
    args.train_labels_dir = config['data']['train_labels_dir']
    args.val_images_dir = config['data']['val_images_dir']
    args.val_labels_dir = config['data']['val_labels_dir']
    args.img_size = config['data']['img_size']
    args.destroyed_idx = config['data'].get('destroyed_idx', 3)
    args.destroyed_value_is_255 = config['data'].get('destroyed_value_is_255', False)
    args.require_post_suffix = config['data'].get('require_post_suffix', False)
    args.augmentation = config['data'].get('augmentation', {'enabled': False})
    
    # SAM配置
    args.sam_type = config['sam']['type']
    args.sam_checkpoint = config['sam']['checkpoint']
    
    # ATL配置
    args.atl_type = config['atl']['type']
    args.last_blocks = config['atl']['last_blocks']
    args.atl_rank = config['atl']['rank']
    args.lora_alpha = config['atl']['lora_alpha']
    args.atl_dropout = config['atl']['dropout']
    
    # SAM Decoder配置
    args.use_sam_decoder = config.get('decoder', {}).get('use_sam_decoder', True)
    args.num_learnable_prompts = config.get('decoder', {}).get('num_learnable_prompts', 1)
    args.use_iou_head = config.get('decoder', {}).get('use_iou_head', False)
    
    # 训练配置
    args.batch_size = config['training']['batch_size']
    args.workers = config['training']['workers']
    args.epochs = config['training']['epochs']
    args.lr = config['training']['lr']
    args.weight_decay = config['training']['weight_decay']
    args.no_amp = config['training']['no_amp']
    
    # 学习率调度器配置
    args.scheduler_config = config['training'].get('scheduler', {
        'type': 'none',
        'enabled': False
    })
    
    # Early Stopping配置
    args.early_stopping_config = config['training'].get('early_stopping', {
        'enabled': False,
        'patience': 15,
        'min_delta': 0.001,
        'monitor': 'val_loss'
    })
    
    # 梯度裁剪配置
    args.grad_clip = config['training'].get('grad_clip', 0.0)
    
    # 损失配置
    args.pos_weight = config['loss']['pos_weight']
    args.ohem_frac = config['loss']['ohem_frac']
    args.w_bce = config['loss']['w_bce']
    args.w_dice = config['loss']['w_dice']
    args.w_ohem = config['loss']['w_ohem']
    
    # 动态 OHEM 配置
    args.dynamic_ohem = config['loss'].get('dynamic_ohem', {'enabled': False})
    args.ohem_frac_min = config['loss'].get('ohem_frac_min', 0.05)
    
    # 自动 pos_weight 配置
    args.auto_pos_weight = config['loss'].get('auto_pos_weight', {'enabled': False})
    
    # 阈值搜索配置
    args.threshold_search = config.get('threshold_search', {
        'enabled': False,
        'interval': 5,
        'metric': 'f1'
    })
    
    # 输出配置
    args.out = config['output']['dir']
    
    return args


def create_scheduler(optimizer, config, num_epochs):
    """创建学习率调度器"""
    if not config.get('enabled', False) or config.get('type', 'none') == 'none':
        return None
    
    sched_type = config['type']
    
    if sched_type == 'cosine':
        T_max = config.get('T_max', num_epochs)
        eta_min = config.get('eta_min', 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        print(f"✓ 使用 CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
        
    elif sched_type == 'plateau':
        mode = config.get('mode', 'min')
        factor = config.get('factor', 0.5)
        patience = config.get('patience', 10)
        min_lr = config.get('min_lr', 1e-6)
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, 
                                     patience=patience, min_lr=min_lr)
        print(f"✓ 使用 ReduceLROnPlateau (mode={mode}, factor={factor}, patience={patience})")
        
    elif sched_type == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"✓ 使用 StepLR (step_size={step_size}, gamma={gamma})")
        
    else:
        print(f"⚠️  未知的scheduler类型: {sched_type}，不使用scheduler")
        return None
    
    return scheduler


def compute_dynamic_ohem_frac(epoch, total_epochs, frac_max, frac_min=0.05):
    """
    计算动态 OHEM 比例，使用余弦下降策略
    早期 epoch 更关注难负样本（frac 较高），后期逐渐降低
    
    Args:
        epoch: 当前 epoch（从1开始）
        total_epochs: 总 epoch 数
        frac_max: 最大 frac 值（训练开始时）
        frac_min: 最小 frac 值（训练结束时）
    
    Returns:
        float: 当前 epoch 的 frac 值
    """
    # 余弦下降：从 frac_max 降到 frac_min
    # progress: 0 -> 1 (epoch 1 -> epochs)
    progress = (epoch - 1) / max(1, total_epochs - 1)
    frac = frac_min + (frac_max - frac_min) * 0.5 * (1 + math.cos(math.pi * progress))
    return frac


def estimate_pos_weight_from_loader(loader, device, cfg, max_samples=None):
    """
    从 DataLoader 估计正负样本比例，自动计算 pos_weight
    
    Args:
        loader: DataLoader
        device: 设备
        cfg: 配置对象
        max_samples: 最大采样数量（用于加速，None 表示使用全部）
    
    Returns:
        float: pos_weight 值，限制在 [1, 8] 范围内
    """
    total_pos = 0
    total_neg = 0
    
    sample_count = 0
    for x, y, _ in loader:
        y = y.to(device)
        valid_mask = (y != 255)
        
        if valid_mask.sum() > 0:
            y_valid = y[valid_mask]
            total_pos += (y_valid == 1).sum().item()
            total_neg += (y_valid == 0).sum().item()
        
        sample_count += 1
        if max_samples is not None and sample_count >= max_samples:
            break
    
    if total_neg == 0:
        # 如果没有负样本，返回默认值
        pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 2.0
    else:
        # pos_weight = 负样本数 / 正样本数
        pos_weight = total_neg / max(1, total_pos)
    
    # 限制在 [1, 8] 范围内
    pos_weight = max(1.0, min(50.0, pos_weight))
    
    return pos_weight


def find_best_threshold_grid_search(enc, head, loader, device, cfg,
                                     metric='f1', thresholds=None):
    """
    在验证集上扫描阈值，找到使指定指标最大的二值化阈值。

    实现：一次 forward 收集 (prob, target) 直方图，然后在 CPU 上并行扫所有阈值。
    与外部 calibrate.py 共享同一份逻辑。

    Args:
        enc, head, loader, device, cfg
        metric: 'f1' / 'precision' / 'recall' / 'iou'
                  注意：'ap' 不依赖阈值，传入会自动改为 'f1'。
        thresholds: 阈值数组；None 时使用 0.05–0.95 step 0.05。

    Returns:
        (best_threshold, best_score)
    """
    if metric == 'ap':
        metric = 'f1'
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    num_bins = 1000
    pos_hist = np.zeros(num_bins, dtype=np.int64)
    neg_hist = np.zeros(num_bins, dtype=np.int64)

    enc.eval()
    head.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            feat = enc(x)
            output = head(feat, (x.shape[2], x.shape[3]))
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            prob = torch.sigmoid(logits).squeeze(1).clamp(0.0, 1.0).float()

            valid = (y != 255)
            prob_v = prob[valid].cpu().numpy()
            y_v = y[valid].cpu().numpy()
            if prob_v.size == 0:
                continue
            bins = np.minimum((prob_v * num_bins).astype(np.int64), num_bins - 1)
            if (y_v == 1).any():
                pos_hist += np.bincount(bins[y_v == 1], minlength=num_bins)
            if (y_v == 0).any():
                neg_hist += np.bincount(bins[y_v == 0], minlength=num_bins)

    if pos_hist.sum() == 0:
        return 0.5, 0.0

    # 对每个阈值算 P/R/F1/IoU
    cum_pos_rev = np.cumsum(pos_hist[::-1])[::-1]
    cum_neg_rev = np.cumsum(neg_hist[::-1])[::-1]
    cum_pos = np.concatenate([cum_pos_rev, [0]]).astype(np.float64)
    cum_neg = np.concatenate([cum_neg_rev, [0]]).astype(np.float64)
    total_pos = float(cum_pos[0])

    best_thr, best_score = 0.5, 0.0
    for thr in thresholds:
        idx = int(min(num_bins, max(0, round(float(thr) * num_bins))))
        tp = cum_pos[idx]
        fp = cum_neg[idx]
        fn = total_pos - tp
        if metric == 'precision':
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric == 'recall':
            score = tp / total_pos if total_pos > 0 else 0.0
        elif metric == 'iou':
            denom = tp + fp + fn
            score = tp / denom if denom > 0 else 0.0
        else:  # f1
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / total_pos if total_pos > 0 else 0.0
            score = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)

    return best_thr, best_score


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    os.makedirs(cfg.out, exist_ok=True)
    
    # 保存配置
    config_save_path = os.path.join(cfg.out, 'train_config.yaml')
    if hasattr(cfg, '_config_dict'):
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg._config_dict, f, allow_unicode=True)
        print(f"配置已保存至: {config_save_path}")

    # 数据加载
    print("加载数据集...")
    aug_cfg = getattr(cfg, 'augmentation', {'enabled': False})
    aug_enabled = aug_cfg.get('enabled', False) if isinstance(aug_cfg, dict) else False
    print(f"数据增强: {'✓ 启用' if aug_enabled else '✗ 未启用'}")
    if aug_enabled:
        print(f"  - 水平翻转: {aug_cfg.get('hflip', True)}")
        print(f"  - 垂直翻转: {aug_cfg.get('vflip', True)}")
        print(f"  - 随机旋转(90°): {aug_cfg.get('rotate90', True)}")
        cj = aug_cfg.get('color_jitter', {})
        if isinstance(cj, dict) and cj.get('enabled', True):
            print(f"  - 颜色抖动: brightness={cj.get('brightness', 0.25)}, "
                  f"contrast={cj.get('contrast', 0.25)}, "
                  f"saturation={cj.get('saturation', 0.1)}")

    train_set = DestroyedBinaryDataset(cfg.train_images_dir, cfg.train_labels_dir,
                                       cfg.img_size, cfg.destroyed_idx, True,
                                       destroyed_value_is_255=cfg.destroyed_value_is_255,
                                       require_post_suffix=cfg.require_post_suffix,
                                       augmentation=aug_cfg)
    val_set = DestroyedBinaryDataset(cfg.val_images_dir, cfg.val_labels_dir,
                                     cfg.img_size, cfg.destroyed_idx, False,
                                     destroyed_value_is_255=cfg.destroyed_value_is_255,
                                     require_post_suffix=cfg.require_post_suffix,
                                     augmentation=None)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, 
                             num_workers=cfg.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1, cfg.batch_size//2), shuffle=False, 
                           num_workers=cfg.workers, pin_memory=True)
    print(f"训练样本: {len(train_set)}, 验证样本: {len(val_set)}")

    # 加载SAM编码器 + ATL注入
    print(f"加载SAM模型 ({cfg.sam_type})...")
    enc = load_sam_image_encoder(cfg.sam_type, cfg.sam_checkpoint, device)
    
    print(f"注入ATL模块 (type={cfg.atl_type}, rank={cfg.atl_rank}, last_blocks={cfg.last_blocks})...")
    trainables = inject_atl_into_vit(enc, n_last_blocks=cfg.last_blocks, atl_type=cfg.atl_type,
                                     r=cfg.atl_rank, alpha=cfg.lora_alpha, p=cfg.atl_dropout,
                                     target_submodules=('attn', 'ffn'))
    print(f"可训练ATL模块数: {len(trainables)}")

    # 创建SAM Mask Decoder头
    print("="*60)
    print("🎯 使用SAM原生Mask Decoder（无需prompt）")
    print("="*60)
    head = SAMMaskDecoderHead(
        transformer_dim=256,  # SAM默认256
        num_learnable_prompts=cfg.num_learnable_prompts,
        use_iou_head=cfg.use_iou_head,
    ).to(device)
    print(f"✓ Mask Decoder已创建")
    print(f"  - 可学习prompt数量: {cfg.num_learnable_prompts}")
    print(f"  - 使用IoU预测头: {cfg.use_iou_head}")
    
    # 优化器
    params = list(head.parameters()) + [p for m in trainables for p in m.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    print(f"可训练参数总数: {total_params:,}")
    
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # 学习率调度器
    scheduler = create_scheduler(optim, cfg.scheduler_config, cfg.epochs)
    
    # Early Stopping
    early_stopping = None
    if cfg.early_stopping_config.get('enabled', False):
        es_patience = cfg.early_stopping_config.get('patience', 15)
        es_delta = cfg.early_stopping_config.get('min_delta', 0.001)
        es_monitor = cfg.early_stopping_config.get('monitor', 'val_loss')
        es_mode = 'min' if 'loss' in es_monitor else 'max'
        early_stopping = EarlyStopping(patience=es_patience, min_delta=es_delta, 
                                      mode=es_mode, verbose=True)
        print(f"✓ 启用 Early Stopping (monitor={es_monitor}, patience={es_patience}, mode={es_mode})")
    
    dice_loss = BinaryDiceLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=not cfg.no_amp)

    # 初始化训练历史记录
    history = {
        'epoch': [],
        'train_loss': [], 'train_bce': [], 'train_dice': [], 'train_ohem': [],
        'train_oa': [], 'train_precision': [], 'train_recall': [],
        'train_f1': [], 'train_iou': [], 'train_miou': [], 'train_ap': [],
        'val_loss': [], 'val_bce': [], 'val_dice': [], 'val_ohem': [],
        'val_oa': [], 'val_precision': [], 'val_recall': [], 
        'val_f1': [], 'val_iou': [], 'val_miou': [], 'val_ap': [],
        'lr': [], 'time': [], 'ohem_frac': [], 'pos_weight': [], 'best_threshold': []
    }
    
    # ⭐ 使用验证损失作为保存指标（best.pt）；
    # 同时维护一份基于 F1 的最佳模型（best_f1.pt），方便部署不依赖 loss 选模。
    best_val_loss = float('inf')
    best_val_f1 = -1.0
    
    # 检查功能启用状态
    dynamic_ohem_enabled = cfg.dynamic_ohem.get('enabled', False) if isinstance(cfg.dynamic_ohem, dict) else False
    auto_pos_weight_enabled = cfg.auto_pos_weight.get('enabled', False) if isinstance(cfg.auto_pos_weight, dict) else False
    threshold_search_enabled = cfg.threshold_search.get('enabled', False)
    
    print("\n" + "="*60)
    print("开始训练循环")
    print("="*60)
    print("⭐ 模型保存策略: 基于验证损失 (Validation Loss)")
    print("   当验证损失降低时保存模型")
    if early_stopping:
        print(f"⏱️  早停监控: {cfg.early_stopping_config['monitor']}")
    if scheduler:
        print(f"📊 学习率调度: {cfg.scheduler_config['type']}")
    if dynamic_ohem_enabled:
        print(f"📉 动态 OHEM: 启用 (frac: {cfg.ohem_frac:.3f} → {cfg.ohem_frac_min:.3f})")
    else:
        print(f"📉 OHEM: 固定 frac={cfg.ohem_frac:.3f}")
    if auto_pos_weight_enabled:
        print(f"⚖️  自动 pos_weight: 启用 (每个 epoch 动态估计)")
    else:
        print(f"⚖️  pos_weight: 固定={cfg.pos_weight:.3f}")
    if threshold_search_enabled:
        interval = cfg.threshold_search.get('interval', 5)
        metric = cfg.threshold_search.get('metric', 'f1')
        print(f"🎯 阈值搜索: 启用 (每 {interval} 个 epoch, 优化 {metric})")
    print("="*60 + "\n")
    
    # 当前使用的 pos_weight 和最佳阈值
    current_pos_weight = cfg.pos_weight
    best_threshold = 0.5
    
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        
        # ==================== Epoch 初始化 ====================
        # 1. 计算动态 OHEM frac
        if dynamic_ohem_enabled:
            current_ohem_frac = compute_dynamic_ohem_frac(epoch, cfg.epochs, cfg.ohem_frac, cfg.ohem_frac_min)
        else:
            current_ohem_frac = cfg.ohem_frac
        
        # 2. 自动估计 pos_weight（每个 epoch 开始时）
        if auto_pos_weight_enabled:
            print(f"  [Epoch {epoch}] 估计 pos_weight...", end=' ', flush=True)
            current_pos_weight = estimate_pos_weight_from_loader(train_loader, device, cfg, max_samples=50)
            print(f"pos_weight={current_pos_weight:.3f}")
        
        # 3. 阈值搜索（每 N 个 epoch）
        if threshold_search_enabled:
            interval = cfg.threshold_search.get('interval', 5)
            metric = cfg.threshold_search.get('metric', 'f1')
            if epoch % interval == 0 or epoch == 1:
                print(f"  [Epoch {epoch}] 执行阈值网格搜索 (优化 {metric})...", end=' ', flush=True)
                best_threshold, best_score = find_best_threshold_grid_search(
                    enc, head, val_loader, device, cfg, metric=metric
                )
                print(f"最佳阈值={best_threshold:.3f}, {metric}={best_score:.4f}")
        
        # ==================== 训练阶段 ====================
        enc.eval()
        head.train()
        for m in trainables:
            m.train()
        
        running = {k: 0. for k in ['loss', 'bce', 'dice', 'ohem']}
        train_eval_metric = BinaryEvaluator(ignore_index=255)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.epochs} [Train]', 
                   ncols=120, leave=True)
        
        for batch_idx, (x, y, _) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=not cfg.no_amp):
                feat = enc(x)  # [B, 256, H, W]
                output = head(feat, (x.shape[2], x.shape[3]))  # [B, 1, H, W] 或 (logits, iou_preds)
                # 处理 head 可能返回元组的情况（当 use_iou_head=True 时）
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                bce = bce_ignore(logits, y, pos_weight=current_pos_weight)
                dice = dice_loss(logits, y)
                eloss = elementwise_bce(logits, y, pos_weight=current_pos_weight)
                ohem = ohem_topk(eloss, frac=current_ohem_frac)
                loss = cfg.w_bce * bce + cfg.w_dice * dice + cfg.w_ohem * ohem
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.grad_clip)
            
            scaler.step(optim)
            scaler.update()
            
            running['loss'] += float(loss)
            running['bce'] += float(bce)
            running['dice'] += float(dice)
            running['ohem'] += float(ohem)
            
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                # 训练统计阈值与验证保持一致：开启阈值搜索时用 best_threshold，否则 0.5
                stat_thr = best_threshold if threshold_search_enabled else 0.5
                pred = (prob > stat_thr).long().squeeze(1)
                for b in range(pred.shape[0]):
                    train_eval_metric.add_batch(y[b], pred[b])
            
            pbar.set_postfix({
                'loss': f"{float(loss):.4f}",
                'bce': f"{float(bce):.4f}",
                'dice': f"{float(dice):.4f}"
            })
        
        niter = len(train_loader)
        train_metrics = {
            'loss': running['loss'] / niter,
            'bce': running['bce'] / niter,
            'dice': running['dice'] / niter,
            'ohem': running['ohem'] / niter,
            'oa': train_eval_metric.Accuracy(),
            'precision': train_eval_metric.Precision(class_idx=1),
            'recall': train_eval_metric.Recall(class_idx=1),
            'f1': train_eval_metric.F1Score(class_idx=1),
            'iou': train_eval_metric.IoU(class_idx=1),
            'miou': train_eval_metric.mIoU(),
            'ap': train_eval_metric.AP(class_idx=1)
        }

        # ==================== 验证阶段 ====================
        # 使用最佳阈值进行验证（如果启用了阈值搜索）
        val_threshold = best_threshold if threshold_search_enabled else 0.5
        val_metrics = validate_with_metrics(enc, head, val_loader, device, cfg, 
                                           thr=val_threshold,
                                           pos_weight=current_pos_weight,
                                           ohem_frac=current_ohem_frac)
        
        epoch_time = time.time() - epoch_start
        
        # 获取当前学习率
        current_lr = optim.param_groups[0]['lr']
        
        # 记录历史
        history['epoch'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_bce'].append(train_metrics['bce'])
        history['train_dice'].append(train_metrics['dice'])
        history['train_ohem'].append(train_metrics['ohem'])
        history['train_oa'].append(train_metrics['oa'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_iou'].append(train_metrics['iou'])
        history['train_miou'].append(train_metrics['miou'])
        history['train_ap'].append(train_metrics['ap'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_bce'].append(val_metrics['bce'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_ohem'].append(val_metrics['ohem'])
        history['val_oa'].append(val_metrics['oa'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_miou'].append(val_metrics['miou'])
        history['val_ap'].append(val_metrics['ap'])
        history['lr'].append(current_lr)
        history['time'].append(epoch_time)
        history['ohem_frac'].append(current_ohem_frac)
        history['pos_weight'].append(current_pos_weight)
        history['best_threshold'].append(best_threshold)
        
        # 打印训练结果
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.epochs} 总结")
        print(f"{'='*60}")
        print(f"训练: Loss={train_metrics['loss']:.4f} | "
              f"BCE={train_metrics['bce']:.4f} | "
              f"Dice={train_metrics['dice']:.4f} | "
              f"OHEM={train_metrics['ohem']:.4f}")
        print(f"      OA={train_metrics['oa']:.4f} | "
              f"Precision={train_metrics['precision']:.4f} | "
              f"Recall={train_metrics['recall']:.4f}")
        print(f"      F1={train_metrics['f1']:.4f} | "
              f"IoU={train_metrics['iou']:.4f} | "
              f"AP={train_metrics['ap']:.4f}")
        print(f"验证: Loss={val_metrics['loss']:.4f} ⭐ | "
              f"OA={val_metrics['oa']:.4f} | "
              f"mIoU={val_metrics['miou']:.4f}")
        print(f"      Precision={val_metrics['precision']:.4f} | "
              f"Recall={val_metrics['recall']:.4f} | "
              f"AP={val_metrics['ap']:.4f}")
        print(f"      F1={val_metrics['f1']:.4f} | "
              f"IoU={val_metrics['iou']:.4f}")
        print(f"学习率: {current_lr:.2e} | 时间: {epoch_time:.1f}s")
        if dynamic_ohem_enabled:
            print(f"OHEM frac: {current_ohem_frac:.3f}", end='')
        if auto_pos_weight_enabled:
            print(f" | pos_weight: {current_pos_weight:.3f}", end='')
        if threshold_search_enabled:
            print(f" | 阈值: {val_threshold:.3f}", end='')
        if dynamic_ohem_enabled or auto_pos_weight_enabled or threshold_search_enabled:
            print()
        
        # 保存检查点
        ckpt = {
            'epoch': epoch,
            'sam_type': cfg.sam_type,
            'cfg': vars(cfg),
            'head': head.state_dict(),
            'enc': enc.state_dict(),
            'history': history
        }
        torch.save(ckpt, os.path.join(cfg.out, 'last.pt'))
        
        # ⭐ 保存最佳模型 - 基于验证损失
        is_best = False
        reason = ""
        
        if val_metrics['loss'] <= best_val_loss:
            is_best = True
            reason = f"Val Loss↓ {best_val_loss:.4f}→{val_metrics['loss']:.4f}"
            best_val_loss = val_metrics['loss']
        
        if is_best:
            torch.save(ckpt, os.path.join(cfg.out, 'best.pt'))
            print(f"✓ 新的最佳模型! {reason}")
            print(f"  当前最佳验证损失: {best_val_loss:.4f}")
            print(f"  对应指标: AP={val_metrics['ap']:.4f}, F1={val_metrics['f1']:.4f}, IoU={val_metrics['iou']:.4f}")

        # 同时按 F1 保存一份（best_f1.pt）
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(ckpt, os.path.join(cfg.out, 'best_f1.pt'))
            print(f"✓ best_f1.pt 更新: F1={best_val_f1:.4f} (thr={best_threshold:.3f})")
        
        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Early Stopping检查
        if early_stopping is not None:
            monitor_metric = cfg.early_stopping_config['monitor']
            if monitor_metric == 'val_loss':
                metric_value = val_metrics['loss']
            elif monitor_metric == 'val_ap':
                metric_value = val_metrics['ap']
            elif monitor_metric == 'val_f1':
                metric_value = val_metrics['f1']
            else:
                metric_value = val_metrics['loss']
            
            if early_stopping(epoch, metric_value):
                print(f"✓ 早停触发，训练结束于 epoch {epoch}")
                break
        
        print(f"{'='*60}\n")
        
        # 每个epoch保存CSV
        save_training_history(history, cfg.out)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"{'='*60}")
    print(f"最终epoch: {epoch}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {cfg.out}")
    print(f"训练历史保存在: {os.path.join(cfg.out, 'training_history.csv')}")


def validate_with_metrics(enc, head, loader, device, cfg, thr=0.5, 
                          pos_weight=None, ohem_frac=None):
    """
    详细验证函数，返回所有评估指标
    
    Args:
        enc: 编码器
        head: 解码器头
        loader: 验证 DataLoader
        device: 设备
        cfg: 配置对象
        thr: 阈值
        pos_weight: pos_weight 值，None 时使用 cfg.pos_weight
        ohem_frac: ohem_frac 值，None 时使用 cfg.ohem_frac
    """
    enc.eval()
    head.eval()
    
    eval_metric = BinaryEvaluator(ignore_index=255)
    dice_loss_fn = BinaryDiceLoss()
    running_loss = {'loss': 0., 'bce': 0., 'dice': 0., 'ohem': 0.}
    
    # 使用传入的参数或默认值
    use_pos_weight = pos_weight if pos_weight is not None else cfg.pos_weight
    use_ohem_frac = ohem_frac if ohem_frac is not None else cfg.ohem_frac
    
    pbar = tqdm(loader, desc='Validating', ncols=120, leave=False)
    
    with torch.no_grad():
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)
            
            feat = enc(x)
            output = head(feat, (x.shape[2], x.shape[3]))
            # 处理 head 可能返回元组的情况（当 use_iou_head=True 时）
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # 使用与训练相同的loss计算
            bce = bce_ignore(logits, y, pos_weight=use_pos_weight)
            dice = dice_loss_fn(logits, y)
            eloss = elementwise_bce(logits, y, pos_weight=use_pos_weight)
            ohem = ohem_topk(eloss, frac=use_ohem_frac)
            loss = cfg.w_bce * bce + cfg.w_dice * dice + cfg.w_ohem * ohem
            
            running_loss['loss'] += float(loss)
            running_loss['bce'] += float(bce)
            running_loss['dice'] += float(dice)
            running_loss['ohem'] += float(ohem)
            
            prob = torch.sigmoid(logits)
            pred = (prob > thr).long().squeeze(1)

            for b in range(pred.shape[0]):
                eval_metric.add_batch(y[b], pred[b])
                # 同时累计概率直方图，使 eval_metric.AP() 返回真正的 PR 曲线下面积
                eval_metric.add_batch_prob(y[b], prob[b, 0])
    
    niter = len(loader)
    
    metrics = {
        'loss': running_loss['loss'] / niter,
        'bce': running_loss['bce'] / niter,
        'dice': running_loss['dice'] / niter,
        'ohem': running_loss['ohem'] / niter,
        'oa': eval_metric.Accuracy(),
        'precision': eval_metric.Precision(class_idx=1),
        'recall': eval_metric.Recall(class_idx=1),
        'f1': eval_metric.F1Score(class_idx=1),
        'iou': eval_metric.IoU(class_idx=1),
        'miou': eval_metric.mIoU(),
        'ap': eval_metric.AP(class_idx=1)
    }
    
    return metrics


def save_training_history(history, output_dir):
    """保存训练历史到CSV文件"""
    csv_path = os.path.join(output_dir, 'training_history.csv')
    
    rows = []
    for i in range(len(history['epoch'])):
        row = {
            'epoch': history['epoch'][i],
            'train_loss': history['train_loss'][i],
            'train_bce': history['train_bce'][i],
            'train_dice': history['train_dice'][i],
            'train_ohem': history['train_ohem'][i],
            'train_oa': history['train_oa'][i],
            'train_precision': history['train_precision'][i],
            'train_recall': history['train_recall'][i],
            'train_ap': history['train_ap'][i],
            'train_f1': history['train_f1'][i],
            'train_iou': history['train_iou'][i],
            'train_miou': history['train_miou'][i],
            'val_loss': history['val_loss'][i],
            'val_bce': history['val_bce'][i],
            'val_dice': history['val_dice'][i],
            'val_ohem': history['val_ohem'][i],
            'val_oa': history['val_oa'][i],
            'val_precision': history['val_precision'][i],
            'val_recall': history['val_recall'][i],
            'val_ap': history['val_ap'][i],
            'val_f1': history['val_f1'][i],
            'val_iou': history['val_iou'][i],
            'val_miou': history['val_miou'][i],
            'lr': history['lr'][i],
            'time(s)': history['time'][i],
            'ohem_frac': history.get('ohem_frac', [0.0] * len(history['epoch']))[i],
            'pos_weight': history.get('pos_weight', [0.0] * len(history['epoch']))[i],
            'best_threshold': history.get('best_threshold', [0.5] * len(history['epoch']))[i]
        }
        rows.append(row)
    
    if rows:
        fieldnames = rows[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='SAM-ATL训练脚本')
    parser.add_argument('--config', type=str, default='SAM-ATL/config_train.yaml',
                       help='YAML配置文件路径')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        alt_config = os.path.join(os.path.dirname(__file__), os.path.basename(args.config))
        if os.path.exists(alt_config):
            args.config = alt_config

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print("请先创建配置文件或检查路径是否正确")
        return
    
    print(f"加载配置文件: {args.config}")
    config_dict = normalize_config_paths(load_config(args.config), args.config)
    cfg = config_to_args(config_dict)
    cfg._config_dict = config_dict
    
    train(cfg)


if __name__ == '__main__':
    main()

