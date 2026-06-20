

import argparse
import csv
import json
import os
import sys
import time
from statistics import mean

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from atl_modules import LoRALinear
from BinaryEvaluator import BinaryEvaluator
from dataset_bin import DestroyedBinaryDataset
from losses import BinaryDiceLoss, bce_ignore, elementwise_bce, ohem_topk
from model_decoder import SAMMaskDecoderHead
from model_loader import head_logits
from sam_atl_inject import inject_atl_into_vit, load_sam_image_encoder
from train import (
    EarlyStopping,
    compute_dynamic_ohem_frac,
    config_to_args,
    create_scheduler,
    estimate_pos_weight_from_loader,
    find_best_threshold_grid_search,
    load_config,
    normalize_config_paths,
    save_training_history,
    validate_with_metrics,
)


def bytes_to_mb(value):
    if value is None:
        return None
    return float(value) / (1024.0 ** 2)


def human_number(value, suffix=""):
    if value is None:
        return "unavailable"
    value = float(value)
    units = ["", "K", "M", "G", "T", "P"]
    idx = 0
    while abs(value) >= 1000.0 and idx < len(units) - 1:
        value /= 1000.0
        idx += 1
    return f"{value:.3f} {units[idx]}{suffix}".strip()


def count_unique_parameters(modules, trainable_only=False):
    seen = set()
    total = 0
    for module in modules:
        for param in module.parameters():
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            if trainable_only and not param.requires_grad:
                continue
            total += param.numel()
    return int(total)


def _linear_macs(module, vectors):
    if isinstance(module, LoRALinear):
        base = module.in_features * module.out_features
        lora = module.in_features * module.r + module.r * module.out_features
        return int(vectors * (base + lora))
    if isinstance(module, nn.Linear):
        return int(vectors * module.in_features * module.out_features)
    return 0


def _add_macs(module, macs):
    module.total_ops += torch.DoubleTensor([int(macs)])


def count_lora_linear_macs(module, inputs, output):
    x = inputs[0]
    vectors = x.numel() // max(1, x.shape[-1])
    _add_macs(module, _linear_macs(module, vectors))


def count_sam_image_attention_macs(module, inputs, output):
    x = inputs[0]
    batch, height, width, channels = x.shape
    tokens = height * width
    vectors = batch * tokens
    head_dim = channels // module.num_heads
    macs = 0
    macs += _linear_macs(module.qkv, vectors)
    macs += batch * module.num_heads * tokens * tokens * head_dim
    macs += batch * module.num_heads * tokens * tokens * head_dim
    macs += _linear_macs(module.proj, vectors)
    _add_macs(module, macs)


def count_sam_decoder_attention_macs(module, inputs, output):
    if len(inputs) != 3:
        return
    q, k, v = inputs
    batch, q_tokens, q_channels = q.shape
    k_tokens = k.shape[1]
    v_tokens = v.shape[1]
    head_dim = module.internal_dim // module.num_heads
    macs = 0
    macs += batch * q_tokens * q_channels * module.internal_dim
    macs += batch * k_tokens * k.shape[-1] * module.internal_dim
    macs += batch * v_tokens * v.shape[-1] * module.internal_dim
    macs += batch * module.num_heads * q_tokens * k_tokens * head_dim
    macs += batch * module.num_heads * q_tokens * k_tokens * head_dim
    macs += batch * q_tokens * module.internal_dim * module.embedding_dim
    _add_macs(module, macs)


def get_thop_custom_ops():
    custom_ops = {LoRALinear: count_lora_linear_macs}
    try:
        from segment_anything.modeling.image_encoder import Attention as ImageEncoderAttention

        custom_ops[ImageEncoderAttention] = count_sam_image_attention_macs
    except Exception:
        pass
    return custom_ops


class ForwardOnlyModel(nn.Module):
    def __init__(self, enc, head):
        super().__init__()
        self.enc = enc
        self.head = head

    def forward(self, x):
        feat = self.enc(x)
        return head_logits(self.head(feat, (x.shape[2], x.shape[3])))


def cleanup_thop_buffers(module):
    for child in module.modules():
        child._buffers.pop("total_ops", None)
        child._buffers.pop("total_params", None)


def estimate_decoder_functional_macs(head, img_size):
    """MACs for SAM decoder functional matmuls not visible to THOP hooks."""
    feature_tokens = max(1, int(img_size) // 16) ** 2
    prompt_tokens = int(getattr(head, "num_learnable_prompts", 1))
    mask_decoder = head.mask_decoder
    output_tokens = 1 + int(mask_decoder.num_mask_tokens)
    query_tokens = output_tokens + prompt_tokens

    transformer = mask_decoder.transformer
    num_heads = int(transformer.num_heads)
    embedding_dim = int(transformer.embedding_dim)

    def attention_matmul_macs(q_tokens, k_tokens, internal_dim):
        head_dim = internal_dim // num_heads
        return 2 * num_heads * q_tokens * k_tokens * head_dim

    macs = 0
    for layer in transformer.layers:
        macs += attention_matmul_macs(query_tokens, query_tokens, embedding_dim)
        macs += attention_matmul_macs(
            query_tokens,
            feature_tokens,
            layer.cross_attn_token_to_image.internal_dim,
        )
        macs += attention_matmul_macs(
            feature_tokens,
            query_tokens,
            layer.cross_attn_image_to_token.internal_dim,
        )

    macs += attention_matmul_macs(
        query_tokens,
        feature_tokens,
        transformer.final_attn_token_to_image.internal_dim,
    )

    upscaled_tokens = feature_tokens * 16
    hypernet_channels = embedding_dim // 8
    macs += int(mask_decoder.num_mask_tokens) * hypernet_channels * upscaled_tokens
    return int(macs)


def snapshot_forward_hooks(module):
    return [(child, child._forward_hooks.copy()) for child in module.modules()]


def restore_forward_hooks(snapshot):
    for child, hooks in snapshot:
        child._forward_hooks = hooks


def create_torch_flops_profiler(device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
    )


def profiler_total_flops(prof):
    return int(sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages()))


def profile_forward_flops_torch_profiler(enc, head, device, img_size):
    input_shape = [1, 3, int(img_size), int(img_size)]
    info = {
        "available": False,
        "input_shape": input_shape,
        "macs": None,
        "flops_2mac": None,
        "flops": None,
        "method": "torch.profiler",
        "note": (
            "Forward pass only. PyTorch profiler reports FLOPs for supported aten matmul/conv ops. "
            "Some unsupported elementwise/interpolation ops may be omitted."
        ),
        "error": None,
        "decoder_functional_macs_estimate": None,
    }
    model = ForwardOnlyModel(enc, head)
    was_enc_training = enc.training
    was_head_training = head.training
    try:
        model.eval()
        dummy = torch.zeros(*input_shape, device=device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

        with create_torch_flops_profiler(device) as prof:
            with torch.inference_mode():
                model(dummy)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        flops = profiler_total_flops(prof)
        if flops <= 0:
            raise RuntimeError("torch.profiler returned 0 FLOPs for supported ops")
        info["available"] = True
        info["flops"] = flops
        info["flops_2mac"] = flops
    except Exception as exc:
        info["error"] = str(exc)
    finally:
        enc.train(was_enc_training)
        head.train(was_head_training)
    return info


def profile_forward_flops(enc, head, device, img_size, skip=False):
    input_shape = [1, 3, int(img_size), int(img_size)]
    info = {
        "available": False,
        "input_shape": input_shape,
        "macs": None,
        "flops_2mac": None,
        "flops": None,
        "method": "THOP + formula fallback",
        "note": (
            "Forward pass only. THOP reports MACs; flops_2mac uses 1 MAC = 2 FLOPs. "
            "SAM decoder functional attention/mask matmuls are added by formula; "
            "some unsupported elementwise/interpolation ops may still be omitted."
        ),
        "error": None,
        "decoder_functional_macs_estimate": None,
    }
    if skip:
        info["error"] = "Skipped by --skip-flops."
        return info

    profiler_info = profile_forward_flops_torch_profiler(enc, head, device, img_size)
    if profiler_info["available"]:
        return profiler_info

    try:
        from thop import profile
    except Exception as exc:
        info["error"] = (
            f"THOP is not installed or cannot be imported: {exc}. "
            "Install it in the same Python environment with: pip install thop. "
            f"torch.profiler fallback also failed: {profiler_info['error']}"
        )
        return info

    model = ForwardOnlyModel(enc, head)
    was_enc_training = enc.training
    was_head_training = head.training
    forward_hooks_snapshot = snapshot_forward_hooks(model)
    try:
        cleanup_thop_buffers(model)
        model.eval()
        dummy = torch.zeros(*input_shape, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        macs, _ = profile(
            model,
            inputs=(dummy,),
            custom_ops=get_thop_custom_ops(),
            verbose=False,
        )
        decoder_extra_macs = estimate_decoder_functional_macs(head, img_size)
        macs = int(macs) + decoder_extra_macs
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        info["available"] = True
        info["macs"] = int(macs)
        info["flops_2mac"] = int(macs * 2)
        info["decoder_functional_macs_estimate"] = int(decoder_extra_macs)
    except Exception as exc:
        info["error"] = f"{exc}; torch.profiler fallback also failed: {profiler_info['error']}"
    finally:
        restore_forward_hooks(forward_hooks_snapshot)
        cleanup_thop_buffers(model)
        enc.train(was_enc_training)
        head.train(was_head_training)
    return info


class GpuMetricTracker:
    def __init__(self, device):
        self.device = device
        self.enabled = device.type == "cuda" and torch.cuda.is_available()
        self.iter_peak_bytes = []
        self.sample_allocated_bytes = []

    def start_iter(self):
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def end_iter(self):
        if not self.enabled:
            return None, None
        torch.cuda.synchronize(self.device)
        current = torch.cuda.memory_allocated(self.device)
        peak = max(torch.cuda.max_memory_allocated(self.device), current)
        self.iter_peak_bytes.append(int(peak))
        self.sample_allocated_bytes.append(int(current))
        return peak, current

    def extend(self, other):
        self.iter_peak_bytes.extend(other.iter_peak_bytes)
        self.sample_allocated_bytes.extend(other.sample_allocated_bytes)

    def summary(self):
        if not self.enabled or not self.iter_peak_bytes:
            return {
                "available": False,
                "samples": 0,
                "peak_allocated_mb": None,
                "avg_iter_peak_allocated_mb": None,
                "avg_sampled_allocated_mb": None,
            }
        return {
            "available": True,
            "samples": len(self.iter_peak_bytes),
            "peak_allocated_mb": bytes_to_mb(max(self.iter_peak_bytes)),
            "avg_iter_peak_allocated_mb": bytes_to_mb(mean(self.iter_peak_bytes)),
            "avg_sampled_allocated_mb": bytes_to_mb(mean(self.sample_allocated_bytes)),
        }


def gpu_summary_text(summary):
    if not summary["available"]:
        return "GPU memory: unavailable (CUDA not in use)"
    return (
        f"Peak GPU memory: {summary['peak_allocated_mb']:.1f} MB | "
        f"Avg per-iteration peak: {summary['avg_iter_peak_allocated_mb']:.1f} MB | "
        f"Avg sampled allocated: {summary['avg_sampled_allocated_mb']:.1f} MB"
    )


def init_resource_history():
    return {
        "epoch": [],
        "train_loop_time_s": [],
        "epoch_total_time_s": [],
        "images_seen": [],
        "avg_iteration_time_s": [],
        "train_time_per_image_s": [],
        "epoch_total_time_per_training_image_s": [],
        "gpu_peak_allocated_mb": [],
        "gpu_avg_iter_peak_allocated_mb": [],
        "gpu_avg_sampled_allocated_mb": [],
        "train_step_flops_per_batch_mean": [],
        "train_step_flops_per_image_mean": [],
        "estimated_train_optimizer_flops_epoch": [],
    }


def append_resource_history(
    history,
    epoch,
    train_loop_time,
    epoch_time,
    images_seen,
    iter_times,
    gpu_summary,
    train_set_len,
    train_flops_samples,
):
    avg_iter_time = mean(iter_times) if iter_times else 0.0
    train_time_per_image = train_loop_time / max(1, images_seen)
    epoch_total_time_per_image = epoch_time / max(1, train_set_len)
    if train_flops_samples:
        train_step_flops_per_batch = mean(sample["flops"] for sample in train_flops_samples)
        train_step_flops_per_image = mean(sample["flops_per_image"] for sample in train_flops_samples)
        estimated_train_optimizer_flops_epoch = train_step_flops_per_image * images_seen
    else:
        train_step_flops_per_batch = None
        train_step_flops_per_image = None
        estimated_train_optimizer_flops_epoch = None
    history["epoch"].append(epoch)
    history["train_loop_time_s"].append(train_loop_time)
    history["epoch_total_time_s"].append(epoch_time)
    history["images_seen"].append(images_seen)
    history["avg_iteration_time_s"].append(avg_iter_time)
    history["train_time_per_image_s"].append(train_time_per_image)
    history["epoch_total_time_per_training_image_s"].append(epoch_total_time_per_image)
    history["gpu_peak_allocated_mb"].append(gpu_summary["peak_allocated_mb"])
    history["gpu_avg_iter_peak_allocated_mb"].append(gpu_summary["avg_iter_peak_allocated_mb"])
    history["gpu_avg_sampled_allocated_mb"].append(gpu_summary["avg_sampled_allocated_mb"])
    history["train_step_flops_per_batch_mean"].append(train_step_flops_per_batch)
    history["train_step_flops_per_image_mean"].append(train_step_flops_per_image)
    history["estimated_train_optimizer_flops_epoch"].append(estimated_train_optimizer_flops_epoch)


def save_resource_history(history, output_dir):
    csv_path = os.path.join(output_dir, "training_resource_history.csv")
    rows = []
    for idx in range(len(history["epoch"])):
        rows.append({key: history[key][idx] for key in history})
    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_resource_summary(summary, output_dir):
    path = os.path.join(output_dir, "resource_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_forward_profile(flops_info):
    h = flops_info["input_shape"][2]
    w = flops_info["input_shape"][3]
    print(f"FLOPs profile input size: {h} x {w}, batch size = 1")
    if flops_info["available"]:
        print(f"FLOPs method: {flops_info.get('method', 'unknown')}")
        if flops_info.get("macs") is not None:
            print(f"Forward MACs: {human_number(flops_info['macs'], 'MACs')}")
            print(f"Forward FLOPs (2 x MACs): {human_number(flops_info['flops_2mac'], 'FLOPs')}")
        else:
            print(f"Forward FLOPs: {human_number(flops_info.get('flops_2mac'), 'FLOPs')}")
        print("FLOPs note: forward pass only; this is not training FLOPs.")
    else:
        print(f"FLOPs unavailable (non-fatal): {flops_info['error']}")


def train_with_details(cfg, flops_img_size=None, skip_flops=False, profile_train_flops_batches=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not cfg.no_amp
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    print(f"Using device: {device}")
    os.makedirs(cfg.out, exist_ok=True)

    config_save_path = os.path.join(cfg.out, "train_config.yaml")
    if hasattr(cfg, "_config_dict"):
        with open(config_save_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg._config_dict, f, allow_unicode=True)
        print(f"Config saved to: {config_save_path}")

    print("Loading datasets...")
    aug_cfg = getattr(cfg, "augmentation", {"enabled": False})
    aug_enabled = aug_cfg.get("enabled", False) if isinstance(aug_cfg, dict) else False
    print(f"Data augmentation: {'enabled' if aug_enabled else 'disabled'}")

    train_set = DestroyedBinaryDataset(
        cfg.train_images_dir,
        cfg.train_labels_dir,
        cfg.img_size,
        cfg.destroyed_idx,
        True,
        destroyed_value_is_255=cfg.destroyed_value_is_255,
        require_post_suffix=cfg.require_post_suffix,
        augmentation=aug_cfg,
    )
    val_set = DestroyedBinaryDataset(
        cfg.val_images_dir,
        cfg.val_labels_dir,
        cfg.img_size,
        cfg.destroyed_idx,
        False,
        destroyed_value_is_255=cfg.destroyed_value_is_255,
        require_post_suffix=cfg.require_post_suffix,
        augmentation=None,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
    )
    if len(train_loader) == 0:
        raise RuntimeError("No training iterations. Reduce batch_size or disable drop_last.")
    print(f"Train samples: {len(train_set)}, val samples: {len(val_set)}")

    print(f"Loading SAM image encoder ({cfg.sam_type})...")
    enc = load_sam_image_encoder(cfg.sam_type, cfg.sam_checkpoint, device)

    print(
        f"Injecting ATL modules: type={cfg.atl_type}, rank={cfg.atl_rank}, "
        f"last_blocks={cfg.last_blocks}"
    )
    trainables = inject_atl_into_vit(
        enc,
        n_last_blocks=cfg.last_blocks,
        atl_type=cfg.atl_type,
        r=cfg.atl_rank,
        alpha=cfg.lora_alpha,
        p=cfg.atl_dropout,
        target_submodules=("attn", "ffn"),
    )
    print(f"Trainable ATL modules: {len(trainables)}")

    head = SAMMaskDecoderHead(
        transformer_dim=256,
        num_learnable_prompts=cfg.num_learnable_prompts,
        use_iou_head=cfg.use_iou_head,
    ).to(device)
    print(
        f"Mask decoder created: prompts={cfg.num_learnable_prompts}, "
        f"use_iou_head={cfg.use_iou_head}"
    )

    params = list(head.parameters()) + [
        p for module in trainables for p in module.parameters() if p.requires_grad
    ]
    trainable_params = count_unique_parameters([enc, head], trainable_only=True)
    total_params = count_unique_parameters([enc, head], trainable_only=False)
    optimizer_param_count = sum(p.numel() for p in params)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Optimizer parameters: {optimizer_param_count:,}")
    print(f"Total model parameters: {total_params:,}")

    flops_info = profile_forward_flops(
        enc,
        head,
        device,
        flops_img_size or cfg.img_size,
        skip=skip_flops,
    )
    print_forward_profile(flops_info)

    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = create_scheduler(optim, cfg.scheduler_config, cfg.epochs)

    early_stopping = None
    if cfg.early_stopping_config.get("enabled", False):
        es_patience = cfg.early_stopping_config.get("patience", 15)
        es_delta = cfg.early_stopping_config.get("min_delta", 0.001)
        es_monitor = cfg.early_stopping_config.get("monitor", "val_loss")
        es_mode = "min" if "loss" in es_monitor else "max"
        early_stopping = EarlyStopping(
            patience=es_patience,
            min_delta=es_delta,
            mode=es_mode,
            verbose=False,
        )
        print(f"Early stopping enabled: monitor={es_monitor}, patience={es_patience}, mode={es_mode}")

    dice_loss = BinaryDiceLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_bce": [],
        "train_dice": [],
        "train_ohem": [],
        "train_oa": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_iou": [],
        "train_miou": [],
        "train_ap": [],
        "val_loss": [],
        "val_bce": [],
        "val_dice": [],
        "val_ohem": [],
        "val_oa": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_iou": [],
        "val_miou": [],
        "val_ap": [],
        "lr": [],
        "time": [],
        "ohem_frac": [],
        "pos_weight": [],
        "best_threshold": [],
    }
    resource_history = init_resource_history()
    global_gpu_tracker = GpuMetricTracker(device)

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    current_pos_weight = cfg.pos_weight
    best_threshold = 0.5
    last_epoch = 0
    early_stop_triggered = False
    train_flops_samples = []
    total_train_images_seen = 0
    validation_pass_count = 0
    threshold_search_pass_count = 0

    dynamic_ohem_enabled = (
        cfg.dynamic_ohem.get("enabled", False) if isinstance(cfg.dynamic_ohem, dict) else False
    )
    auto_pos_weight_enabled = (
        cfg.auto_pos_weight.get("enabled", False) if isinstance(cfg.auto_pos_weight, dict) else False
    )
    threshold_search_enabled = cfg.threshold_search.get("enabled", False)

    print("\n" + "=" * 60)
    print("Starting training loop with resource details")
    print("=" * 60)
    print("Training time per image is derived from training iteration time / images seen.")
    print("It includes forward, loss, backward, optimizer step, transfer, and CUDA sync.")
    print("Epoch total time per training image also includes validation/threshold-search overhead.")
    print("=" * 60 + "\n")

    for epoch in range(1, cfg.epochs + 1):
        last_epoch = epoch
        epoch_start = time.perf_counter()

        if dynamic_ohem_enabled:
            current_ohem_frac = compute_dynamic_ohem_frac(
                epoch,
                cfg.epochs,
                cfg.ohem_frac,
                cfg.ohem_frac_min,
            )
        else:
            current_ohem_frac = cfg.ohem_frac

        if auto_pos_weight_enabled:
            current_pos_weight = estimate_pos_weight_from_loader(
                train_loader,
                device,
                cfg,
                max_samples=50,
            )

        if threshold_search_enabled:
            interval = cfg.threshold_search.get("interval", 5)
            metric = cfg.threshold_search.get("metric", "f1")
            if epoch % interval == 0 or epoch == 1:
                best_threshold, best_score = find_best_threshold_grid_search(
                    enc,
                    head,
                    val_loader,
                    device,
                    cfg,
                    metric=metric,
                )
                threshold_search_pass_count += 1

        enc.eval()
        head.train()
        for module in trainables:
            module.train()

        running = {key: 0.0 for key in ["loss", "bce", "dice", "ohem"]}
        train_eval_metric = BinaryEvaluator(ignore_index=255)
        epoch_gpu_tracker = GpuMetricTracker(device)
        iter_times = []
        images_seen = 0
        train_loop_start = time.perf_counter()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.epochs} [Train]",
            ncols=120,
            leave=True,
        )
        for batch_idx, (x, y, _) in enumerate(pbar):
            iter_start = time.perf_counter()
            epoch_gpu_tracker.start_iter()
            profile_this_iter = (
                not skip_flops
                and profile_train_flops_batches > 0
                and len(train_flops_samples) < profile_train_flops_batches
            )
            prof = create_torch_flops_profiler(device) if profile_this_iter else None
            exc_info = (None, None, None)

            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")
            batch_images = int(x.shape[0])
            try:
                if prof is not None:
                    prof.__enter__()

                optim.zero_grad(set_to_none=True)

                with torch.amp.autocast(autocast_device, enabled=amp_enabled):
                    feat = enc(x)
                    logits = head_logits(head(feat, (x.shape[2], x.shape[3])))
                    bce = bce_ignore(logits, y, pos_weight=current_pos_weight)
                    dice = dice_loss(logits, y)
                    eloss = elementwise_bce(logits, y, pos_weight=current_pos_weight)
                    ohem = ohem_topk(eloss, frac=current_ohem_frac)
                    loss = cfg.w_bce * bce + cfg.w_dice * dice + cfg.w_ohem * ohem

                scaler.scale(loss).backward()

                if cfg.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.grad_clip)

                scaler.step(optim)
                scaler.update()
            except BaseException:
                exc_info = sys.exc_info()
                raise
            finally:
                if prof is not None:
                    prof.__exit__(*exc_info)
                    if exc_info[0] is None:
                        prof_flops = profiler_total_flops(prof)
                        train_flops_samples.append(
                            {
                                "batch_size": batch_images,
                                "flops": prof_flops,
                                "flops_per_image": prof_flops / max(1, batch_images),
                                "scope": "forward + loss + backward + optimizer step",
                                "method": "torch.profiler",
                            }
                        )

            peak_bytes, _ = epoch_gpu_tracker.end_iter()
            iter_time = time.perf_counter() - iter_start
            iter_times.append(iter_time)
            images_seen += batch_images
            total_train_images_seen += batch_images

            running["loss"] += float(loss.detach())
            running["bce"] += float(bce.detach())
            running["dice"] += float(dice.detach())
            running["ohem"] += float(ohem.detach())

            with torch.no_grad():
                prob = torch.sigmoid(logits)
                stat_thr = best_threshold if threshold_search_enabled else 0.5
                pred = (prob > stat_thr).long().squeeze(1)
                for b in range(pred.shape[0]):
                    train_eval_metric.add_batch(y[b], pred[b])

            postfix = {
                "loss": f"{float(loss.detach()):.4f}",
                "bce": f"{float(bce.detach()):.4f}",
                "dice": f"{float(dice.detach()):.4f}",
                "ms/img": f"{(iter_time / max(1, batch_images)) * 1000.0:.1f}",
            }
            if peak_bytes is not None:
                postfix["peakMB"] = f"{bytes_to_mb(peak_bytes):.0f}"
            pbar.set_postfix(postfix)

        train_loop_time = time.perf_counter() - train_loop_start
        global_gpu_tracker.extend(epoch_gpu_tracker)
        epoch_gpu_summary = epoch_gpu_tracker.summary()

        niter = len(train_loader)
        train_metrics = {
            "loss": running["loss"] / niter,
            "bce": running["bce"] / niter,
            "dice": running["dice"] / niter,
            "ohem": running["ohem"] / niter,
            "oa": train_eval_metric.Accuracy(),
            "precision": train_eval_metric.Precision(class_idx=1),
            "recall": train_eval_metric.Recall(class_idx=1),
            "f1": train_eval_metric.F1Score(class_idx=1),
            "iou": train_eval_metric.IoU(class_idx=1),
            "miou": train_eval_metric.mIoU(),
            "ap": train_eval_metric.AP(class_idx=1),
        }

        val_threshold = best_threshold if threshold_search_enabled else 0.5
        val_metrics = validate_with_metrics(
            enc,
            head,
            val_loader,
            device,
            cfg,
            thr=val_threshold,
            pos_weight=current_pos_weight,
            ohem_frac=current_ohem_frac,
        )
        validation_pass_count += 1

        epoch_time = time.perf_counter() - epoch_start
        current_lr = optim.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_bce"].append(train_metrics["bce"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_ohem"].append(train_metrics["ohem"])
        history["train_oa"].append(train_metrics["oa"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_iou"].append(train_metrics["iou"])
        history["train_miou"].append(train_metrics["miou"])
        history["train_ap"].append(train_metrics["ap"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_bce"].append(val_metrics["bce"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_ohem"].append(val_metrics["ohem"])
        history["val_oa"].append(val_metrics["oa"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_miou"].append(val_metrics["miou"])
        history["val_ap"].append(val_metrics["ap"])
        history["lr"].append(current_lr)
        history["time"].append(epoch_time)
        history["ohem_frac"].append(current_ohem_frac)
        history["pos_weight"].append(current_pos_weight)
        history["best_threshold"].append(best_threshold)

        append_resource_history(
            resource_history,
            epoch,
            train_loop_time,
            epoch_time,
            images_seen,
            iter_times,
            epoch_gpu_summary,
            len(train_set),
            train_flops_samples,
        )

        ckpt = {
            "epoch": epoch,
            "sam_type": cfg.sam_type,
            "cfg": vars(cfg),
            "head": head.state_dict(),
            "enc": enc.state_dict(),
            "history": history,
            "resource_history": resource_history,
            "forward_profile": flops_info,
        }
        torch.save(ckpt, os.path.join(cfg.out, "last.pt"))

        is_best = False
        reason = ""
        if val_metrics["loss"] <= best_val_loss:
            is_best = True
            reason = f"Val Loss: {best_val_loss:.4f} -> {val_metrics['loss']:.4f}"
            best_val_loss = val_metrics["loss"]

        if is_best:
            torch.save(ckpt, os.path.join(cfg.out, "best.pt"))
            print(f"New best.pt saved ({reason})")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(ckpt, os.path.join(cfg.out, "best_f1.pt"))
            print(f"best_f1.pt updated: F1={best_val_f1:.4f} (thr={best_threshold:.3f})")

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        save_training_history(history, cfg.out)
        save_resource_history(resource_history, cfg.out)

        if early_stopping is not None:
            monitor_metric = cfg.early_stopping_config["monitor"]
            if monitor_metric == "val_loss":
                metric_value = val_metrics["loss"]
            elif monitor_metric == "val_ap":
                metric_value = val_metrics["ap"]
            elif monitor_metric == "val_f1":
                metric_value = val_metrics["f1"]
            else:
                metric_value = val_metrics["loss"]

            if early_stopping(epoch, metric_value):
                early_stop_triggered = True
                break

    global_gpu_summary = global_gpu_tracker.summary()
    avg_train_time_per_image = (
        mean(resource_history["train_time_per_image_s"])
        if resource_history["train_time_per_image_s"]
        else None
    )
    avg_epoch_total_time_per_image = (
        mean(resource_history["epoch_total_time_per_training_image_s"])
        if resource_history["epoch_total_time_per_training_image_s"]
        else None
    )
    train_step_flops_per_batch_mean = (
        mean(sample["flops"] for sample in train_flops_samples)
        if train_flops_samples
        else None
    )
    train_step_flops_per_image_mean = (
        mean(sample["flops_per_image"] for sample in train_flops_samples)
        if train_flops_samples
        else None
    )
    estimated_train_optimizer_flops_total = (
        train_step_flops_per_image_mean * total_train_images_seen
        if train_step_flops_per_image_mean is not None
        else None
    )
    forward_flops_per_image = flops_info.get("flops_2mac") if flops_info.get("available") else None
    estimated_validation_forward_flops = (
        forward_flops_per_image * len(val_set) * validation_pass_count
        if forward_flops_per_image is not None
        else None
    )
    estimated_threshold_search_forward_flops = (
        forward_flops_per_image * len(val_set) * threshold_search_pass_count
        if forward_flops_per_image is not None
        else None
    )
    estimated_training_process_flops = None
    if estimated_train_optimizer_flops_total is not None:
        estimated_training_process_flops = estimated_train_optimizer_flops_total
        if estimated_validation_forward_flops is not None:
            estimated_training_process_flops += estimated_validation_forward_flops
        if estimated_threshold_search_forward_flops is not None:
            estimated_training_process_flops += estimated_threshold_search_forward_flops

    summary = {
        "script": "train-details.py",
        "device": str(device),
        "batch_size": cfg.batch_size,
        "train_images": len(train_set),
        "val_images": len(val_set),
        "epochs_completed": last_epoch,
        "early_stopped": early_stop_triggered,
        "parameters": {
            "trainable": trainable_params,
            "optimizer": optimizer_param_count,
            "total": total_params,
        },
        "forward_profile": flops_info,
        "training_flops_profile": {
            "available": bool(train_flops_samples),
            "method": "torch.profiler",
            "profiled_batches": len(train_flops_samples),
            "requested_profile_batches": profile_train_flops_batches,
            "scope": "Actual training step: forward + loss + backward + optimizer step.",
            "train_step_flops_per_batch_mean": train_step_flops_per_batch_mean,
            "train_step_flops_per_image_mean": train_step_flops_per_image_mean,
            "total_train_images_seen": total_train_images_seen,
            "estimated_train_optimizer_flops_total": estimated_train_optimizer_flops_total,
            "validation_pass_count": validation_pass_count,
            "threshold_search_pass_count": threshold_search_pass_count,
            "estimated_validation_forward_flops": estimated_validation_forward_flops,
            "estimated_threshold_search_forward_flops": estimated_threshold_search_forward_flops,
            "estimated_training_process_flops": estimated_training_process_flops,
            "note": (
                "This is a PyTorch-op FLOPs estimate. It includes profiled training-step "
                "model/loss/backward/optimizer tensor math and estimates validation/search "
                "using forward FLOPs. DataLoader, PIL/NumPy work, CSV/checkpoint I/O, and "
                "unsupported profiler ops are not counted."
            ),
            "samples": train_flops_samples,
        },
        "time": {
            "training_time_per_image_s_mean": avg_train_time_per_image,
            "epoch_total_time_per_training_image_s_mean": avg_epoch_total_time_per_image,
            "training_time_note": (
                "training_time_per_image_s_mean is train loop time / images seen; "
                "it includes forward, loss, backward, optimizer step, transfer, and CUDA sync."
            ),
        },
        "gpu_memory": global_gpu_summary,
        "notes": [
            "forward_profile is forward-pass-only at the listed input size.",
            "training_flops_profile estimates training-step FLOPs from real profiled training batches.",
            "GPU memory is peak allocated memory; average allocated memory is sampled after each training iteration.",
        ],
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
    }
    save_resource_summary(summary, cfg.out)

    print(f"\n{'=' * 60}")
    print("Training complete")
    print(f"{'=' * 60}")
    print(f"Final epoch: {last_epoch}")
    if early_stop_triggered:
        print("Stopped by early stopping.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val F1: {best_val_f1:.4f}")
    if avg_train_time_per_image is not None:
        print(
            "Mean processing time per image (training): "
            f"{avg_train_time_per_image * 1000.0:.2f} ms/image "
            f"(from training iterations; batch size={cfg.batch_size})"
        )
    if avg_epoch_total_time_per_image is not None:
        print(
            "Mean epoch total time per training image: "
            f"{avg_epoch_total_time_per_image * 1000.0:.2f} ms/image"
        )
    if train_step_flops_per_image_mean is not None:
        print(
            "Mean training step FLOPs per image: "
            f"{human_number(train_step_flops_per_image_mean, 'FLOPs')} "
            "(forward + loss + backward + optimizer, profiled real batches)"
        )
    if estimated_training_process_flops is not None:
        print(
            "Estimated training process FLOPs: "
            f"{human_number(estimated_training_process_flops, 'FLOPs')} "
            "(training steps + validation/search forward estimates)"
        )
    print(gpu_summary_text(global_gpu_summary))
    print_forward_profile(flops_info)
    print(f"Model saved in: {cfg.out}")
    print(f"Training history: {os.path.join(cfg.out, 'training_history.csv')}")
    print(f"Resource history: {os.path.join(cfg.out, 'training_resource_history.csv')}")
    print(f"Resource summary: {os.path.join(cfg.out, 'resource_summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="SAM-ATL training with resource details")
    parser.add_argument(
        "--config",
        type=str,
        default="SAM-ATL/config_train.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--flops-img-size",
        type=int,
        default=None,
        help="Input size used for forward FLOPs profiling. Defaults to data.img_size.",
    )
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip FLOPs profiling if you only need timing/memory.",
    )
    parser.add_argument(
        "--profile-train-flops-batches",
        type=int,
        default=1,
        help=(
            "Number of real training batches to profile for training-step FLOPs. "
            "The profiled step includes forward, loss, backward, and optimizer. "
            "Use 0 to disable."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        alt_config = os.path.join(os.path.dirname(__file__), os.path.basename(args.config))
        if os.path.exists(alt_config):
            args.config = alt_config

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        return

    print(f"Loading config: {args.config}")
    config_dict = normalize_config_paths(load_config(args.config), args.config)
    cfg = config_to_args(config_dict)
    cfg._config_dict = config_dict
    train_with_details(
        cfg,
        flops_img_size=args.flops_img_size,
        skip_flops=args.skip_flops,
        profile_train_flops_batches=args.profile_train_flops_batches,
    )


if __name__ == "__main__":
    main()
