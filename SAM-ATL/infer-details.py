
import argparse
import csv
import json
import os
import sys
import time
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from atl_modules import LoRALinear
from infer import create_overlay, load_config, normalize_config_paths, postprocess, preprocess
from model_loader import build_and_load_model, head_logits


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

    def summary(self):
        if not self.enabled or not self.iter_peak_bytes:
            return {
                "available": False,
                "samples": 0,
                "peak_allocated_mb": None,
                "avg_per_image_peak_allocated_mb": None,
                "avg_sampled_allocated_mb": None,
            }
        return {
            "available": True,
            "samples": len(self.iter_peak_bytes),
            "peak_allocated_mb": bytes_to_mb(max(self.iter_peak_bytes)),
            "avg_per_image_peak_allocated_mb": bytes_to_mb(mean(self.iter_peak_bytes)),
            "avg_sampled_allocated_mb": bytes_to_mb(mean(self.sample_allocated_bytes)),
        }


def gpu_summary_text(summary):
    if not summary["available"]:
        return "GPU memory: unavailable (CUDA not in use)"
    return (
        f"Peak GPU memory: {summary['peak_allocated_mb']:.1f} MB | "
        f"Avg per-image peak: {summary['avg_per_image_peak_allocated_mb']:.1f} MB | "
        f"Avg sampled allocated: {summary['avg_sampled_allocated_mb']:.1f} MB"
    )


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


def image_file_list(single_image, images_dir):
    if single_image:
        return [single_image]
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    return [
        os.path.join(images_dir, name)
        for name in sorted(os.listdir(images_dir))
        if name.lower().endswith(exts)
    ]


def save_detail_rows(rows, output_dir):
    if not rows:
        return
    path = os.path.join(output_dir, "inference_resource_details.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary(summary, output_dir):
    path = os.path.join(output_dir, "inference_resource_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_inference_with_details(
    config,
    flops_img_size=None,
    skip_flops=False,
    flops_only=False,
    profile_infer_flops_images=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = config["model"]["checkpoint"]
    sam_checkpoint = config["model"]["sam_checkpoint"]

    inf_cfg = config["inference"]
    img_size = inf_cfg["img_size"]
    single_image = inf_cfg.get("single_image")
    images_dir = inf_cfg.get("images_dir")
    output_dir = inf_cfg["output_dir"]
    thr = float(inf_cfg.get("threshold", 0.5))
    save_prob_map = inf_cfg.get("save_prob_map", False)
    save_binary_mask = inf_cfg.get("save_binary_mask", True)
    save_overlay = inf_cfg.get("save_overlay", False)

    print(f"Loading model: {ckpt_path}")
    enc, head, info = build_and_load_model(ckpt_path, sam_checkpoint, config, device)
    resolved = info["resolved"]
    print(
        "Resolved model config: "
        f"sam={resolved['sam_type']}, atl={resolved['atl_type']}, "
        f"last_blocks={resolved['last_blocks']}, prompts={resolved['num_learnable_prompts']}, "
        f"use_iou_head={resolved['use_iou_head']}"
    )
    print(f"Encoder load: missing={len(info['missing'])}, unexpected={len(info['unexpected'])}")
    print(f"Threshold: {thr:.3f}")

    trainable_params = count_unique_parameters([enc, head], trainable_only=True)
    total_params = count_unique_parameters([enc, head], trainable_only=False)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total model parameters: {total_params:,}")

    flops_info = profile_forward_flops(
        enc,
        head,
        device,
        flops_img_size or img_size,
        skip=skip_flops,
    )
    print_forward_profile(flops_info)

    if flops_only:
        os.makedirs(output_dir, exist_ok=True)
        summary = {
            "script": "infer-details.py",
            "mode": "flops-only",
            "device": str(device),
            "checkpoint": ckpt_path,
            "batch_size": 1,
            "parameters": {
                "trainable": trainable_params,
                "total": total_params,
            },
            "forward_profile": flops_info,
            "notes": [
                "FLOPs are forward-pass-only at the listed input size.",
                "This mode does not run dataset inference or measure per-image time.",
            ],
        }
        save_summary(summary, output_dir)
        print(f"FLOPs-only summary JSON: {os.path.join(output_dir, 'inference_resource_summary.json')}")
        return

    os.makedirs(output_dir, exist_ok=True)
    if save_prob_map:
        os.makedirs(os.path.join(output_dir, "prob_maps"), exist_ok=True)
    if save_binary_mask:
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    if save_overlay:
        os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    if single_image:
        if not os.path.exists(single_image):
            print(f"Error: image file not found: {single_image}")
            return
    elif images_dir:
        if not os.path.exists(images_dir):
            print(f"Error: images_dir not found: {images_dir}")
            return
    else:
        print("Error: specify either single_image or images_dir")
        return

    paths = image_file_list(single_image, images_dir)
    if not paths:
        print("Warning: no images found")
        return

    print("\nStarting inference...")
    print("Inference model time is measured with batch size = 1.")
    rows = []
    gpu_tracker = GpuMetricTracker(device)
    inference_flops_samples = []

    iterator = paths
    if len(paths) > 1:
        iterator = tqdm(paths, desc="Inference", ncols=100, unit="image")

    enc.eval()
    head.eval()
    for img_path in iterator:
        basename = os.path.basename(img_path)
        name = os.path.splitext(basename)[0]

        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        img = Image.open(img_path)
        x, pad, orig = preprocess(img, img_size)
        x = x.to(device, non_blocking=device.type == "cuda")
        preprocess_time = time.perf_counter() - preprocess_start

        gpu_tracker.start_iter()
        forward_start = time.perf_counter()
        profile_this_image = (
            not skip_flops
            and profile_infer_flops_images > 0
            and len(inference_flops_samples) < profile_infer_flops_images
        )
        prof = create_torch_flops_profiler(device) if profile_this_image else None
        exc_info = (None, None, None)
        try:
            if prof is not None:
                prof.__enter__()
            with torch.inference_mode():
                feat = enc(x)
                logits = head_logits(head(feat, (x.shape[2], x.shape[3])))
                prob = torch.sigmoid(logits)
        except BaseException:
            exc_info = sys.exc_info()
            raise
        finally:
            if prof is not None:
                prof.__exit__(*exc_info)
                if exc_info[0] is None:
                    prof_flops = profiler_total_flops(prof)
                    inference_flops_samples.append(
                        {
                            "image": basename,
                            "batch_size": 1,
                            "flops": prof_flops,
                            "scope": "Actual PyTorch inference path: encoder + decoder + sigmoid.",
                            "method": "torch.profiler",
                        }
                    )
        peak_bytes, allocated_bytes = gpu_tracker.end_iter()
        forward_time = time.perf_counter() - forward_start

        postprocess_start = time.perf_counter()
        prob_map = postprocess(prob, pad, orig)
        prob_array = np.array(prob_map)

        if save_prob_map:
            prob_map.save(os.path.join(output_dir, "prob_maps", f"{name}_prob.png"))
        if save_binary_mask:
            mask = (prob_array > int(thr * 255)).astype(np.uint8) * 255
            Image.fromarray(mask, "L").save(os.path.join(output_dir, "masks", f"{name}.png"))
        if save_overlay:
            mask_binary = (prob_array > int(thr * 255)).astype(np.uint8) * 255
            create_overlay(img, mask_binary).save(
                os.path.join(output_dir, "overlays", f"{name}_overlay.png")
            )
        postprocess_time = time.perf_counter() - postprocess_start
        total_time = time.perf_counter() - total_start

        rows.append(
            {
                "image": basename,
                "orig_width": orig[0],
                "orig_height": orig[1],
                "batch_size": 1,
                "preprocess_time_s": preprocess_time,
                "model_forward_time_s": forward_time,
                "postprocess_and_save_time_s": postprocess_time,
                "processing_time_s": total_time,
                "gpu_peak_allocated_mb": bytes_to_mb(peak_bytes),
                "gpu_sampled_allocated_mb": bytes_to_mb(allocated_bytes),
                "inference_pipeline_torch_flops": inference_flops_samples[-1]["flops"]
                if profile_this_image and inference_flops_samples
                else None,
            }
        )

        if len(paths) == 1:
            print(
                f"{basename}: processing={total_time:.4f}s, "
                f"model_forward={forward_time:.4f}s (batch size=1)"
            )

    save_detail_rows(rows, output_dir)

    gpu_summary = gpu_tracker.summary()
    avg_processing_time = mean(row["processing_time_s"] for row in rows)
    avg_forward_time = mean(row["model_forward_time_s"] for row in rows)
    avg_preprocess_time = mean(row["preprocess_time_s"] for row in rows)
    avg_postprocess_time = mean(row["postprocess_and_save_time_s"] for row in rows)
    avg_inference_pipeline_torch_flops = (
        mean(sample["flops"] for sample in inference_flops_samples)
        if inference_flops_samples
        else None
    )

    summary = {
        "script": "infer-details.py",
        "device": str(device),
        "checkpoint": ckpt_path,
        "images": len(rows),
        "batch_size": 1,
        "threshold": thr,
        "parameters": {
            "trainable": trainable_params,
            "total": total_params,
        },
        "forward_profile": flops_info,
        "inference_flops_profile": {
            "available": bool(inference_flops_samples),
            "method": "torch.profiler",
            "profiled_images": len(inference_flops_samples),
            "requested_profile_images": profile_infer_flops_images,
            "scope": "Actual PyTorch inference path: encoder + decoder + sigmoid.",
            "pipeline_torch_flops_per_image_mean": avg_inference_pipeline_torch_flops,
            "note": (
                "This profiles real inference tensor math for selected images. "
                "PIL preprocessing, NumPy/PIL postprocessing, thresholding after CPU conversion, "
                "image writing, and unsupported profiler ops are not counted as FLOPs."
            ),
            "samples": inference_flops_samples,
        },
        "time": {
            "processing_time_per_image_s_mean": avg_processing_time,
            "model_forward_time_per_image_s_mean": avg_forward_time,
            "preprocess_time_per_image_s_mean": avg_preprocess_time,
            "postprocess_and_save_time_per_image_s_mean": avg_postprocess_time,
            "inference_time_note": "Model forward inference time is measured one image at a time, batch size = 1.",
        },
        "gpu_memory": gpu_summary,
        "notes": [
            "forward_profile is synthetic forward-pass-only at the listed input size.",
            "inference_flops_profile is the actual PyTorch inference path for selected images.",
            "GPU memory is peak allocated memory; average allocated memory is sampled after each image.",
        ],
    }
    save_summary(summary, output_dir)

    print(f"\nInference complete. Results saved to: {output_dir}")
    if save_binary_mask:
        print(f"Binary masks: {os.path.join(output_dir, 'masks')}")
    if save_prob_map:
        print(f"Probability maps: {os.path.join(output_dir, 'prob_maps')}")
    if save_overlay:
        print(f"Overlays: {os.path.join(output_dir, 'overlays')}")

    print("\nResource usage summary")
    print(f"Images processed: {len(rows)}")
    print(
        "Processing time per image: "
        f"{avg_processing_time * 1000.0:.2f} ms/image "
        "(end-to-end preprocess + model + postprocess/save)"
    )
    print(
        "Inference model time per image: "
        f"{avg_forward_time * 1000.0:.2f} ms/image (batch size = 1)"
    )
    if avg_inference_pipeline_torch_flops is not None:
        print(
            "Inference pipeline PyTorch FLOPs per image: "
            f"{human_number(avg_inference_pipeline_torch_flops, 'FLOPs')} "
            "(encoder + decoder + sigmoid; batch size = 1)"
        )
    print(gpu_summary_text(gpu_summary))
    print_forward_profile(flops_info)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Per-image resource CSV: {os.path.join(output_dir, 'inference_resource_details.csv')}")
    print(f"Resource summary JSON: {os.path.join(output_dir, 'inference_resource_summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="SAM-ATL inference with resource details")
    parser.add_argument(
        "--config",
        type=str,
        default="config_infer.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--flops-img-size",
        type=int,
        default=None,
        help="Input size used for forward FLOPs profiling. Defaults to inference.img_size.",
    )
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip FLOPs profiling if you only need timing/memory.",
    )
    parser.add_argument(
        "--flops-only",
        action="store_true",
        help="Only load the checkpoint and report FLOPs/parameters; do not process images.",
    )
    parser.add_argument(
        "--profile-infer-flops-images",
        type=int,
        default=1,
        help=(
            "Number of real inference images to profile for pipeline PyTorch FLOPs. "
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
    config = normalize_config_paths(load_config(args.config), args.config)
    run_inference_with_details(
        config,
        flops_img_size=args.flops_img_size,
        skip_flops=args.skip_flops,
        flops_only=args.flops_only,
        profile_infer_flops_images=args.profile_infer_flops_images,
    )


if __name__ == "__main__":
    main()
