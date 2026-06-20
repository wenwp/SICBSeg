import argparse
import os
import time
import yaml

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from BinaryEvaluator import BinaryEvaluator
from dataset_bin import DestroyedBinaryDataset
from losses import BinaryDiceLoss, bce_ignore, elementwise_bce, ohem_topk
from model_loader import build_and_load_model, head_logits
from train import (
    EarlyStopping,
    compute_dynamic_ohem_frac,
    create_scheduler,
    estimate_pos_weight_from_loader,
    find_best_threshold_grid_search,
    save_training_history,
    validate_with_metrics,
)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    data_cfg = config.get("data", {})
    for key in ("train_images_dir", "train_labels_dir", "val_images_dir", "val_labels_dir"):
        if key in data_cfg:
            data_cfg[key] = _resolve_input_path(data_cfg[key], config_dir)

    model_cfg = config.get("model", {})
    for key in ("checkpoint", "sam_checkpoint"):
        if key in model_cfg and model_cfg[key] is not None:
            model_cfg[key] = _resolve_input_path(model_cfg[key], config_dir)

    transfer_cfg = config.get("transfer", {})
    if "init_checkpoint" in transfer_cfg and transfer_cfg["init_checkpoint"] is not None:
        transfer_cfg["init_checkpoint"] = _resolve_input_path(
            transfer_cfg["init_checkpoint"], config_dir
        )

    out_cfg = config.get("output", {})
    if "dir" in out_cfg:
        out_cfg["dir"] = _resolve_output_path(out_cfg["dir"], config_dir)
    return config


def config_to_args(config):
    class Args:
        pass

    args = Args()
    transfer_cfg = config.get("transfer", {})
    model_cfg = config.get("model", {})
    data_cfg = config["data"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]

    args.init_checkpoint = transfer_cfg.get("init_checkpoint") or model_cfg.get("checkpoint")
    args.sam_checkpoint = model_cfg["sam_checkpoint"]
    args.train_encoder_adapters = transfer_cfg.get("train_encoder_adapters", True)
    args.train_decoder = transfer_cfg.get("train_decoder", True)
    args.drop_last = transfer_cfg.get("drop_last", False)

    args.train_images_dir = data_cfg["train_images_dir"]
    args.train_labels_dir = data_cfg["train_labels_dir"]
    args.val_images_dir = data_cfg["val_images_dir"]
    args.val_labels_dir = data_cfg["val_labels_dir"]
    args.img_size = data_cfg["img_size"]
    args.destroyed_idx = data_cfg.get("destroyed_idx", 3)
    args.destroyed_value_is_255 = data_cfg.get("destroyed_value_is_255", False)
    args.require_post_suffix = data_cfg.get("require_post_suffix", False)
    args.augmentation = data_cfg.get("augmentation", {"enabled": False})

    args.batch_size = train_cfg["batch_size"]
    args.workers = train_cfg["workers"]
    args.epochs = train_cfg["epochs"]
    args.lr = train_cfg["lr"]
    args.weight_decay = train_cfg["weight_decay"]
    args.no_amp = train_cfg.get("no_amp", False)
    args.grad_clip = train_cfg.get("grad_clip", 0.0)
    args.scheduler_config = train_cfg.get("scheduler", {"enabled": False, "type": "none"})
    args.early_stopping_config = train_cfg.get(
        "early_stopping",
        {"enabled": False, "monitor": "val_loss", "patience": 10, "min_delta": 0.001},
    )

    args.pos_weight = loss_cfg["pos_weight"]
    args.ohem_frac = loss_cfg["ohem_frac"]
    args.ohem_frac_min = loss_cfg.get("ohem_frac_min", 0.05)
    args.w_bce = loss_cfg["w_bce"]
    args.w_dice = loss_cfg["w_dice"]
    args.w_ohem = loss_cfg["w_ohem"]
    args.dynamic_ohem = loss_cfg.get("dynamic_ohem", {"enabled": False})
    args.auto_pos_weight = loss_cfg.get("auto_pos_weight", {"enabled": False})

    args.threshold_search = config.get(
        "threshold_search", {"enabled": False, "interval": 5, "metric": "f1"}
    )
    args.out = config["output"]["dir"]
    return args


def _set_trainable_submodules_train(module):
    for submodule in module.modules():
        if any(p.requires_grad for p in submodule.parameters(recurse=True)):
            submodule.train()


def _init_history():
    return {
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


def _append_history(history, epoch, train_metrics, val_metrics, lr, elapsed, ohem_frac,
                    pos_weight, best_threshold):
    history["epoch"].append(epoch)
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key in ("loss", "bce", "dice", "ohem", "oa", "precision", "recall",
                    "f1", "iou", "miou", "ap"):
            history[f"{prefix}_{key}"].append(metrics[key])
    history["lr"].append(lr)
    history["time"].append(elapsed)
    history["ohem_frac"].append(ohem_frac)
    history["pos_weight"].append(pos_weight)
    history["best_threshold"].append(best_threshold)


def transfer_train(cfg, raw_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not cfg.no_amp
    os.makedirs(cfg.out, exist_ok=True)

    config_save_path = os.path.join(cfg.out, "transfer_config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(raw_config, f, allow_unicode=True, sort_keys=False)

    print(f"Device: {device}")
    print(f"Source checkpoint: {cfg.init_checkpoint}")
    print(f"Output dir: {cfg.out}")

    if not cfg.init_checkpoint or not os.path.exists(cfg.init_checkpoint):
        raise FileNotFoundError(f"init_checkpoint not found: {cfg.init_checkpoint}")

    aug_cfg = cfg.augmentation if isinstance(cfg.augmentation, dict) else {"enabled": False}
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
        drop_last=cfg.drop_last,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Transfer train samples: {len(train_set)}, val samples: {len(val_set)}")

    enc, head, info = build_and_load_model(cfg.init_checkpoint, cfg.sam_checkpoint, raw_config, device)
    resolved = info["resolved"]
    cfg.sam_type = resolved["sam_type"]
    cfg.atl_type = resolved["atl_type"]
    cfg.last_blocks = resolved["last_blocks"]
    cfg.atl_rank = resolved["atl_rank"]
    cfg.lora_alpha = resolved["lora_alpha"]
    cfg.atl_dropout = resolved["atl_dropout"]
    cfg.num_learnable_prompts = resolved["num_learnable_prompts"]
    cfg.use_iou_head = resolved["use_iou_head"]

    print(
        "Loaded model config: "
        f"sam={cfg.sam_type}, atl={cfg.atl_type}, last_blocks={cfg.last_blocks}, "
        f"rank={cfg.atl_rank}, prompts={cfg.num_learnable_prompts}, "
        f"use_iou_head={cfg.use_iou_head}"
    )
    print(f"Encoder load: missing={len(info['missing'])}, unexpected={len(info['unexpected'])}")

    if not cfg.train_encoder_adapters:
        for p in enc.parameters():
            p.requires_grad = False
    if not cfg.train_decoder:
        for p in head.parameters():
            p.requires_grad = False

    params = [p for p in list(enc.parameters()) + list(head.parameters()) if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters. Enable train_encoder_adapters or train_decoder.")

    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = create_scheduler(optim, cfg.scheduler_config, cfg.epochs)

    early_stopping = None
    if cfg.early_stopping_config.get("enabled", False):
        monitor = cfg.early_stopping_config.get("monitor", "val_loss")
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping_config.get("patience", 10),
            min_delta=cfg.early_stopping_config.get("min_delta", 0.001),
            mode="min" if "loss" in monitor else "max",
            verbose=True,
        )

    dice_loss = BinaryDiceLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = _init_history()
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    current_pos_weight = cfg.pos_weight
    best_threshold = 0.5

    dynamic_ohem_enabled = (
        cfg.dynamic_ohem.get("enabled", False) if isinstance(cfg.dynamic_ohem, dict) else False
    )
    auto_pos_weight_enabled = (
        cfg.auto_pos_weight.get("enabled", False) if isinstance(cfg.auto_pos_weight, dict) else False
    )
    threshold_search_enabled = cfg.threshold_search.get("enabled", False)

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        if dynamic_ohem_enabled:
            current_ohem_frac = compute_dynamic_ohem_frac(
                epoch, cfg.epochs, cfg.ohem_frac, cfg.ohem_frac_min
            )
        else:
            current_ohem_frac = cfg.ohem_frac

        if auto_pos_weight_enabled:
            current_pos_weight = estimate_pos_weight_from_loader(
                train_loader, device, cfg, max_samples=50
            )
            print(f"[Epoch {epoch}] pos_weight={current_pos_weight:.3f}")

        if threshold_search_enabled:
            interval = cfg.threshold_search.get("interval", 5)
            metric = cfg.threshold_search.get("metric", "f1")
            if epoch == 1 or epoch % interval == 0:
                best_threshold, best_score = find_best_threshold_grid_search(
                    enc, head, val_loader, device, cfg, metric=metric
                )
                print(f"[Epoch {epoch}] best_threshold={best_threshold:.3f}, {metric}={best_score:.4f}")

        enc.eval()
        if cfg.train_encoder_adapters:
            _set_trainable_submodules_train(enc)
        if cfg.train_decoder:
            head.train()
        else:
            head.eval()

        running = {k: 0.0 for k in ("loss", "bce", "dice", "ohem")}
        train_eval_metric = BinaryEvaluator(ignore_index=255)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Transfer]", ncols=120)

        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
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

            running["loss"] += float(loss)
            running["bce"] += float(bce)
            running["dice"] += float(dice)
            running["ohem"] += float(ohem)

            with torch.no_grad():
                prob = torch.sigmoid(logits)
                stat_thr = best_threshold if threshold_search_enabled else 0.5
                pred = (prob > stat_thr).long().squeeze(1)
                for b in range(pred.shape[0]):
                    train_eval_metric.add_batch(y[b], pred[b])
                    train_eval_metric.add_batch_prob(y[b], prob[b, 0])

            pbar.set_postfix(
                {
                    "loss": f"{float(loss):.4f}",
                    "bce": f"{float(bce):.4f}",
                    "dice": f"{float(dice):.4f}",
                }
            )

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

        elapsed = time.time() - epoch_start
        current_lr = optim.param_groups[0]["lr"]
        _append_history(
            history,
            epoch,
            train_metrics,
            val_metrics,
            current_lr,
            elapsed,
            current_ohem_frac,
            current_pos_weight,
            best_threshold,
        )

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, train_f1={train_metrics['f1']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_f1={val_metrics['f1']:.4f}, "
            f"val_iou={val_metrics['iou']:.4f}, val_ap={val_metrics['ap']:.4f}, "
            f"thr={val_threshold:.3f}, lr={current_lr:.2e}, time={elapsed:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "sam_type": cfg.sam_type,
            "cfg": vars(cfg),
            "transfer": {
                "source_checkpoint": cfg.init_checkpoint,
                "train_encoder_adapters": cfg.train_encoder_adapters,
                "train_decoder": cfg.train_decoder,
            },
            "head": head.state_dict(),
            "enc": enc.state_dict(),
            "history": history,
        }
        torch.save(ckpt, os.path.join(cfg.out, "last.pt"))

        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(ckpt, os.path.join(cfg.out, "best.pt"))
            print(f"Saved best.pt by val_loss={best_val_loss:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(ckpt, os.path.join(cfg.out, "best_f1.pt"))
            print(f"Saved best_f1.pt by val_f1={best_val_f1:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        if early_stopping is not None:
            monitor = cfg.early_stopping_config.get("monitor", "val_loss")
            metric_value = {
                "val_loss": val_metrics["loss"],
                "val_ap": val_metrics["ap"],
                "val_f1": val_metrics["f1"],
            }.get(monitor, val_metrics["loss"])
            if early_stopping(epoch, metric_value):
                print(f"Early stopping at epoch {epoch}")
                break

        save_training_history(history, cfg.out)

    print(f"Transfer training finished. Results saved to: {cfg.out}")


def main():
    parser = argparse.ArgumentParser(description="Transfer train SAM-ATL from a trained checkpoint")
    parser.add_argument("--config", type=str, default="config_trans.yaml", help="YAML config path")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        alt_config = os.path.join(os.path.dirname(__file__), os.path.basename(args.config))
        if os.path.exists(alt_config):
            args.config = alt_config

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    raw_config = normalize_config_paths(load_config(args.config), args.config)
    cfg = config_to_args(raw_config)
    cfg._config_dict = raw_config
    transfer_train(cfg, raw_config)


if __name__ == "__main__":
    main()
