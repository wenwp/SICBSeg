
import argparse
import os
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_bin import DestroyedBinaryDataset
from model_loader import build_and_load_model, head_logits


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
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
    model_cfg = config.get('model', {})
    for key in ('checkpoint', 'sam_checkpoint'):
        if key in model_cfg:
            model_cfg[key] = _resolve_input_path(model_cfg[key], config_dir)

    data_cfg = config.get('data', {})
    for key in ('val_images_dir', 'val_labels_dir'):
        if key in data_cfg:
            data_cfg[key] = _resolve_input_path(data_cfg[key], config_dir)

    out_cfg = config.get('output', {})
    if 'dir' in out_cfg:
        out_cfg['dir'] = _resolve_output_path(out_cfg['dir'], config_dir)
    return config


def _collect_prob_histograms(enc, head, loader, device, num_bins=1000):
    """
    一次性遍历验证集，把所有有效像素的概率累计到直方图。
    返回 (pos_hist, neg_hist)，长度均为 num_bins。
    """
    pos_hist = np.zeros(num_bins, dtype=np.int64)
    neg_hist = np.zeros(num_bins, dtype=np.int64)

    enc.eval()
    head.eval()
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc='收集概率分布', ncols=100):
            x = x.to(device)
            y = y.to(device)
            logits = head_logits(head(enc(x), (x.shape[2], x.shape[3])))
            prob = torch.sigmoid(logits).squeeze(1)  # [B, H, W]

            valid = (y != 255)
            prob_v = prob[valid].clamp(0.0, 1.0).float().cpu().numpy()
            y_v = y[valid].cpu().numpy()

            bins = np.minimum((prob_v * num_bins).astype(np.int64), num_bins - 1)
            if (y_v == 1).any():
                pos_hist += np.bincount(bins[y_v == 1], minlength=num_bins)
            if (y_v == 0).any():
                neg_hist += np.bincount(bins[y_v == 0], minlength=num_bins)

    return pos_hist, neg_hist


def _sweep_metrics(pos_hist, neg_hist, thresholds):
    """
    给定一组阈值，从直方图算每个阈值下的 (P, R, F1, IoU)。
    """
    num_bins = len(pos_hist)
    # 累积：bin >= idx 的总数
    cum_pos = np.concatenate([np.cumsum(pos_hist[::-1])[::-1], [0]]).astype(np.float64)
    cum_neg = np.concatenate([np.cumsum(neg_hist[::-1])[::-1], [0]]).astype(np.float64)
    # cum_pos[i] = sum(pos_hist[i:])  → 真实正且 prob 落在 bin i..end 的像素数
    # cum_neg[i] 同理
    total_pos = cum_pos[0]
    total_neg = cum_neg[0]

    results = []
    for thr in thresholds:
        # bin 索引：prob >= thr 等价于 bin >= int(thr*num_bins)
        idx = int(min(num_bins, max(0, round(thr * num_bins))))
        tp = cum_pos[idx]
        fp = cum_neg[idx]
        fn = total_pos - tp
        tn = total_neg - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        results.append({
            'thr': float(thr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })
    return results


def sweep(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    ckpt_path = config['model']['checkpoint']
    sam_checkpoint = config['model']['sam_checkpoint']

    data_cfg = config['data']
    val_images_dir = data_cfg['val_images_dir']
    val_labels_dir = data_cfg['val_labels_dir']
    img_size = data_cfg['img_size']
    destroyed_idx = data_cfg.get('destroyed_idx', 3)
    destroyed_value_is_255 = data_cfg.get('destroyed_value_is_255', False)
    require_post_suffix = data_cfg.get('require_post_suffix', False)
    batch_size = data_cfg.get('batch_size', 4)
    workers = data_cfg.get('workers', 4)

    out_dir = config['output']['dir']

    print("加载验证数据集...")
    ds = DestroyedBinaryDataset(
        val_images_dir, val_labels_dir, img_size, destroyed_idx, False,
        destroyed_value_is_255=destroyed_value_is_255,
        require_post_suffix=require_post_suffix,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    print(f"验证样本数: {len(ds)}")

    print(f"加载模型: {ckpt_path}")
    enc, head, info = build_and_load_model(ckpt_path, sam_checkpoint, config, device)
    resolved = info['resolved']
    print(f"  ✓ 解析配置: sam={resolved['sam_type']}, atl={resolved['atl_type']}, "
          f"last_blocks={resolved['last_blocks']}, prompts={resolved['num_learnable_prompts']}, "
          f"use_iou_head={resolved['use_iou_head']}")
    print(f"  ✓ encoder load (missing={len(info['missing'])}, unexpected={len(info['unexpected'])})")

    # 1) 一次 forward 收集直方图
    pos_hist, neg_hist = _collect_prob_histograms(enc, head, dl, device, num_bins=1000)
    if pos_hist.sum() == 0:
        print("警告: 验证集中没有任何正类像素，无法做 F1 校准。")
        return

    # 2) 在直方图上扫细粒度阈值
    thresholds = np.arange(0.01, 1.00, 0.01)
    results = _sweep_metrics(pos_hist, neg_hist, thresholds)

    # 3) 找各指标的最佳阈值
    best = {}
    for metric in ('f1', 'iou', 'precision', 'recall'):
        best_item = max(results, key=lambda r: r[metric])
        best[metric] = {'thr': best_item['thr'], metric: best_item[metric]}

    # 兼容旧字段（best_thr.json 中保留 thr/f1）
    payload = {
        'thr': best['f1']['thr'],
        'f1': best['f1']['f1'],
        'best_per_metric': best,
        'resolved_config': resolved,
        'sweep': results,
    }

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, 'best_thr.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n✓ 校准完成!")
    for m in ('f1', 'iou', 'precision', 'recall'):
        print(f"  最佳 {m:9s}: thr={best[m]['thr']:.2f}, {m}={best[m][m]:.4f}")
    print(f"  结果已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SAM-ATL 阈值校准脚本')
    parser.add_argument('--config', type=str, default='config_calibrate.yaml',
                        help='YAML 配置文件路径')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        alt_config = os.path.join(os.path.dirname(__file__), os.path.basename(args.config))
        if os.path.exists(alt_config):
            args.config = alt_config

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return

    print(f"加载配置文件: {args.config}")
    config = normalize_config_paths(load_config(args.config), args.config)
    sweep(config)


if __name__ == '__main__':
    main()
