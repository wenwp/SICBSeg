
import argparse
import os
import yaml
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

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

    inf_cfg = config.get('inference', {})
    for key in ('single_image', 'images_dir'):
        if key in inf_cfg and inf_cfg[key] is not None:
            inf_cfg[key] = _resolve_input_path(inf_cfg[key], config_dir)
    if 'output_dir' in inf_cfg:
        inf_cfg['output_dir'] = _resolve_output_path(inf_cfg['output_dir'], config_dir)
    return config


def preprocess(img: Image.Image, img_size: int):
    """保持长宽比缩放、居中 padding 到 img_size×img_size，再做 ImageNet 归一化"""
    img = img.convert('RGB')
    w, h = img.size
    s = img_size / max(w, h)
    nw, nh = int(w * s), int(h * s)
    img = img.resize((nw, nh), Image.BILINEAR)
    pl, pt = (img_size - nw) // 2, (img_size - nh) // 2
    canvas = Image.new('RGB', (img_size, img_size), (0, 0, 0))
    canvas.paste(img, (pl, pt))

    x = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0), (pl, pt, nw, nh), (w, h)


def postprocess(prob: torch.Tensor, pad, orig_hw):
    """剥掉 padding，缩放回原始尺寸"""
    pl, pt, nw, nh = pad
    W, H = orig_hw
    p = prob[0, 0].cpu().numpy()
    p = p[pt:pt + nh, pl:pl + nw]
    p = Image.fromarray((p * 255).astype(np.uint8), 'L').resize((W, H), Image.BILINEAR)
    return p


def create_overlay(image: Image.Image, mask: np.ndarray, alpha=0.5):
    img_array = np.array(image.convert('RGB'))
    overlay = img_array.copy()
    overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    return Image.fromarray(overlay.astype(np.uint8))


def run_inference(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    ckpt_path = config['model']['checkpoint']
    sam_checkpoint = config['model']['sam_checkpoint']

    inf_cfg = config['inference']
    img_size = inf_cfg['img_size']
    single_image = inf_cfg.get('single_image')
    images_dir = inf_cfg.get('images_dir')
    output_dir = inf_cfg['output_dir']
    thr = float(inf_cfg.get('threshold', 0.5))
    save_prob_map = inf_cfg.get('save_prob_map', False)
    save_binary_mask = inf_cfg.get('save_binary_mask', True)
    save_overlay = inf_cfg.get('save_overlay', False)

    print(f"加载模型: {ckpt_path}")
    enc, head, info = build_and_load_model(ckpt_path, sam_checkpoint, config, device)
    resolved = info['resolved']
    print(f"  ✓ 解析配置: sam={resolved['sam_type']}, atl={resolved['atl_type']}, "
          f"last_blocks={resolved['last_blocks']}, prompts={resolved['num_learnable_prompts']}, "
          f"use_iou_head={resolved['use_iou_head']}")
    print(f"  ✓ encoder load (missing={len(info['missing'])}, unexpected={len(info['unexpected'])})")
    print(f"使用阈值: {thr:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    if save_prob_map:
        os.makedirs(os.path.join(output_dir, 'prob_maps'), exist_ok=True)
    if save_binary_mask:
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    if save_overlay:
        os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)

    def run_path(img_path, pbar=None):
        basename = os.path.basename(img_path)
        name = os.path.splitext(basename)[0]
        if pbar is not None:
            pbar.set_postfix({'当前': basename})

        img = Image.open(img_path)
        x, pad, orig = preprocess(img, img_size)
        x = x.to(device)

        with torch.no_grad():
            feat = enc(x)
            logits = head_logits(head(feat, (x.shape[2], x.shape[3])))
            prob = torch.sigmoid(logits)

        prob_map = postprocess(prob, pad, orig)
        prob_array = np.array(prob_map)

        if save_prob_map:
            prob_map.save(os.path.join(output_dir, 'prob_maps', f'{name}_prob.png'))
        if save_binary_mask:
            mask = (prob_array > int(thr * 255)).astype(np.uint8) * 255
            Image.fromarray(mask, 'L').save(os.path.join(output_dir, 'masks', f'{name}.png'))
        if save_overlay:
            mask_binary = (prob_array > int(thr * 255)).astype(np.uint8) * 255
            create_overlay(img, mask_binary).save(
                os.path.join(output_dir, 'overlays', f'{name}_overlay.png')
            )

    print("\n开始推理...")
    if single_image:
        if not os.path.exists(single_image):
            print(f"错误: 图像文件不存在: {single_image}")
            return
        print(f"处理图像: {os.path.basename(single_image)}")
        run_path(single_image)
        print(f"\n✓ 推理完成! 结果保存在: {output_dir}")
    elif images_dir:
        if not os.path.exists(images_dir):
            print(f"错误: 图像目录不存在: {images_dir}")
            return
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        if not image_files:
            print(f"警告: 在 {images_dir} 中未找到图像文件")
            return
        print(f"找到 {len(image_files)} 张图像")
        with tqdm(image_files, desc='推理进度', ncols=100, unit='张') as pbar:
            for f in pbar:
                run_path(os.path.join(images_dir, f), pbar)
        print(f"\n✓ 批量推理完成! 结果保存在: {output_dir}")
        if save_binary_mask:
            print(f"  - 二值掩码: {os.path.join(output_dir, 'masks')}")
        if save_prob_map:
            print(f"  - 概率图:   {os.path.join(output_dir, 'prob_maps')}")
        if save_overlay:
            print(f"  - 叠加图:   {os.path.join(output_dir, 'overlays')}")
    else:
        print("错误: 必须指定 single_image 或 images_dir")


def main():
    parser = argparse.ArgumentParser(description='SAM-ATL 推理脚本')
    parser.add_argument('--config', type=str, default='config_infer.yaml',
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
    run_inference(config)


if __name__ == '__main__':
    main()
