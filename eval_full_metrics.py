"""
Full evaluation for the paper (PraNet protocol, 5 test sets).
=============================================================
Differences from test_pranet_split.py (kept for backward compatibility):
  * Predictions are upsampled to the ORIGINAL ground-truth resolution before
    scoring (the convention used by PraNet / Polyp-PVT evaluation toolboxes).
  * Adds MAE, Boundary IoU (Cheng et al., CVPR 2021) and HD95, so the
    "boundary-aware" claim is backed by boundary-specific metrics.
  * Writes a JSON file that aggregate_results.py turns into LaTeX tables.

Usage:
  python eval_full_metrics.py --run_name clean_full_s42 [--no_boundary] [--no_deep_sup]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro

TEST_SETS = [
    ('Kvasir', 'dataset/TestDataset/Kvasir'),
    ('ClinicDB', 'dataset/TestDataset/CVC-ClinicDB'),
    ('ColonDB', 'dataset/TestDataset/CVC-ColonDB'),
    ('ETIS', 'dataset/TestDataset/ETIS-LaribPolypDB'),
    ('CVC-300', 'dataset/TestDataset/CVC-300'),
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', required=True)
    p.add_argument('--no_deep_sup', action='store_true')
    p.add_argument('--no_boundary', action='store_true')
    p.add_argument('--img_size', type=int, default=352)
    p.add_argument('--out_dir', default='results_json')
    return p.parse_args()


def find_mask(mask_dir, img_name):
    stem = os.path.splitext(img_name)[0]
    for ext in EXTS:
        path = os.path.join(mask_dir, stem + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(img_name)


def boundary_region(mask, d):
    """Pixels of `mask` within distance d of its contour (Boundary IoU definition)."""
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask, iterations=d, border_value=0)
    return mask & ~eroded


def boundary_iou(pred, gt, dilation_ratio=0.02):
    h, w = gt.shape
    d = max(1, int(round(dilation_ratio * np.sqrt(h * h + w * w))))
    gb, pb = boundary_region(gt, d), boundary_region(pred, d)
    union = (gb | pb).sum()
    return 1.0 if union == 0 else float((gb & pb).sum() / union)


def hd95(pred, gt):
    """95th-percentile symmetric Hausdorff distance in pixels (original resolution)."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(np.sqrt(gt.shape[0] ** 2 + gt.shape[1] ** 2))  # worst case: image diagonal
    pc = pred & ~binary_erosion(pred)
    gc = gt & ~binary_erosion(gt)
    dt_g = distance_transform_edt(~gc)
    dt_p = distance_transform_edt(~pc)
    d = np.concatenate([dt_g[pc], dt_p[gc]])
    return float(np.percentile(d, 95))


@torch.no_grad()
def evaluate(model, device, root, img_size):
    img_dir, msk_dir = os.path.join(root, 'images'), os.path.join(root, 'masks')
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(EXTS))
    per_img = {'dice': [], 'iou': [], 'mae': [], 'biou': [], 'hd95': []}
    for name in names:
        img = Image.open(os.path.join(img_dir, name)).convert('RGB')
        gt = np.asarray(Image.open(find_mask(msk_dir, name)).convert('L'), dtype=np.float32) / 255.0
        gt_bin = gt > 0.5

        x = np.asarray(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(device)
        logit = model(x)['seg']
        prob = torch.sigmoid(F.interpolate(logit, size=gt.shape, mode='bilinear', align_corners=False))
        prob = prob.squeeze().cpu().numpy()
        pred_bin = prob > 0.5

        inter = (pred_bin & gt_bin).sum()
        ps, gs = pred_bin.sum(), gt_bin.sum()
        per_img['dice'].append((2 * inter + 1e-5) / (ps + gs + 1e-5))
        per_img['iou'].append((inter + 1e-5) / (ps + gs - inter + 1e-5))
        per_img['mae'].append(float(np.abs(prob - gt).mean()))
        per_img['biou'].append(boundary_iou(pred_bin, gt_bin))
        per_img['hd95'].append(hd95(pred_bin, gt_bin))
    return {k: float(np.mean(v)) for k, v in per_img.items()} | {'n': len(names)}


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = f'polyp_segmentation/checkpoints/b_mamba_best_{args.run_name}.pth'
    model = BMambaModel_Pro(pretrained=False, img_size=args.img_size,
                            use_deep_sup=not args.no_deep_sup, use_boundary=not args.no_boundary)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    results = {}
    for name, root in TEST_SETS:
        r = evaluate(model, device, root, args.img_size)
        results[name] = r
        print(f"{name:9s} n={r['n']:3d}  mDice={r['dice']:.4f}  mIoU={r['iou']:.4f}  "
              f"MAE={r['mae']:.4f}  BIoU={r['biou']:.4f}  HD95={r['hd95']:.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f'{args.run_name}.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
