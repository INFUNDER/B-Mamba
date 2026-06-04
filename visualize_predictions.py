import argparse
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# CLI & Setup
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--cod_root",    default=".")
    p.add_argument("--output_dir",  default="./visualizations")
    p.add_argument("--split",       default="Test")
    p.add_argument("--img_size",    type=int, default=384)
    p.add_argument("--n_samples",   type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default=None)
    return p.parse_args()

# [Functions split_cam_noncam, load_model, preprocess, predict remain same as your logic]
# ... (Keeping your existing logic for data handling and model loading) ...

def split_cam_noncam(img_paths):
    noncam_set = set()
    noncam = []
    for p in img_paths:
        if "NonCAM" in p.stem or "NONCAM" in p.stem.upper():
            noncam.append(p)
            noncam_set.add(p)
    cam = [p for p in img_paths if p not in noncam_set and "CAM" in p.stem.upper()]
    return cam, noncam

def load_model(model_name, ckpt_path, cod_root, img_size, device):
    model_dir  = str(cod_root / model_name)
    shared_dir = str(cod_root / "shared")
    sys.path.insert(0, shared_dir)
    sys.path.insert(0, model_dir)
    for mod_name in list(sys.modules.keys()):
        if mod_name == "model": del sys.modules[mod_name]
    
    from model import CODModel
    model = CODModel(pretrained=False, img_size=img_size, mamba_depth=4, 
                     mamba_d_state=16, transformer_depth=4, transformer_heads=8).to(device)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    sys.path.pop(0); sys.path.pop(0)
    return model

def preprocess(img_path, img_size):
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
           transforms.ToTensor()(img)).unsqueeze(0)

@torch.no_grad()
def predict(model, tensor, device, orig_hw):
    out  = model(tensor.to(device))
    prob = torch.sigmoid(F.interpolate(out["seg"], size=orig_hw, mode="bilinear", align_corners=False))
    prob_np = prob.squeeze().cpu().float().numpy()
    return (prob_np > 0.5).astype(np.uint8) * 255

# ─────────────────────────────────────────────────────────────────────────────
# PAPER-QUALITY FIGURE BUILDER (Clean White Style)
# ─────────────────────────────────────────────────────────────────────────────

COL_TITLES = ["Image", "GT", "ResNet-Mamba", "ResNet-Trans.", "Swin-Mamba"]

def build_figure(rows, title, out_path):
    """
    Creates a clean, white-background figure for academic papers.
    """
    num_rows = len(rows)
    num_cols = 5
    
    # Adjust CELL size for paper spacing (inches)
    CELL_SIZE = 2.0 
    fig, axes = plt.subplots(num_rows, num_cols, 
                             figsize=(num_cols * CELL_SIZE, num_rows * CELL_SIZE),
                             constrained_layout=True)
    
    fig.patch.set_facecolor('white')

    for r, row in enumerate(rows):
        # Prepare image list: Image, GT, Pred1, Pred2, Pred3
        images = [row["orig"], row["gt"]] + row["preds"]
        
        for c, img in enumerate(images):
            ax = axes[r, c]
            
            # Use gray cmap for masks, None for RGB
            if c == 0:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
            
            # Remove all ticks and spines for a clean look
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Column Titles (Top row only)
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=12, fontweight='bold', pad=10)
            
            # Row Labels (Left column only) - Using short name to avoid overlap
            if c == 0:
                display_name = row["name"].split('-')[-1][:15] # Truncate long names
                ax.set_ylabel(display_name, fontsize=10, rotation=90, labelpad=5)

    # Save with high DPI and no extra whitespace
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Execution Logic
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    cod_root, data_root = Path(args.cod_root), Path(args.data_root)
    img_dir, gt_dir = data_root/args.split/"Image", data_root/args.split/"GT_Object"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    cam_imgs, noncam_imgs = split_cam_noncam(all_imgs)

    # Load Models
    models = {}
    for name in ["resnet_mamba", "resnet_transformer", "swin_mamba"]:
        ckpt = cod_root / "checkpoints" / name / "best_model.pth"
        models[name] = load_model(name, ckpt, cod_root, args.img_size, device) if ckpt.exists() else None

    def process_subset(subset, filename, title):
        samples = rng.sample(subset, min(args.n_samples, len(subset)))
        rows = []
        for ip in samples:
            gt_p = gt_dir / (ip.stem + ".png")
            orig = np.array(Image.open(ip).convert("RGB"))
            gt = np.array(Image.open(gt_p).convert("L")) if gt_p.exists() else np.zeros(orig.shape[:2])
            
            tensor = preprocess(ip, args.img_size)
            preds = []
            for name in ["resnet_mamba", "resnet_transformer", "swin_mamba"]:
                if models[name]:
                    preds.append(predict(models[name], tensor, device, orig.shape[:2]))
                else:
                    preds.append(np.zeros(orig.shape[:2], dtype=np.uint8))
            
            rows.append({"orig": orig, "gt": gt, "preds": preds, "name": ip.stem})
        
        build_figure(rows, title, out_dir / filename)

    process_subset(cam_imgs, "comparison_CAM.png", "CAM Results")
    process_subset(noncam_imgs, "comparison_NonCAM.png", "Non-CAM Results")

if __name__ == "__main__":
    main()