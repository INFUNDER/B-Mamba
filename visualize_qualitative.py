"""
Generates Figure 3: Qualitative comparison on challenging test images
from unseen CVC-ColonDB and ETIS-Larib datasets.
Columns:
  1. Input Image
  2. Ground Truth Mask
  3. S6 Baseline (Pure S6, BC off)
  4. B-Mamba 2D (Ours)
  5. Multi-Contour Overlay (Green: GT, Red: B-Mamba 2D, Yellow: S6 Baseline)
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from polyp_segmentation.b_mamba.b_mamba_2d import BMamba2D

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_models(device):
    # Pure S6 Baseline
    ckpt_s6 = 'polyp_segmentation/checkpoints/b_mamba_best_q1_baseline_s42.pth'
    m_s6 = BMamba2D(pretrained=False, img_size=352, use_boundary=False, use_bpe_bypass=False)
    m_s6.load_state_dict(torch.load(ckpt_s6, map_location=device))
    m_s6.to(device).eval()

    # B-Mamba 2D (Full)
    ckpt_bc = 'polyp_segmentation/checkpoints/b_mamba_best_q1_full_s42.pth'
    m_bc = BMamba2D(pretrained=False, img_size=352, use_boundary=True, use_bpe_bypass=True)
    m_bc.load_state_dict(torch.load(ckpt_bc, map_location=device))
    m_bc.to(device).eval()

    return m_s6, m_bc

def get_contours(mask_bin):
    m = (mask_bin > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating qualitative visualizations on {device}...")
    m_s6, m_bc = load_models(device)

    # 4 challenging samples showing diverse clinical topologies
    test_cases = [
        ('ColonDB #250', 'dataset/TestDataset/CVC-ColonDB/images/250.png', 'dataset/TestDataset/CVC-ColonDB/masks/250.png'),
        ('ColonDB #20',  'dataset/TestDataset/CVC-ColonDB/images/20.png',  'dataset/TestDataset/CVC-ColonDB/masks/20.png'),
        ('ETIS #110',    'dataset/TestDataset/ETIS-LaribPolypDB/images/110.png', 'dataset/TestDataset/ETIS-LaribPolypDB/masks/110.png'),
        ('ETIS #15',     'dataset/TestDataset/ETIS-LaribPolypDB/images/15.png',  'dataset/TestDataset/ETIS-LaribPolypDB/masks/15.png'),
    ]

    fig, axs = plt.subplots(4, 5, figsize=(15, 12.5))
    plt.subplots_adjust(wspace=0.03, hspace=0.10)

    for row_idx, (label, img_path, msk_path) in enumerate(test_cases):
        if not os.path.exists(img_path) or not os.path.exists(msk_path):
            print(f"Warning: {img_path} or {msk_path} missing.")
            continue
            
        pil_img = Image.open(img_path).convert('RGB')
        pil_gt  = Image.open(msk_path).convert('L')

        # Preprocess for model input (352x352)
        img_resized = np.asarray(pil_img.resize((352, 352), Image.BILINEAR), dtype=np.float32) / 255.0
        x_tensor = torch.from_numpy(((img_resized - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Ground truth at 352x352
        gt_resized = np.asarray(pil_gt.resize((352, 352), Image.NEAREST), dtype=np.float32) / 255.0
        gt_bin = gt_resized > 0.5

        with torch.no_grad():
            out_s6 = m_s6(x_tensor)['seg']
            out_bc = m_bc(x_tensor)['seg']
            pred_s6 = torch.sigmoid(out_s6).squeeze().cpu().numpy()
            pred_bc = torch.sigmoid(out_bc).squeeze().cpu().numpy()

        p_s6_bin = pred_s6 > 0.5
        p_bc_bin = pred_bc > 0.5

        dice_s6 = (2.0 * (p_s6_bin & gt_bin).sum() + 1e-6) / (p_s6_bin.sum() + gt_bin.sum() + 1e-6)
        dice_bc = (2.0 * (p_bc_bin & gt_bin).sum() + 1e-6) / (p_bc_bin.sum() + gt_bin.sum() + 1e-6)

        # Col 0: Input Image
        axs[row_idx, 0].imshow(img_resized)
        axs[row_idx, 0].set_ylabel(label, fontsize=12, fontweight='bold')
        axs[row_idx, 0].set_xticks([])
        axs[row_idx, 0].set_yticks([])

        # Col 1: Ground Truth
        axs[row_idx, 1].imshow(gt_bin, cmap='gray')
        axs[row_idx, 1].axis('off')

        # Col 2: S6 Baseline
        axs[row_idx, 2].imshow(p_s6_bin, cmap='gray')
        axs[row_idx, 2].set_xlabel(f'Dice: {dice_s6:.3f}', fontsize=11, fontweight='semibold', color='#555555', labelpad=4)
        axs[row_idx, 2].set_xticks([])
        axs[row_idx, 2].set_yticks([])

        # Col 3: B-Mamba 2D
        axs[row_idx, 3].imshow(p_bc_bin, cmap='gray')
        diff_str = f'+{dice_bc - dice_s6:.3f}' if dice_bc >= dice_s6 else f'{dice_bc - dice_s6:.3f}'
        axs[row_idx, 3].set_xlabel(f'Dice: {dice_bc:.3f} ({diff_str})', fontsize=11, fontweight='bold', color='#B20000', labelpad=4)
        axs[row_idx, 3].set_xticks([])
        axs[row_idx, 3].set_yticks([])

        # Col 4: Contour Overlay
        axs[row_idx, 4].imshow(img_resized)
        
        # Green: Ground Truth (drawn solid with high visibility)
        for c in get_contours(gt_bin):
            c_sq = c.squeeze()
            if len(c_sq.shape) == 2 and len(c_sq) > 3:
                axs[row_idx, 4].plot(c_sq[:, 0], c_sq[:, 1], color='#00EE00', linewidth=2.2, linestyle='-')

        # Yellow: S6 Baseline (drawn dashed)
        for c in get_contours(p_s6_bin):
            c_sq = c.squeeze()
            if len(c_sq.shape) == 2 and len(c_sq) > 3:
                axs[row_idx, 4].plot(c_sq[:, 0], c_sq[:, 1], color='#FFDD00', linewidth=2.0, linestyle='--')
                
        # Red: B-Mamba 2D (drawn solid)
        for c in get_contours(p_bc_bin):
            c_sq = c.squeeze()
            if len(c_sq.shape) == 2 and len(c_sq) > 3:
                axs[row_idx, 4].plot(c_sq[:, 0], c_sq[:, 1], color='#FF2222', linewidth=2.0, linestyle='-')

        axs[row_idx, 4].set_xticks([])
        axs[row_idx, 4].set_yticks([])

        if row_idx == 0:
            axs[0, 0].set_title('(a) Input Image', fontsize=13, fontweight='bold', pad=8)
            axs[0, 1].set_title('(b) Ground Truth', fontsize=13, fontweight='bold', pad=8)
            axs[0, 2].set_title('(c) S6 Baseline', fontsize=13, fontweight='bold', pad=8)
            axs[0, 3].set_title('(d) B-Mamba 2D (Ours)', fontsize=13, fontweight='bold', pad=8)
            axs[0, 4].set_title('(e) Overlay Contours', fontsize=13, fontweight='bold', pad=8)

    # Global legend for contours at bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00EE00', lw=2.5, linestyle='-', label='Ground Truth'),
        Line2D([0], [0], color='#FF2222', lw=2.5, linestyle='-', label='B-Mamba 2D (Ours)'),
        Line2D([0], [0], color='#FFDD00', lw=2.5, linestyle='--', label='S6 Baseline (No BC)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc',
               bbox_to_anchor=(0.5, 0.015))

    out_file = 'paper/figures/qualitative.png'
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Successfully generated qualitative figure: {out_file}")

if __name__ == '__main__':
    main()
