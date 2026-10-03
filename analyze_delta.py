"""
Mechanistic analysis: does boundary conditioning change the SSM step size Delta at polyp edges?
=============================================================================================
For every BC-SSM block we hook the step size  Delta_t = softplus(W_dt r_t + b)  (mean over channels),
reshape it to the h x w token grid and compare it with the ground-truth boundary density of each token.

Reported per block (averaged over all test images of the 5 datasets):
  * mean Delta on boundary tokens vs. non-boundary tokens and their ratio
  * Spearman rank correlation between Delta and boundary density
  * mean "memory retention" exp(Delta * mean(A)) on boundary vs. non-boundary tokens

Run it on the full model and on the no-boundary model (e.g. clean_deepsup_s42) to compare.
Also saves a qualitative figure to paper/figures/delta_<run>.png.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_full_metrics import EXTS, MEAN, STD, TEST_SETS, find_mask
from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', required=True)
    p.add_argument('--no_boundary', action='store_true')
    p.add_argument('--no_deep_sup', action='store_true')
    p.add_argument('--img_size', type=int, default=352)
    p.add_argument('--n_fig', type=int, default=6)
    p.add_argument('--fig_block', type=int, default=1, help='1-indexed block shown in the figure')
    return p.parse_args()


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-8))


def gt_boundary_density(gt, grid_hw, width=5):
    """Fraction of GT-boundary pixels inside each token cell."""
    g = torch.from_numpy(gt).float()[None, None]
    k = 2 * width + 1
    dil = F.max_pool2d(g, k, 1, width)
    ero = -F.max_pool2d(-g, k, 1, width)
    bnd = (dil - ero).clamp(0, 1)
    return F.adaptive_avg_pool2d(bnd, grid_hw)[0, 0]


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BMambaModel_Pro(pretrained=False, img_size=args.img_size,
                            use_deep_sup=not args.no_deep_sup, use_boundary=not args.no_boundary)
    ckpt = f'polyp_segmentation/checkpoints/b_mamba_best_{args.run_name}.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    blocks = model.b_mamba.blocks
    captured = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            captured[i] = F.softplus(out)  # (B, L, D): per-channel Delta
        return hook

    for i, blk in enumerate(blocks):
        blk.dt_proj.register_forward_hook(make_hook(i))
    # Slowest decay rate per channel (closest to 0): this state carries long-range memory.
    a_slow = [(-torch.exp(b.A_log)).max(dim=1).values.to(device) for b in blocks]  # (D,)

    nb = len(blocks)
    stats = {i: {'d_b': [], 'd_n': [], 'rho': [], 'r_b': [], 'r_n': []} for i in range(nb)}
    fig_items = []
    for ds_name, root in TEST_SETS:
        img_dir, msk_dir = os.path.join(root, 'images'), os.path.join(root, 'masks')
        for name in sorted(f for f in os.listdir(img_dir) if f.lower().endswith(EXTS)):
            img = Image.open(os.path.join(img_dir, name)).convert('RGB')
            gt = (np.asarray(Image.open(find_mask(msk_dir, name)).convert('L').resize(
                (args.img_size, args.img_size), Image.NEAREST)) > 127).astype(np.float32)
            x = np.asarray(img.resize((args.img_size, args.img_size), Image.BILINEAR), np.float32) / 255.0
            x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))[None].to(device)
            model(x)
            L = captured[0].shape[1]
            h = w = int(round(L ** 0.5))
            dens = gt_boundary_density(gt, (h, w)).flatten().to(device)
            is_b = dens > 0.05
            if is_b.sum() == 0 or (~is_b).sum() == 0:
                continue
            for i in range(nb):
                d_full = captured[i][0]                    # (L, D)
                d = d_full.mean(-1)                        # (L,)
                ret = torch.exp(d_full * a_slow[i]).mean(-1)  # slowest-state retention per token
                stats[i]['d_b'].append(d[is_b].mean().item())
                stats[i]['d_n'].append(d[~is_b].mean().item())
                stats[i]['r_b'].append(ret[is_b].mean().item())
                stats[i]['r_n'].append(ret[~is_b].mean().item())
                stats[i]['rho'].append(spearman(d, dens))
            if len(fig_items) < args.n_fig and ds_name in ('ColonDB', 'ETIS'):
                fig_items.append((np.asarray(img.resize((args.img_size, args.img_size))), gt,
                                  captured[args.fig_block - 1][0].mean(-1).reshape(h, w).cpu().numpy()))

    print(f'=== Delta analysis: {args.run_name} (boundary conditioning: {not args.no_boundary}) ===')
    print('block | Delta(bnd) | Delta(non) | ratio | retention(bnd) | retention(non) | Spearman rho')
    for i in range(nb):
        s = {k: np.mean(v) for k, v in stats[i].items()}
        print(f'{i + 1:5d} | {s["d_b"]:10.4f} | {s["d_n"]:10.4f} | {s["d_b"] / s["d_n"]:5.3f} | '
              f'{s["r_b"]:14.4f} | {s["r_n"]:14.4f} | {s["rho"]:+.3f} (n={len(stats[i]["rho"])})')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(fig_items)
        fig, ax = plt.subplots(3, n, figsize=(2.2 * n, 6.6))
        for j, (im, g, dmap) in enumerate(fig_items):
            ax[0, j].imshow(im)
            ax[1, j].imshow(g, cmap='gray')
            up = np.asarray(Image.fromarray(dmap.astype(np.float32)).resize(im.shape[1::-1], Image.BICUBIC))
            ax[2, j].imshow(im)
            ax[2, j].imshow(up, cmap='inferno', alpha=0.6)
            for r in range(3):
                ax[r, j].axis('off')
        for r, t in enumerate(['Image', 'Ground truth', 'Learned $\\Delta$ (block %d)' % args.fig_block]):
            ax[r, 0].set_title(t, fontsize=9, loc='left')
        os.makedirs('paper/figures', exist_ok=True)
        out = f'paper/figures/delta_{args.run_name}.png'
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        print(f'Saved {out}')
    except ImportError:
        print('matplotlib not available - skipped figure.')


if __name__ == '__main__':
    main()
