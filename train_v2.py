"""
Training recipe v2 (generalisation-focused). Original train_sota_polyp.py is untouched so
all earlier ablations stay reproducible.

Changes vs v1:
  * --strong_aug      colour jitter / blur / random scale-translate-rotate (aligned image+mask)
  * layer-wise LR     pretrained Swin backbone uses --backbone_lr, everything else --lr
  * grad clipping     --clip 1.0
  * EMA weights       evaluated/saved instead of raw weights (smoother, generalises better)
  * --full_data       train on ALL images (no 10% hold-out); saves the FINAL EMA weights
                      (epoch count should be chosen from a normal run with validation)
"""
import os
import copy
import random
import argparse
import subprocess


def set_best_gpu():
    try:
        smi_out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                           '--format=csv,nounits,noheader']).decode('utf-8')
        free_memory = [int(x) for x in smi_out.strip().split('\n')]
        best_gpu = free_memory.index(max(free_memory))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
        print(f"Auto-selected GPU {best_gpu} with {max(free_memory)} MB free memory.")
    except Exception as e:
        print(f"Could not auto-select GPU: {e}")


set_best_gpu()

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from polyp_segmentation.shared.dataset_polyp import PolypDataset
from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro
from polyp_segmentation.shared.metrics_polyp import structure_loss, MedicalMetrics
from polyp_segmentation.b_mamba.b_mamba_v3 import BMambaModelV3, boundary_target, boundary_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', type=str, default='v2')
    p.add_argument('--no_deep_sup', action='store_true')
    p.add_argument('--no_boundary', action='store_true')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--train_root', type=str, default='dataset/PraNet_Official/TrainDataset')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--strong_aug', action='store_true')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone_lr', type=float, default=2e-5)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--full_data', action='store_true')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--arch', choices=['pro', 'v3'], default='pro')
    p.add_argument('--edge_w', type=float, default=1.0, help='weight of the auxiliary boundary loss (v3)')
    return p.parse_args()


@torch.no_grad()
def ema_update(ema_model, model, decay):
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        if v.dtype.is_floating_point:
            v.mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
        else:
            v.copy_(msd[k])


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 352
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    img_path = os.path.join(args.train_root, 'images') + '/'
    gt_path = os.path.join(args.train_root, 'masks') + '/'
    ds_train = PolypDataset(img_path, gt_path, trainsize=img_size, is_train=True, strong_aug=args.strong_aug)
    ds_val = PolypDataset(img_path, gt_path, trainsize=img_size, is_train=False)

    idx = list(range(len(ds_train)))
    random.shuffle(idx)
    if args.full_data:
        train_idx, val_idx = idx, idx[:max(1, int(len(idx) * 0.1))]  # val only for monitoring (seen data)
        print(f"FULL DATA mode: training on all {len(train_idx)} images; final EMA weights are saved.")
    else:
        v = int(len(idx) * 0.1)
        train_idx, val_idx = idx[:-v], idx[-v:]

    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(ds_val, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    if args.arch == 'v3':
        model = BMambaModelV3(pretrained=True, img_size=img_size,
                              use_deep_sup=not args.no_deep_sup,
                              use_boundary=not args.no_boundary).to(device)
    else:
        model = BMambaModel_Pro(pretrained=True, img_size=img_size,
                                use_deep_sup=not args.no_deep_sup,
                                use_boundary=not args.no_boundary).to(device)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)

    bb = [p_ for n, p_ in model.named_parameters() if n.startswith('backbone.')]
    rest = [p_ for n, p_ in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = optim.AdamW([{'params': bb, 'lr': args.backbone_lr},
                             {'params': rest, 'lr': args.lr}], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    metrics = MedicalMetrics(threshold=0.5)
    best_dice = 0.0
    ckpt_dir = 'polyp_segmentation/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(ckpt_dir, f'b_mamba_best_{args.run_name}.pth')

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for images, gts in train_loader:
            images, gts = images.to(device), gts.to(device)
            optimizer.zero_grad()
            preds = model(images)
            if not args.no_deep_sup:
                loss = (structure_loss(preds['seg'], gts) + structure_loss(preds['seg3'], gts)
                        + structure_loss(preds['seg4'], gts))
            else:
                loss = structure_loss(preds['seg'], gts)
            if args.arch == 'v3':
                loss = loss + args.edge_w * boundary_loss(preds['edge'], boundary_target(gts))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            ema_update(ema, model, args.ema_decay)
            tot += loss.item()
        scheduler.step()

        metrics.reset()
        with torch.no_grad():
            for images, gts in val_loader:
                metrics.update(ema(images.to(device))['seg'], gts.to(device))
        vm = metrics.get_metrics()
        print(f"=== Epoch [{epoch}/{args.epochs}] === train loss {tot / len(train_loader):.4f} | "
              f"EMA val Dice {vm['dice']:.4f} IoU {vm['iou']:.4f}", flush=True)

        if args.full_data:
            if epoch == args.epochs:
                torch.save(ema.state_dict(), out_path)
                print(f"--> Saved FINAL EMA weights to {out_path}")
        elif vm['dice'] > best_dice:
            best_dice = vm['dice']
            torch.save(ema.state_dict(), out_path)
            print(f"--> Saved new best EMA model, val Dice {best_dice:.4f}")


if __name__ == '__main__':
    main()
