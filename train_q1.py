"""
Training Script for B-Mamba 2D (Q1 Journal Architecture)
=======================================================
Incorporates:
  * Multi-scale 4-way cross-scan boundary conditioning
  * Supervised boundary loss with morphological gradient targets
  * Multi-scale deep supervision (P2, P3, P4)
  * Layer-wise backbone learning rate decay
  * Exponential Moving Average (EMA) weight consolidation
  * Strong photometric and geometric data augmentation
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
from polyp_segmentation.shared.metrics_polyp import structure_loss, MedicalMetrics
from polyp_segmentation.b_mamba.b_mamba_2d import BMamba2D, boundary_target, boundary_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', type=str, required=True)
    p.add_argument('--no_boundary', action='store_true', help='Disable e_t boundary conditioning in SSM')
    p.add_argument('--no_bpe_bypass', action='store_true', help='Disable raw F_r boundary feature bypass')
    p.add_argument('--no_deep_sup', action='store_true', help='Disable deep supervision')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--train_root', type=str, default='dataset/PraNet_Official/TrainDataset')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--strong_aug', action='store_true', default=True)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone_lr', type=float, default=2e-5)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--edge_w', type=float, default=1.0, help='Weight for auxiliary boundary loss')
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
    v = int(len(idx) * 0.1) # 10% hold-out validation
    train_idx, val_idx = idx[:-v], idx[-v:]

    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(ds_val, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = BMamba2D(pretrained=True, img_size=img_size,
                     use_boundary=not args.no_boundary,
                     use_bpe_bypass=not args.no_bpe_bypass,
                     use_deep_sup=not args.no_deep_sup).to(device)
                     
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)

    bb = [p_ for n, p_ in model.named_parameters() if n.startswith('swin.')]
    rest = [p_ for n, p_ in model.named_parameters() if not n.startswith('swin.')]
    optimizer = optim.AdamW([{'params': bb, 'lr': args.backbone_lr},
                             {'params': rest, 'lr': args.lr}], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    metrics = MedicalMetrics(threshold=0.5)
    best_dice = 0.0
    ckpt_dir = 'polyp_segmentation/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(ckpt_dir, f'b_mamba_best_{args.run_name}.pth')

    print(f"=== Starting Training {args.run_name} (Seed {args.seed}) ===")
    print(f"Flags: use_boundary={not args.no_boundary}, use_bpe_bypass={not args.no_bpe_bypass}, use_deep_sup={not args.no_deep_sup}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        tot_seg = 0.0
        tot_edge = 0.0
        
        for images, gts in train_loader:
            images, gts = images.to(device), gts.to(device)
            optimizer.zero_grad()
            preds = model(images)
            
            # Segmentation loss
            if not args.no_deep_sup:
                loss_seg = (structure_loss(preds['seg'], gts) +
                            structure_loss(preds['seg3'], gts) +
                            structure_loss(preds['seg4'], gts))
            else:
                loss_seg = structure_loss(preds['seg'], gts)
                
            # Boundary loss with morphological gradient target
            target_edge = boundary_target(gts)
            loss_edge = args.edge_w * boundary_loss(preds['edge'], target_edge)
            
            loss = loss_seg + loss_edge
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            ema_update(ema, model, args.ema_decay)
            
            tot_loss += loss.item()
            tot_seg += loss_seg.item()
            tot_edge += loss_edge.item()
            
        scheduler.step()

        # Validation with EMA weights
        metrics.reset()
        with torch.no_grad():
            for images, gts in val_loader:
                metrics.update(ema(images.to(device))['seg'], gts.to(device))
        vm = metrics.get_metrics()
        
        n_batches = len(train_loader)
        print(f"Epoch [{epoch:3d}/{args.epochs}] | Loss: {tot_loss/n_batches:.4f} (Seg: {tot_seg/n_batches:.4f}, Edge: {tot_edge/n_batches:.4f}) | Val mDice: {vm['dice']:.4f} mIoU: {vm['iou']:.4f}", flush=True)

        if vm['dice'] > best_dice:
            best_dice = vm['dice']
            torch.save(ema.state_dict(), out_path)
            print(f"  --> Saved new best EMA model (Val mDice: {best_dice:.4f}) to {out_path}", flush=True)

    print(f"Training complete! Best Val mDice: {best_dice:.4f}")

if __name__ == '__main__':
    main()
