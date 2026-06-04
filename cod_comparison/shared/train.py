"""
Shared Training Script — COD Comparison Study
===============================================
Used by all 3 models. Import model from the correct folder.

Fair comparison ensured by:
  - Same hyperparameters
  - Same loss (CODLoss from shared/losses.py)
  - Same metrics (CODMetrics from shared/metrics.py)
  - Same dataset loader (COD10KDataset from shared/dataset.py)
  - Same random seed
  - Same 100 epochs
  - Same LR schedule
"""

import argparse
import sys
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim

# ── paths — caller adds shared/ and model dir to sys.path ────────────────────
from dataset import get_dataloaders
from losses  import CODLoss
from metrics import CODMetrics
from model   import CODModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',    required=True)
    p.add_argument('--save_dir',     required=True)
    p.add_argument('--model_name',   required=True,
                   help='resnet_mamba | resnet_transformer | swin_mamba')
    p.add_argument('--gpu',          type=int,   default=0,
                   help='CUDA device index for this model')
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--img_size',     type=int,   default=384)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--resume',       default=None)
    p.add_argument('--no_pretrain',  action='store_true')
    # Swin-specific
    p.add_argument('--swin_weights', default=None)
    return p.parse_args()


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f'  Saved → {path}')


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt    = torch.load(path, map_location='cpu', weights_only=False)
    saved   = ckpt.get('model', ckpt)
    current = model.state_dict()
    matched, skipped = {}, []
    for k, v in saved.items():
        if k in current and current[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)
    current.update(matched)
    model.load_state_dict(current)
    pct = 100 * len(matched) / max(len(saved), 1)
    print(f'  Loaded {len(matched)}/{len(saved)} keys ({pct:.0f}%)')
    if skipped:
        print(f'  Skipped {len(skipped)} keys')
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
    except Exception as e:
        print(f'  Optimizer not restored ({e})')
    return ckpt.get('epoch', 0), ckpt.get('best_sm', 0.0)


def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    log_every  = max(1, len(loader) // 5)
    t0 = time.time()
    for i, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device,  non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss, comps = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if (i+1) % log_every == 0:
            print(f'  Ep{epoch} [{i+1}/{len(loader)}]  '
                  f'loss={comps["total"]:.4f}  bce={comps["bce"]:.4f}  '
                  f'iou={comps["iou"]:.4f}  bnd={comps["bnd_fine"]:.4f}  '
                  f'bnd_c={comps["bnd_coarse"]:.4f}  ({time.time()-t0:.1f}s)')
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    metrics    = CODMetrics()
    total_loss = 0.0
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device,  non_blocking=True)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss, _ = criterion(outputs, masks)
        total_loss += loss.item()
        metrics.update(outputs['seg'], masks)
    results = metrics.compute()
    results['loss'] = total_loss / len(loader)
    return results


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'  Model : {args.model_name}')
    print(f'  Device: {device}')
    print(f'{"="*60}\n')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(
        root=args.data_root, size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers)

    # Build model — kwargs passed for flexibility across architectures
    model = CODModel(
        pretrained=not args.no_pretrain,
        img_size=args.img_size,
        swin_weights=args.swin_weights,
        mamba_depth=4, mamba_d_state=16,
        transformer_depth=4, transformer_heads=8,
    ).to(device)

    model.freeze_backbone()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    backbone_params = list(model.backbone.parameters())
    backbone_ids    = {id(p) for p in backbone_params}

    # hires_boundary exists only in Swin_Mamba — handle gracefully
    hires_params = []
    if hasattr(model, 'hires_bnd'):
        hires_params = list(model.hires_bnd.parameters())
    elif hasattr(model, 'hires_boundary'):
        hires_params = list(model.hires_boundary.parameters())
    hires_ids = {id(p) for p in hires_params}

    head_params = [p for p in model.parameters()
                   if p.requires_grad
                   and id(p) not in backbone_ids
                   and id(p) not in hires_ids]
    backbone_trainable = [p for p in backbone_params if p.requires_grad]

    optimizer = optim.AdamW([
        {'params': backbone_trainable, 'lr': args.lr * 0.1},
        {'params': head_params,        'lr': args.lr},
        {'params': hires_params,       'lr': args.lr * 5},
    ], weight_decay=1e-4)

    # ── LR Schedule: warmup 5 epochs then cosine decay ────────────────────────
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        progress = (epoch - 5) / max(args.epochs - 5, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = CODLoss()
    scaler    = torch.amp.GradScaler('cuda')

    start_epoch, best_sm = 1, 0.0
    if args.resume:
        start_epoch, best_sm = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler)
        start_epoch += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    train_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = optimizer.param_groups[1]['lr']
        print(f'\n─── [{args.model_name}] Epoch {epoch}/{args.epochs}  lr={lr_now:.2e} ───')
        train_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        scheduler.step()

        do_eval = ((epoch % 2 == 0 and epoch >= 20)
                   or (epoch % 5 == 0)
                   or (epoch == args.epochs))
        if do_eval:
            results = eval_epoch(model, test_loader, criterion, device)
            sm = results.get('Sm', 0.0)
            print(
                f'\n  [Eval] [{args.model_name}]  '
                f'Sm={results.get("Sm",0):.4f}  '
                f'Em_mean={results.get("Em_mean",0):.4f}  '
                f'Em_max={results.get("Em_max",0):.4f}  '
                f'wFm={results.get("wFm",0):.4f}  '
                f'Fm_mean={results.get("Fm_mean",0):.4f}  '
                f'Fm_max={results.get("Fm_max",0):.4f}  '
                f'MAE={results.get("MAE",0):.4f}  '
                f'loss={results.get("loss",0):.4f}'
            )
            if sm > best_sm:
                best_sm = sm
                save_checkpoint({
                    'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_sm': best_sm, 'metrics': results,
                    'model_name': args.model_name,
                    'img_size': args.img_size,
                }, save_dir / 'best_model.pth')
                print(f'  ★ New best Sm={best_sm:.4f}')

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_sm': best_sm,
            }, save_dir / f'epoch_{epoch:03d}.pth')

    total_time = time.time() - train_start
    hours = int(total_time // 3600)
    mins  = int((total_time % 3600) // 60)
    print(f'\n{"="*60}')
    print(f'  [{args.model_name}] Training complete.')
    print(f'  Best Sm     : {best_sm:.4f}')
    print(f'  Total time  : {hours}h {mins}m')
    print(f'  Checkpoints : {save_dir}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()