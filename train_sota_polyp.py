import os
import subprocess
import random

def set_best_gpu():
    try:
        smi_out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']).decode('utf-8')
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
import torch.nn.functional as F
import torch.nn.functional as F
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='sota')
    parser.add_argument('--no_deep_sup', action='store_true')
    parser.add_argument('--no_boundary', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    # Use dataset/PraNet_Official/TrainDataset for the leak-free official split
    parser.add_argument('--train_root', type=str, default='dataset/PraNet_Split/TrainDataset')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def train_sota():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    batch_size = 8 # slightly larger batch size for stability
    learning_rate = 1e-4
    epochs = args.epochs
    img_size = 352
    
    # Dataset (PraNet Protocol: 900 Kvasir + 550 ClinicDB)
    torch.manual_seed(args.seed)
    data_img_path = os.path.join(args.train_root, 'images') + '/'
    data_gt_path = os.path.join(args.train_root, 'masks') + '/'
    print(f"Training root: {args.train_root}")
    
    print("Loading datasets and creating Train/Val split...")
    dataset_train = PolypDataset(data_img_path, data_gt_path, trainsize=img_size, is_train=True)
    dataset_val = PolypDataset(data_img_path, data_gt_path, trainsize=img_size, is_train=False)
    
    # 90/10 split on the composite 1450 dataset
    indices = list(range(len(dataset_train)))
    random.seed(args.seed)
    random.shuffle(indices)
    
    val_size = int(len(indices) * 0.1)
    train_idx, val_idx = indices[:-val_size], indices[-val_size:]
    
    train_sub = Subset(dataset_train, train_idx)
    val_sub = Subset(dataset_val, val_idx)
    
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model (SOTA Pro Version)
    model = BMambaModel_Pro(
        pretrained=True, 
        img_size=img_size, 
        use_deep_sup=not args.no_deep_sup, 
        use_boundary=not args.no_boundary
    )
    model.to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    metrics = MedicalMetrics(threshold=0.5)
    best_dice = 0.0
    checkpoint_dir = 'polyp_segmentation/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Starting SOTA training pipeline...")
    for epoch in range(1, epochs + 1):
        # TRAIN PASS
        model.train()
        train_loss = 0.0
        
        for i, (images, gts) in enumerate(train_loader):
            images = images.to(device)
            gts = gts.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            
            # Deep Supervision Loss
            if not args.no_deep_sup:
                loss1 = structure_loss(preds['seg'], gts)
                loss2 = structure_loss(preds['seg3'], gts)
                loss3 = structure_loss(preds['seg4'], gts)
                loss = loss1 + loss2 + loss3
            else:
                loss = structure_loss(preds['seg'], gts)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATION PASS
        model.eval()
        metrics.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, gts in val_loader:
                images = images.to(device)
                gts = gts.to(device)
                
                preds = model(images)
                loss = structure_loss(preds['seg'], gts)
                val_loss += loss.item()
                
                metrics.update(preds['seg'], gts)
                
        val_metrics = metrics.get_metrics()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"=== Epoch [{epoch}/{epochs}] ===")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'b_mamba_best_{args.run_name}.pth'))
            print(f"--> Saved new SOTA model with Val Dice: {best_dice:.4f}")
        print("="*30 + "\n")

if __name__ == '__main__':
    train_sota()
