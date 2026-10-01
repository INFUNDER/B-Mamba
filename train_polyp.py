import os
import torch
import torch.optim as optim
from polyp_segmentation.shared.dataset_polyp import get_loader
from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel
from polyp_segmentation.shared.metrics_polyp import compute_dice_loss, MedicalMetrics

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    batch_size = 4
    learning_rate = 1e-4
    epochs = 10
    img_size = 352
    
    # Dataset
    train_img_path = 'dataset/Kvasir-SEG/images/'
    train_gt_path = 'dataset/Kvasir-SEG/masks/'
    
    # Ensure directories exist
    if not os.path.exists(train_img_path):
        print(f"Error: {train_img_path} not found.")
        return
        
    print("Loading dataset...")
    train_loader = get_loader(train_img_path, train_gt_path, batchsize=batch_size, trainsize=img_size)
    print(f"Dataset size: {len(train_loader.dataset)} images")
    
    # Model
    model = BMambaModel(pretrained=True, img_size=img_size)
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Metrics Tracker
    metrics = MedicalMetrics(threshold=0.5)
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        metrics.reset()
        
        for i, (images, gts) in enumerate(train_loader):
            images = images.to(device)
            gts = gts.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(images)
            seg_pred = preds['seg']
            
            # Compute loss
            loss = compute_dice_loss(seg_pred, gts)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            metrics.update(seg_pred, gts)
            
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        train_metrics = metrics.get_metrics()
        print(f"=== Epoch {epoch} Summary ===")
        print(f"Avg Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Dice Score (DSC): {train_metrics['dice']:.4f}")
        print(f"IoU Score: {train_metrics['iou']:.4f}")
        print("=============================\n")

if __name__ == '__main__':
    train()
