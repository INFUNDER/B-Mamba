import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

def main():
    cod_root = Path("/home/ronit.28010/COD_Project/cod_comparison")
    data_root = Path("/home/ronit.28010/COD_Project/dataset/COD10K")
    
    sys.path.insert(0, str(cod_root / "shared"))
    sys.path.insert(0, str(cod_root / "resnet_transformer"))
    from dataset import get_dataloaders
    from metrics import CODMetrics
    from model import CODModel
    
    device = torch.device("cpu")
    model = CODModel(pretrained=False, img_size=384, mamba_depth=4, 
                     mamba_d_state=16, transformer_depth=4, transformer_heads=8).to(device)
    
    ckpt_path = cod_root / "checkpoints" / "resnet_transformer" / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    
    _, test_loader = get_dataloaders(root=data_root, size=384, batch_size=1, num_workers=1)
    
    metrics = CODMetrics()
    
    for i, batch in enumerate(test_loader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        pred_logits = outputs['seg']
        pred = torch.sigmoid(pred_logits).squeeze(1).cpu().numpy()[0]
        gt = masks.squeeze(1).cpu().numpy()[0]
        
        # Calculate stats for this one sample
        gt_b = (gt > 0.5).astype(np.float32)
        pred_b = (pred >= 0.5).astype(np.float32)
        
        tp = (pred_b * gt_b).sum()
        fp = (pred_b * (1 - gt_b)).sum()
        fn = ((1 - pred_b) * gt_b).sum()
        
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f_score = 1.3 * prec * rec / (0.3 * prec + rec + 1e-8)
        
        print(f"Sample {i}:")
        print(f"  gt sum: {gt_b.sum()}")
        print(f"  pred >= 0.5 sum: {pred_b.sum()}")
        print(f"  tp: {tp}, fp: {fp}, fn: {fn}")
        print(f"  prec: {prec:.4f}, rec: {rec:.4f}, f_score: {f_score:.4f}")
        print(f"  Sm: {metrics._s_measure(pred, gt):.4f}")
        print(f"  wFm: {metrics._weighted_f(pred, gt):.4f}")
        print(f"  MAE: {metrics._mae(pred, gt):.4f}")
        break

if __name__ == "__main__":
    main()
