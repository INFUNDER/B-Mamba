import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

def load_model(model_name, ckpt_path, cod_root, img_size, device):
    model_dir  = str(cod_root / model_name)
    shared_dir = str(cod_root / "shared")
    
    # insert paths to sys.path
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
        
    for mod_name in list(sys.modules.keys()):
        if mod_name == "model": 
            del sys.modules[mod_name]
    
    from model import CODModel
    model = CODModel(pretrained=False, img_size=img_size, mamba_depth=4, 
                     mamba_d_state=16, transformer_depth=4, transformer_heads=8).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    
    # Clean up sys.path
    sys.path.remove(shared_dir)
    sys.path.remove(model_dir)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cod_root = Path("/home/ronit.28010/COD_Project/cod_comparison")
    data_root = Path("/home/ronit.28010/COD_Project/dataset/COD10K")
    
    sys.path.insert(0, str(cod_root / "shared"))
    from dataset import get_dataloaders
    from metrics import CODMetrics
    sys.path.pop(0)
    
    _, test_loader = get_dataloaders(root=data_root, size=384, batch_size=1, num_workers=4)
    
    models = ["resnet_transformer", "resnet_mamba", "swin_mamba"]
    
    for model_name in models:
        ckpt_path = cod_root / "checkpoints" / model_name / "best_model.pth"
        if not ckpt_path.exists():
            print(f"Checkpoint for {model_name} not found at {ckpt_path}. Skipping.")
            continue
            
        print(f"\nEvaluating model: {model_name}")
        model = load_model(model_name, ckpt_path, cod_root, img_size=384, device=device)
        
        metrics = CODMetrics()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Run forward pass (all models output a dict with 'seg')
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                # Update metrics
                metrics.update(outputs['seg'], masks)
                
                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1}/{len(test_loader)} samples...")
                    
        results = metrics.compute()
        print(f"Results for {model_name}:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
