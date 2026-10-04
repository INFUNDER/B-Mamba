import os
import torch
import matplotlib.pyplot as plt
from polyp_segmentation.shared.dataset_polyp import PolypDataset
from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro
import numpy as np

def load_model(ckpt_path, no_boundary=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BMambaModel_Pro(pretrained=False, img_size=352, use_boundary=(not no_boundary))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_s6 = load_model('polyp_segmentation/checkpoints/b_mamba_best_clean_deepsup_s42.pth', no_boundary=True)
    model_bc = load_model('polyp_segmentation/checkpoints/b_mamba_best_clean_full_s42.pth', no_boundary=False)
    
    datasets = [
        ('ColonDB', 'dataset/TestDataset/CVC-ColonDB/images/', 'dataset/TestDataset/CVC-ColonDB/masks/'),
        ('ETIS', 'dataset/TestDataset/ETIS-LaribPolypDB/images/', 'dataset/TestDataset/ETIS-LaribPolypDB/masks/')
    ]
    
    samples = []
    
    for name, img_root, gt_root in datasets:
        dataset = PolypDataset(img_root, gt_root, trainsize=352)
        # Select 2 images from each dataset
        for idx in [20, 25]:
            img, gt = dataset[idx]
            samples.append((name, img, gt))
            
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, (name, img_tensor, gt_tensor) in enumerate(samples):
        img_input = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_s6 = torch.sigmoid(model_s6(img_input)['seg']).squeeze().cpu().numpy()
            pred_bc = torch.sigmoid(model_bc(img_input)['seg']).squeeze().cpu().numpy()
            
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(np.array([0.229, 0.224, 0.225]) * img_np + np.array([0.485, 0.456, 0.406]), 0, 1)
        gt_np = gt_tensor.squeeze().numpy()
        
        axs[i, 0].imshow(img_np)
        axs[i, 0].set_ylabel(name, fontsize=14)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        
        axs[i, 1].imshow(gt_np, cmap='gray')
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(pred_s6, cmap='gray')
        axs[i, 2].axis('off')
        
        axs[i, 3].imshow(pred_bc, cmap='gray')
        axs[i, 3].axis('off')
        
        if i == 0:
            axs[i, 0].set_title('Image', fontsize=14)
            axs[i, 1].set_title('GT', fontsize=14)
            axs[i, 2].set_title('S6 (BC off)', fontsize=14)
            axs[i, 3].set_title('B-Mamba', fontsize=14)
            
    plt.tight_layout()
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/qualitative.png', bbox_inches='tight', dpi=300)
    print("Saved to paper/figures/qualitative.png")

if __name__ == '__main__':
    visualize()
