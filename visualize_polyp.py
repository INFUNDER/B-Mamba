import os
import torch
import matplotlib.pyplot as plt
from polyp_segmentation.shared.dataset_polyp import PolypDataset
from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro
import numpy as np

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_size = 352
    
    # Load Model
    model_path = 'polyp_segmentation/checkpoints/b_mamba_best_sota.pth'
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained model at {model_path}. Please complete training first.")
        return
        
    model = BMambaModel_Pro(pretrained=False, img_size=img_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Dataset (we'll use a few samples from Kvasir-SEG for demonstration, 
    # but ideally this should point to CVC-ClinicDB for cross-dataset)
    img_root = 'dataset/Kvasir-SEG/images/'
    gt_root = 'dataset/Kvasir-SEG/masks/'
    
    dataset = PolypDataset(img_root, gt_root, trainsize=img_size)
    
    output_dir = 'visualizations_polyp'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize 5 random samples
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, gt_tensor = dataset[idx]
            
            # Predict
            img_input = image_tensor.unsqueeze(0).to(device)
            preds = model(img_input)
            pred_mask = torch.sigmoid(preds['seg']).squeeze().cpu().numpy()
            
            # Un-normalize image for plotting
            img_np = image_tensor.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            gt_np = gt_tensor.squeeze().numpy()
            
            # Plot
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_np)
            axs[0].set_title("Endoscopic Image")
            axs[0].axis('off')
            
            axs[1].imshow(gt_np, cmap='gray')
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis('off')
            
            axs[2].imshow(pred_mask, cmap='gray')
            axs[2].set_title("B-Mamba Prediction")
            axs[2].axis('off')
            
            save_path = os.path.join(output_dir, f'sample_{i+1}_idx{idx}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    visualize()
