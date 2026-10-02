import os
import torch
from polyp_segmentation.shared.dataset_polyp import get_loader
from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro
from polyp_segmentation.shared.metrics_polyp import MedicalMetrics

def evaluate_dataset(model, device, img_path, gt_path, name):
    if not os.path.exists(img_path):
        print(f"Directory {img_path} not found.")
        return
        
    loader = get_loader(img_path, gt_path, batchsize=1, trainsize=352, shuffle=False, is_train=False)
    metrics = MedicalMetrics(threshold=0.5)
    
    with torch.no_grad():
        for images, gts in loader:
            images = images.to(device)
            gts = gts.to(device)
            
            preds = model(images)
            seg_pred = preds['seg']
            
            metrics.update(seg_pred, gts)
            
    final_metrics = metrics.get_metrics()
    print(f"=== Results for {name} ===")
    print(f"Dice Score (DSC): {final_metrics['dice']:.4f}")
    print(f"IoU Score: {final_metrics['iou']:.4f}")
    print("="*40)

def test_pranet_protocol():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_size = 352
    model_path = 'polyp_segmentation/checkpoints/b_mamba_best_sota.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    model = BMambaModel_Pro(pretrained=False, img_size=img_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\n--- PraNet Evaluation Protocol ---")
    evaluate_dataset(model, device, 
                     'dataset/PraNet_Split/TestDataset/Kvasir/images/',
                     'dataset/PraNet_Split/TestDataset/Kvasir/masks/',
                     'Unseen Kvasir (100 images)')
                     
    evaluate_dataset(model, device, 
                     'dataset/PraNet_Split/TestDataset/ClinicDB/images/',
                     'dataset/PraNet_Split/TestDataset/ClinicDB/masks/',
                     'Unseen ClinicDB (~62 images)')

if __name__ == '__main__':
    test_pranet_protocol()
