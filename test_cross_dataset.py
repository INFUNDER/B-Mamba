import os
import torch
from polyp_segmentation.shared.dataset_polyp import get_loader
from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel
from polyp_segmentation.shared.metrics_polyp import MedicalMetrics

def test_cross_dataset():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_size = 352
    
    # 1. Load the model trained on Kvasir-SEG
    model_path = 'polyp_segmentation/checkpoints/b_mamba_best_sota.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    model = BMambaModel(pretrained=False, img_size=img_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Point to the unseen CVC-ClinicDB dataset
    test_img_path = 'dataset/CVC-ClinicDB/images/'
    test_gt_path = 'dataset/CVC-ClinicDB/masks/'
    
    if not os.path.exists(test_img_path):
        print("CVC-ClinicDB dataset not found. Please download it and place it in dataset/CVC-ClinicDB/ (images/ and masks/).")
        return
        
    test_loader = get_loader(test_img_path, test_gt_path, batchsize=1, trainsize=img_size, shuffle=False)
    metrics = MedicalMetrics(threshold=0.5)
    
    print("Running cross-dataset evaluation on CVC-ClinicDB...")
    with torch.no_grad():
        for i, (images, gts) in enumerate(test_loader):
            images = images.to(device)
            gts = gts.to(device)
            
            preds = model(images)
            seg_pred = preds['seg']
            
            metrics.update(seg_pred, gts)
            
    final_metrics = metrics.get_metrics()
    print("=== Cross-Dataset Results (Trained on Kvasir, Tested on ClinicDB) ===")
    print(f"Dice Score (DSC): {final_metrics['dice']:.4f}")
    print(f"IoU Score: {final_metrics['iou']:.4f}")
    print("=======================================================================")

if __name__ == '__main__':
    test_cross_dataset()
