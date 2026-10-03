import os
import torch
from polyp_segmentation.shared.dataset_polyp import get_loader
from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro
from polyp_segmentation.shared.metrics_polyp import MedicalMetrics
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='sota')
    parser.add_argument('--no_deep_sup', action='store_true')
    parser.add_argument('--no_boundary', action='store_true')
    return parser.parse_args()

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
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_size = 352
    model_path = f'polyp_segmentation/checkpoints/b_mamba_best_{args.run_name}.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    model = BMambaModel_Pro(
        pretrained=False, 
        img_size=img_size,
        use_deep_sup=not args.no_deep_sup,
        use_boundary=not args.no_boundary
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\n--- Full PraNet Evaluation Protocol (5 Datasets) ---")
    datasets = [
        ('Kvasir', 'dataset/TestDataset/Kvasir/images/', 'dataset/TestDataset/Kvasir/masks/'),
        ('CVC-ClinicDB', 'dataset/TestDataset/CVC-ClinicDB/images/', 'dataset/TestDataset/CVC-ClinicDB/masks/'),
        ('CVC-ColonDB', 'dataset/TestDataset/CVC-ColonDB/images/', 'dataset/TestDataset/CVC-ColonDB/masks/'),
        ('ETIS-LaribPolypDB', 'dataset/TestDataset/ETIS-LaribPolypDB/images/', 'dataset/TestDataset/ETIS-LaribPolypDB/masks/'),
        ('EndoScene (CVC-300)', 'dataset/TestDataset/CVC-300/images/', 'dataset/TestDataset/CVC-300/masks/')
    ]
    
    for name, img_path, gt_path in datasets:
        evaluate_dataset(model, device, img_path, gt_path, name)

if __name__ == '__main__':
    test_pranet_protocol()
