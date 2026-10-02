import torch
import torch.nn.functional as F

class MedicalMetrics:
    """
    Computes standard medical image segmentation metrics:
    - Dice Similarity Coefficient (DSC)
    - Intersection over Union (IoU)
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.count = 0

    def update(self, pred, gt):
        """
        pred: torch.Tensor of shape (B, 1, H, W) - logits or probabilities
        gt: torch.Tensor of shape (B, 1, H, W) - binary mask (0 or 1)
        """
        # Ensure pred is in [0, 1]
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
            
        pred = (pred > self.threshold).float()
        gt = (gt > 0.5).float()
        
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            p = pred[i].view(-1)
            g = gt[i].view(-1)
            
            intersection = (p * g).sum()
            
            # Dice = 2 * |A AND B| / (|A| + |B|)
            dice = (2. * intersection + 1e-5) / (p.sum() + g.sum() + 1e-5)
            
            # IoU = |A AND B| / (|A| OR |B|)
            iou = (intersection + 1e-5) / (p.sum() + g.sum() - intersection + 1e-5)
            
            self.total_dice += dice.item()
            self.total_iou += iou.item()
            self.count += 1

    def get_metrics(self):
        if self.count == 0:
            return {"dice": 0.0, "iou": 0.0}
        return {
            "dice": self.total_dice / self.count,
            "iou": self.total_iou / self.count
        }

def compute_dice_loss(pred, gt, smooth=1.0):
    """
    Differentiable Dice Loss for training.
    pred: logits
    gt: binary mask
    """
    pred = torch.sigmoid(pred)
    
    # Flatten
    pred = pred.view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    
    intersection = (pred * gt).sum(-1)
    union = pred.sum(-1) + gt.sum(-1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Loss is 1 - Dice
    return 1.0 - dice.mean()

def structure_loss(pred, mask):
    """
    Structure loss combining weighted BCE and weighted IoU.
    Standard loss for polyp segmentation architectures.
    """
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter + 1)
    
    return (wbce + wiou).mean()
