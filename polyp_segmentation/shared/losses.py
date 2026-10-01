"""
Shared Loss Functions — COD Comparison Study
=============================================
Same for ALL 3 models. Ensures fair comparison.

Weights (tuned in v7):
  w_bce   = 1.0
  w_iou   = 0.5   (reduced — was dominating, suppressed boundary sharpness)
  w_dice  = 0.5
  w_bnd   = 2.0   (increased — stronger edge supervision)
  w_bnd_c = 0.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(pred, target, uncertainty=None):
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    if uncertainty is not None:
        loss = loss * (1.0 + uncertainty.detach())
    return loss.mean()


def iou_loss(pred, target, eps=1e-6):
    p     = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(1,2,3))
    union = (p + target - p * target).sum(dim=(1,2,3))
    return (1 - (inter + eps) / (union + eps)).mean()


def dice_loss(pred, target, eps=1e-6):
    p     = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(1,2,3))
    total = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return (1 - (2 * inter + eps) / (total + eps)).mean()


def boundary_loss(pred_boundary, gt_mask, dilation=4, eps=1e-6):
    kernel     = torch.ones(1, 1, dilation*2+1, dilation*2+1, device=gt_mask.device)
    dilated    = (F.conv2d(gt_mask,   kernel, padding=dilation) > 0).float()
    eroded     = (F.conv2d(1-gt_mask, kernel, padding=dilation) < 1).float()
    gt_bnd     = (dilated - eroded).clamp(0, 1)
    pos        = gt_bnd.sum()
    neg        = (1 - gt_bnd).sum()
    pos_w      = (neg / (pos + eps)).clamp(max=10.0)
    loss = F.binary_cross_entropy_with_logits(
        pred_boundary, gt_bnd, pos_weight=pos_w, reduction='mean')
    return loss, gt_bnd


class CODLoss(nn.Module):
    def __init__(
        self,
        w_bce=1.0,
        w_iou=0.5,
        w_dice=0.5,
        w_boundary=2.0,
        w_boundary_coarse=0.8,
    ):
        super().__init__()
        self.w_bce             = w_bce
        self.w_iou             = w_iou
        self.w_dice            = w_dice
        self.w_boundary        = w_boundary
        self.w_boundary_coarse = w_boundary_coarse

    def forward(self, outputs, gt_mask):
        """
        outputs must contain: 'seg', 'boundary', 'boundary_coarse', 'uncertainty'
        gt_mask : (B, 1, H, W) binary float
        """
        pred_seg   = outputs['seg']
        pred_bnd_f = outputs['boundary']
        pred_bnd_c = outputs['boundary_coarse']
        uncertainty= outputs['uncertainty']

        l_bce  = bce_loss(pred_seg, gt_mask, uncertainty)
        l_iou  = iou_loss(pred_seg, gt_mask)
        l_dice = dice_loss(pred_seg, gt_mask)
        l_bnd_f, _ = boundary_loss(pred_bnd_f, gt_mask)
        l_bnd_c, _ = boundary_loss(pred_bnd_c, gt_mask)

        total = (
            self.w_bce             * l_bce    +
            self.w_iou             * l_iou    +
            self.w_dice            * l_dice   +
            self.w_boundary        * l_bnd_f  +
            self.w_boundary_coarse * l_bnd_c
        )

        return total, {
            'bce':        l_bce.item(),
            'iou':        l_iou.item(),
            'dice':       l_dice.item(),
            'bnd_fine':   l_bnd_f.item(),
            'bnd_coarse': l_bnd_c.item(),
            'total':      total.item(),
        }