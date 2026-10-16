"""
B-Mamba v3  (Phase 2: make boundary conditioning actually matter)
==================================================================
Differences from BMambaModel_Pro (which is left untouched):

  1. SUPERVISED boundary head (on FPN p4, stride 16).  Boundary tokens are no longer a Sobel filter on an
     11x11 channel-mean map; they come from features trained to predict the polyp contour
     (morphological gradient of the GT mask).  This also removes the specular-highlight failure of Sobel,
     because the head learns that highlights are not polyp borders.
  2. BC-SSM at TWO scales: the original p5 branch (11x11 tokens, depth 4) plus a new p4 branch
     (22x22 = 484 tokens, depth 2) whose output is added residually to p4 through a zero-initialised gate.
  3. Clean control: use_boundary=False zeroes ONLY the conditioning input of every SSM block (exact S6),
     while the architecture, parameter count and edge-head auxiliary loss stay identical.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from polyp_segmentation.b_mamba.b_mamba_model import (
    SwinBaseBackbone, TextureEnhancement, BMambaBranch, ProgressiveDecoder_Pro)


def boundary_target(mask, k=7):
    """Morphological gradient of a binary mask (B,1,H,W) -> thin contour band."""
    pad = k // 2
    dil = F.max_pool2d(mask, k, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, k, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


def boundary_loss(logit, target):
    """BCE (positive-weighted, contours are rare) + soft Dice."""
    pos_w = ((1 - target).sum() / target.sum().clamp(min=1)).clamp(1, 30)
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_w)
    p = torch.sigmoid(logit)
    inter = (p * target).sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
    return bce + dice.mean()


class BoundaryHead(nn.Module):
    def __init__(self, in_ch=256, mid=256):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(mid, 1, 1)

    def forward(self, x):
        f = self.feat(x)
        logit = self.out(f)
        return f * (1 + torch.sigmoid(logit)), logit   # edge-gated features, edge logit


class BMambaModelV3(nn.Module):
    FPN_OUT = 256
    BRANCH_CH = 512

    def __init__(self, mamba_depth=4, p4_depth=2, mamba_d_state=16, pretrained=True,
                 img_size=352, use_deep_sup=True, use_boundary=True):
        super().__init__()
        self.use_deep_sup = use_deep_sup
        self.use_boundary = use_boundary
        ch = self.BRANCH_CH
        self.backbone = SwinBaseBackbone(pretrained=pretrained, input_size=img_size)
        self.fpn = FeaturePyramidNetwork(in_channels_list=SwinBaseBackbone.OUT_CHANNELS,
                                         out_channels=self.FPN_OUT)
        self.res_proj = nn.Sequential(nn.Conv2d(self.FPN_OUT, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.mamba_proj = nn.Sequential(nn.Conv2d(self.FPN_OUT, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.texture_enh = TextureEnhancement(ch=ch)

        self.edge_head = BoundaryHead(self.FPN_OUT, self.FPN_OUT)
        self.bnd5_proj = nn.Sequential(nn.Conv2d(self.FPN_OUT, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

        self.b_mamba5 = BMambaBranch(channels=ch, depth=mamba_depth, d_state=mamba_d_state)
        self.b_mamba4 = BMambaBranch(channels=self.FPN_OUT, depth=p4_depth, d_state=mamba_d_state)
        self.gate4 = nn.Parameter(torch.zeros(1))      # p4 branch starts as identity

        self.fusion = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.decoder = ProgressiveDecoder_Pro(self.FPN_OUT, ch)

    def forward(self, x):
        tgt = x.shape[2:]
        c1, c2, c3, c4 = self.backbone(x)
        raw = self.fpn({'0': c1, '1': c2, '2': c3, '3': c4})
        fpn = {'p2': raw['0'], 'p3': raw['1'], 'p4': raw['2'], 'p5': raw['3']}

        bnd4, edge_logit = self.edge_head(fpn['p4'])               # (B,256,22,22), (B,1,22,22)
        if self.use_boundary:
            bnd5 = self.bnd5_proj(F.adaptive_avg_pool2d(bnd4, fpn['p5'].shape[-2:]))
            bnd4_in = bnd4
        else:
            bnd5, bnd4_in = None, None                              # exact S6 control

        # stride-16 BC-SSM (484 tokens), residual with zero-init gate
        fpn['p4'] = fpn['p4'] + self.gate4 * self.b_mamba4(fpn['p4'], bnd4_in)

        # stride-32 BC-SSM (121 tokens) + texture branch
        feat_r, _ = self.texture_enh(self.res_proj(fpn['p5']))
        feat_m = self.b_mamba5(self.mamba_proj(fpn['p5']), bnd5)
        if feat_m.shape[-2:] != feat_r.shape[-2:]:
            feat_m = F.interpolate(feat_m, size=feat_r.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([feat_r, feat_m], dim=1))

        out2, out3, out4 = self.decoder(fused, fpn)
        seg = F.interpolate(out2, size=tgt, mode='bilinear', align_corners=False)
        if self.training:
            res = {'seg': seg,
                   'edge': F.interpolate(edge_logit, size=tgt, mode='bilinear', align_corners=False)}
            if self.use_deep_sup:
                res['seg3'] = F.interpolate(out3, size=tgt, mode='bilinear', align_corners=False)
                res['seg4'] = F.interpolate(out4, size=tgt, mode='bilinear', align_corners=False)
            return res
        return {'seg': seg}
