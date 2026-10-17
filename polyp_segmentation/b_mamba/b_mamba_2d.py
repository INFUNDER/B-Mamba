"""
B-Mamba 2D: Multi-Scale 4-Way Cross-Scan Boundary-Conditioned Mamba
==================================================================
Key innovations for Q1 Medical Imaging Journals (IEEE TMI / MedIA):
  1. 4-Way Omni-Directional Cross-Scan: Replaces 1D causal scan with 4 symmetric traversals
     (horizontal forward, horizontal backward, vertical forward, vertical backward), ensuring
     lesion boundaries in all 4 cardinal directions enforce causal memory resets.
  2. Multi-Scale Skip Conditioning: BC-SSM blocks are deployed at P4 (22x22, stride 16)
     and P3 (44x44, stride 8) in addition to bottleneck P5 (11x11, stride 32). Small polyps
     span 4 to 64 tokens at these resolutions, directly sharpening skip features before the decoder.
  3. Supervised Boundary Stream: Morphological gradient contour prediction on P3/P4 features,
     isolating real lesion borders from specular glints and mucosal folds.
  4. Decoupled Factorial Architecture: Strictly separates SSM boundary gating (e_t) from
     raw boundary prior bypass (F_r) for airtight ablation isolation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.ops import FeaturePyramidNetwork

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_CUDA_SSM = True
except Exception:
    HAS_CUDA_SSM = False


def selective_scan_pytorch(u, delta, A_log, B, C, D):
    """
    Fallback batched selective scan in pure PyTorch.
      u: (B, L, D)
      delta: (B, L, D)
      A_log: (D, N)
      B: (B, L, N)
      C: (B, L, N)
      D: (D)
    """
    B_sz, L, d_inner = u.shape
    dA = torch.exp(delta.unsqueeze(-1) * (-torch.exp(A_log)))
    dB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    h = torch.zeros(B_sz, d_inner, A_log.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for i in range(L):
        h = dA[:, i] * h + dB_u[:, i]
        ys.append((h * C[:, i].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=1)
    return y + u * D.unsqueeze(0).unsqueeze(0)


def boundary_target(mask, k=7):
    """Morphological gradient of binary mask (B, 1, H, W) -> explicit contour target."""
    pad = k // 2
    dil = F.max_pool2d(mask, k, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, k, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


def boundary_loss(logit, target):
    """Positive-weighted BCE + soft Dice loss for thin contours."""
    pos_w = ((1 - target).sum() / target.sum().clamp(min=1)).clamp(1, 30)
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_w)
    p = torch.sigmoid(logit)
    inter = (p * target).sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
    return bce + dice.mean()


class CrossScanBCSSMBlock(nn.Module):
    """
    4-Way Omni-Directional Boundary-Conditioned SSM Block.
    Traverses the 2D feature map in 4 complementary directions:
      1. Row-major forward  (L -> R, T -> B)
      2. Row-major backward (R -> L, B -> T)
      3. Col-major forward  (T -> B, L -> R)
      4. Col-major backward (B -> T, R -> L)
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2, bnd_dim=None):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expand)
        self.d_state = d_state
        if bnd_dim is None:
            bnd_dim = dim
        self.bnd_dim = bnd_dim
        
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv // 2, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()
        
        # Projection for SSM parameters conditioned on [u, boundary]
        self.x_proj = nn.Linear(self.d_inner + bnd_dim, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x, boundary_feat=None):
        """
        x: (B, C, H, W)
        boundary_feat: (B, bnd_dim, H, W) or None
        """
        B_sz, C, H, W = x.shape
        residual = x
        
        # Normalization and input linear projection
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        xz = self.in_proj(x_norm.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_in, z = xz.chunk(2, dim=1)
        
        # Local 2D depthwise convolution
        u = self.act(self.conv2d(x_in)) # (B, d_inner, H, W)
        
        # 4-way cross scan on feature map u
        u1 = u.flatten(2).transpose(1, 2)                            # (B, HW, d_inner)
        u2 = u.flatten(2).flip(-1).transpose(1, 2)                   # (B, HW, d_inner)
        u3 = u.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)       # (B, HW, d_inner)
        u4 = u.permute(0, 1, 3, 2).flatten(2).flip(-1).transpose(1, 2) # (B, HW, d_inner)
        u_4way = torch.cat([u1, u2, u3, u4], dim=0)                  # (4B, HW, d_inner)
        
        # 4-way cross scan on boundary features
        if boundary_feat is not None:
            b1 = boundary_feat.flatten(2).transpose(1, 2)
            b2 = boundary_feat.flatten(2).flip(-1).transpose(1, 2)
            b3 = boundary_feat.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)
            b4 = boundary_feat.permute(0, 1, 3, 2).flatten(2).flip(-1).transpose(1, 2)
            b_4way = torch.cat([b1, b2, b3, b4], dim=0)
            x_cond = torch.cat([u_4way, b_4way], dim=-1)
        else:
            b_zero = torch.zeros(4 * B_sz, H * W, self.bnd_dim, device=x.device, dtype=x.dtype)
            x_cond = torch.cat([u_4way, b_zero], dim=-1)
            
        x_dbl = self.x_proj(x_cond)
        B_ssm = x_dbl[..., :self.d_state]
        C_ssm = x_dbl[..., self.d_state:self.d_state * 2]
        delta = F.softplus(self.dt_proj(x_dbl[..., self.d_state * 2:]))
        
        # Fast CUDA selective scan when available, fallback to PyTorch
        if HAS_CUDA_SSM and u.is_cuda:
            u_t = u_4way.transpose(1, 2).contiguous()       # (4B, d_inner, HW)
            delta_t = delta.transpose(1, 2).contiguous()   # (4B, d_inner, HW)
            A = -torch.exp(self.A_log)                     # (d_inner, d_state)
            B_t = B_ssm.transpose(1, 2).contiguous()       # (4B, d_state, HW)
            C_t = C_ssm.transpose(1, 2).contiguous()       # (4B, d_state, HW)
            y_4way_t = selective_scan_fn(u_t, delta_t, A, B_t, C_t, self.D)
            y_4way = y_4way_t.transpose(1, 2)              # (4B, HW, d_inner)
        else:
            y_4way = selective_scan_pytorch(u_4way, delta, self.A_log, B_ssm, C_ssm, self.D)
            
        # Unpack the 4 scan paths
        y1, y2, y3, y4 = y_4way.chunk(4, dim=0)
        
        # Spatial inversion back to canonical (B, d_inner, H, W)
        y1 = y1.transpose(1, 2).reshape(B_sz, self.d_inner, H, W)
        y2 = y2.transpose(1, 2).flip(-1).reshape(B_sz, self.d_inner, H, W)
        y3 = y3.transpose(1, 2).reshape(B_sz, self.d_inner, W, H).permute(0, 1, 3, 2)
        y4 = y4.transpose(1, 2).flip(-1).reshape(B_sz, self.d_inner, W, H).permute(0, 1, 3, 2)
        
        # Omni-directional fusion
        y_fused = (y1 + y2 + y3 + y4) * 0.25
        
        # Gate and output projection
        out = y_fused * self.act(z)
        out = self.out_proj(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out + residual


class MultiScaleBoundaryHead(nn.Module):
    """
    Supervised Boundary Head operating on P3 (44x44) and P4 (22x22).
    Generates multi-scale boundary representations E3, E4, E5 and predicts
    explicit lesion contour logits supervised with morphological gradient targets.
    """
    def __init__(self, in_ch=256, mid_ch=128):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.edge_out = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, p3, p4):
        # p3: (B, 256, 44, 44), p4: (B, 256, 22, 22)
        f3 = self.conv3(p3)
        edge_logit = self.edge_out(f3) # (B, 1, 44, 44)
        edge_gate = torch.sigmoid(edge_logit)
        
        # Multi-scale boundary features
        e3 = f3 * (1 + edge_gate) # (B, 128, 44, 44)
        f4 = self.conv4(p4)
        e4 = F.interpolate(e3, size=p4.shape[-2:], mode='bilinear', align_corners=False) + f4
        e5 = F.adaptive_avg_pool2d(e4, (11, 11))
        
        return {'e3': e3, 'e4': e4, 'e5': e5, 'edge_logit': edge_logit}


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)) for r in rates])
        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_ch, out_ch, 1), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        gp = F.interpolate(self.gp(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat(feats + [gp], dim=1))


class ProgressiveDecoder2D(nn.Module):
    """
    Progressive Decoder receiving boundary-purged skip connections at P4, P3, P2.
    """
    def __init__(self, fpn_ch=256, fused_ch=512):
        super().__init__()
        self.aspp = ASPP(fused_ch, 256)
        self.lat4 = nn.Sequential(nn.Conv2d(fpn_ch, 128, 1), nn.ReLU(inplace=True))
        self.lat3 = nn.Sequential(nn.Conv2d(fpn_ch, 64, 1), nn.ReLU(inplace=True))
        self.lat2 = nn.Sequential(nn.Conv2d(fpn_ch, 32, 1), nn.ReLU(inplace=True))
        
        self.ref4 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.ref3 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.ref2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        
        # Deep supervision heads
        self.head4 = nn.Conv2d(128, 1, 1)
        self.head3 = nn.Conv2d(64, 1, 1)
        self.head2 = nn.Conv2d(32, 1, 1)

    def forward(self, fused, p4_skip, p3_skip, p2_skip):
        x4 = self.aspp(fused)
        x4 = F.interpolate(x4, size=p4_skip.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.ref4(torch.cat([x4, self.lat4(p4_skip)], dim=1))
        out4 = self.head4(x4)
        
        x3 = F.interpolate(x4, size=p3_skip.shape[-2:], mode='bilinear', align_corners=False)
        x3 = self.ref3(torch.cat([x3, self.lat3(p3_skip)], dim=1))
        out3 = self.head3(x3)
        
        x2 = F.interpolate(x3, size=p2_skip.shape[-2:], mode='bilinear', align_corners=False)
        x2 = self.ref2(torch.cat([x2, self.lat2(p2_skip)], dim=1))
        out2 = self.head2(x2)
        
        return out2, out3, out4


class BMamba2D(nn.Module):
    """
    Full Q1 Architecture: Multi-Scale 4-Way Cross-Scan Boundary-Conditioned Mamba.
    """
    FPN_OUT = 256
    BOTTLENECK_CH = 512
    BND_CH = 128

    def __init__(self, pretrained=True, img_size=352,
                 use_boundary=True,       # Gating e_t in SSM blocks
                 use_bpe_bypass=True,     # Raw F_r feature bypass into bottleneck fusion
                 use_deep_sup=True):      # Multi-scale deep supervision
        super().__init__()
        self.use_boundary = use_boundary
        self.use_bpe_bypass = use_bpe_bypass
        self.use_deep_sup = use_deep_sup
        
        # Swin-Base Backbone
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
            img_size=(img_size, img_size))
        out_channels = [128, 256, 512, 1024]
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(in_channels_list=out_channels, out_channels=self.FPN_OUT)
        
        # Boundary Head
        self.bnd_head = MultiScaleBoundaryHead(in_ch=self.FPN_OUT, mid_ch=self.BND_CH)
        
        # Bottleneck projections
        self.p5_proj = nn.Sequential(nn.Conv2d(self.FPN_OUT, self.BOTTLENECK_CH, 1),
                                     nn.BatchNorm2d(self.BOTTLENECK_CH), nn.ReLU(inplace=True))
        
        # 4-Way BC-SSM at Bottleneck (P5: 11x11, 121 tokens)
        self.bcssm_p5 = CrossScanBCSSMBlock(dim=self.BOTTLENECK_CH, d_state=16, expand=2, bnd_dim=self.BND_CH)
        
        # 4-Way BC-SSM Skip Filters at P4 (22x22, 484 tokens) and P3 (44x44, 1936 tokens)
        self.bcssm_p4 = CrossScanBCSSMBlock(dim=self.FPN_OUT, d_state=16, expand=1, bnd_dim=self.BND_CH)
        self.bcssm_p3 = CrossScanBCSSMBlock(dim=self.FPN_OUT, d_state=16, expand=1, bnd_dim=self.BND_CH)
        
        # BPE Bypass projection for F_r
        if self.use_bpe_bypass:
            self.bpe_bypass_proj = nn.Sequential(
                nn.Conv2d(self.BND_CH, self.BOTTLENECK_CH, 1),
                nn.BatchNorm2d(self.BOTTLENECK_CH), nn.ReLU(inplace=True)
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(self.BOTTLENECK_CH * 2, self.BOTTLENECK_CH, 1),
                nn.BatchNorm2d(self.BOTTLENECK_CH), nn.ReLU(inplace=True)
            )
        else:
            self.fusion = nn.Identity()
            
        # Progressive Decoder
        self.decoder = ProgressiveDecoder2D(self.FPN_OUT, self.BOTTLENECK_CH)

    def forward(self, x):
        tgt = x.shape[2:]
        
        # 1. Backbone forward
        swin_feats = [f.permute(0, 3, 1, 2).contiguous() for f in self.swin(x)]
        raw_fpn = self.fpn({'0': swin_feats[0], '1': swin_feats[1], '2': swin_feats[2], '3': swin_feats[3]})
        p2, p3, p4, p5 = raw_fpn['0'], raw_fpn['1'], raw_fpn['2'], raw_fpn['3']
        
        # 2. Extract multi-scale boundaries
        bnd = self.bnd_head(p3, p4)
        e3 = bnd['e3'] if self.use_boundary else None
        e4 = bnd['e4'] if self.use_boundary else None
        e5 = bnd['e5'] if self.use_boundary else None
        
        # 3. Multi-scale 4-way BC-SSM filtering
        p5_feat = self.p5_proj(p5)
        p5_filtered = self.bcssm_p5(p5_feat, e5)
        
        # Filter skip connections: purge mucosal context before decoder
        p4_filtered = self.bcssm_p4(p4, e4)
        p3_filtered = self.bcssm_p3(p3, e3)
        p2_skip = p2 # High resolution fine detail
        
        # 4. Bottleneck fusion
        if self.use_bpe_bypass:
            f_r = self.bpe_bypass_proj(bnd['e5'])
            fused = self.fusion(torch.cat([f_r, p5_filtered], dim=1))
        else:
            fused = p5_filtered
            
        # 5. Progressive decoding with deep supervision
        out2, out3, out4 = self.decoder(fused, p4_filtered, p3_filtered, p2_skip)
        
        seg = F.interpolate(out2, size=tgt, mode='bilinear', align_corners=False)
        
        if self.training:
            edge_pred = F.interpolate(bnd['edge_logit'], size=tgt, mode='bilinear', align_corners=False)
            res = {'seg': seg, 'edge': edge_pred}
            if self.use_deep_sup:
                res['seg3'] = F.interpolate(out3, size=tgt, mode='bilinear', align_corners=False)
                res['seg4'] = F.interpolate(out4, size=tgt, mode='bilinear', align_corners=False)
            return res
            
        return {'seg': seg}
