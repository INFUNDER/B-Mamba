"""
Model: ResNet50 + Pure Mamba
=============================
Backbone  : ResNet-50 (ImageNet pretrained)
Neck      : FPN (4 levels)
Branch    : ResNet features → Pure Mamba SSM (global sequence modelling)
Heads     : BoundaryHead, HighResBoundaryStream, UncertaintyHead
Decoder   : Progressive (ASPP + lateral connections)

Pure Mamba = real selective scan (S6), not a gated conv approximation.
Falls back to pure-PyTorch if mamba_ssm CUDA kernel not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.ops import FeaturePyramidNetwork


# ── Pure Mamba ────────────────────────────────────────────────────────────────

def selective_scan_pytorch(u, delta, A_log, B, C, D):
    B_sz, L, d_inner = u.shape
    dA   = torch.exp(delta.unsqueeze(-1) * (-torch.exp(A_log)))
    dB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    h    = torch.zeros(B_sz, d_inner, A_log.shape[1], device=u.device, dtype=u.dtype)
    ys   = []
    for i in range(L):
        h = dA[:, i] * h + dB_u[:, i]
        ys.append((h * C[:, i].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=1)
    return y + u * D.unsqueeze(0).unsqueeze(0)


class PureMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner      = int(dim * expand)
        self.d_inner = d_inner
        self.d_state = d_state
        self.norm    = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                 padding=d_conv-1, groups=d_inner, bias=True)
        self.act     = nn.SiLU()
        self.x_proj  = nn.Linear(d_inner, d_state*2 + d_inner, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log   = nn.Parameter(torch.log(A))
        self.D       = nn.Parameter(torch.ones(d_inner))
        self.out_proj= nn.Linear(d_inner, dim, bias=False)
        self._use_fast = False
        try:
            from mamba_ssm import Mamba as _MF
            self._fast = _MF(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self._use_fast = True
        except Exception:
            pass

    def _forward_pure(self, x):
        B, L, _ = x.shape
        xz      = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv  = self.conv1d(x_in.transpose(1,2))[:,:,:L].transpose(1,2)
        x_conv  = self.act(x_conv)
        x_dbl   = self.x_proj(x_conv)
        B_ssm   = x_dbl[..., :self.d_state]
        C_ssm   = x_dbl[..., self.d_state:self.d_state*2]
        delta   = F.softplus(self.dt_proj(x_dbl[..., self.d_state*2:]))
        y = selective_scan_pytorch(x_conv, delta, self.A_log, B_ssm, C_ssm, self.D)
        return self.out_proj(y * self.act(z))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        return (self._fast(x) if self._use_fast else self._forward_pure(x)) + residual


class MambaBranch(nn.Module):
    def __init__(self, channels=512, depth=4, d_state=16):
        super().__init__()
        self.proj_in  = nn.Conv2d(channels, channels, 1)
        self.blocks   = nn.ModuleList([PureMambaBlock(channels, d_state=d_state)
                                       for _ in range(depth)])
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.norm     = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x   = self.proj_in(x)
        seq = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        return self.proj_out(seq.transpose(1,2).reshape(B, C, H, W))


# ── ResNet-50 Backbone ────────────────────────────────────────────────────────

class ResNetBackbone(nn.Module):
    """ResNet-50, returns 4 feature maps at strides 4,8,16,32."""
    OUT_CHANNELS = [256, 512, 1024, 2048]

    def __init__(self, pretrained=True):
        super().__init__()
        base = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # stride 4,  256ch
        self.layer2 = base.layer2   # stride 8,  512ch
        self.layer3 = base.layer3   # stride 16, 1024ch
        self.layer4 = base.layer4   # stride 32, 2048ch

    def forward(self, x):
        x  = self.layer0(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4

    def freeze_stages(self, n=1):
        layers = [self.layer0, self.layer1, self.layer2, self.layer3]
        for layer in layers[:n+1]:
            for p in layer.parameters():
                p.requires_grad = False


# ── Shared Heads ──────────────────────────────────────────────────────────────

class TextureEnhancement(nn.Module):
    def __init__(self, ch=512):
        super().__init__()
        self.hf = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2, padding_mode='replicate'),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer('sx', sx.view(1,1,3,3))
        self.register_buffer('sy', sy.view(1,1,3,3))
        self.fuse = nn.Sequential(nn.Conv2d(ch+1,ch,1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

    def forward(self, x):
        hf   = self.hf(x)
        fm   = x.mean(dim=1, keepdim=True)
        edge = torch.sigmoid(torch.sqrt(F.conv2d(fm,self.sx,padding=1)**2 +
                                        F.conv2d(fm,self.sy,padding=1)**2 + 1e-6))
        if edge.shape[-2:] != hf.shape[-2:]:
            edge = F.interpolate(edge, size=hf.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([hf*(1+edge), edge], dim=1))


class BoundaryHead(nn.Module):
    def __init__(self, ch=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,1,1))

    def forward(self, x, size):
        return F.interpolate(self.net(x), size=size, mode='bilinear', align_corners=False)


class HighResBoundaryStream(nn.Module):
    def __init__(self, fpn_ch=256):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(fpn_ch+1,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),        nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=2,dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,1,1))

    def forward(self, p2, coarse, size):
        b = F.interpolate(coarse, size=p2.shape[2:], mode='bilinear', align_corners=False)
        return F.interpolate(self.refine(torch.cat([p2,b],dim=1)),
                             size=size, mode='bilinear', align_corners=False)


class UncertaintyHead(nn.Module):
    def __init__(self, ch=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(ch,64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,1,1), nn.Sigmoid())

    def forward(self, x, size):
        return F.interpolate(self.head(x), size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=r,dilation=r),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)) for r in rates])
        self.gp   = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(in_ch,out_ch,1), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(out_ch*(len(rates)+1),out_ch,1),
                                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        gp    = F.interpolate(self.gp(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat(feats+[gp], dim=1))


class ProgressiveDecoder(nn.Module):
    def __init__(self, fpn_ch=256, fused_ch=512):
        super().__init__()
        self.aspp = ASPP(fused_ch, 256)
        self.lat4 = nn.Sequential(nn.Conv2d(fpn_ch,128,1), nn.ReLU(inplace=True))
        self.lat3 = nn.Sequential(nn.Conv2d(fpn_ch, 64,1), nn.ReLU(inplace=True))
        self.lat2 = nn.Sequential(nn.Conv2d(fpn_ch, 32,1), nn.ReLU(inplace=True))
        self.ref4 = nn.Sequential(nn.Conv2d(256+128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.ref3 = nn.Sequential(nn.Conv2d(128+64,  64,3,padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True))
        self.ref2 = nn.Sequential(nn.Conv2d(64+32+1, 32,3,padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True))
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, fused, fpn, bnd):
        x = self.aspp(fused)
        x = F.interpolate(x, size=fpn['p4'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.ref4(torch.cat([x, self.lat4(fpn['p4'])], dim=1))
        x = F.interpolate(x, size=fpn['p3'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.ref3(torch.cat([x, self.lat3(fpn['p3'])], dim=1))
        x = F.interpolate(x, size=fpn['p2'].shape[-2:], mode='bilinear', align_corners=False)
        b = F.interpolate(bnd, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.ref2(torch.cat([x, self.lat2(fpn['p2']), torch.sigmoid(b)], dim=1))
        return self.head(x)


# ── Full Model ────────────────────────────────────────────────────────────────

class CODModel(nn.Module):
    FPN_OUT   = 256
    BRANCH_CH = 512

    def __init__(self, mamba_depth=4, mamba_d_state=16, pretrained=True, **kwargs):
        super().__init__()
        self.backbone    = ResNetBackbone(pretrained=pretrained)
        self.fpn         = FeaturePyramidNetwork(
            in_channels_list=ResNetBackbone.OUT_CHANNELS, out_channels=self.FPN_OUT)
        ch = self.BRANCH_CH
        self.proj        = nn.Sequential(nn.Conv2d(self.FPN_OUT,ch,1),
                                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.mamba       = MambaBranch(channels=ch, depth=mamba_depth, d_state=mamba_d_state)
        self.texture_enh = TextureEnhancement(ch=ch)
        self.fusion      = nn.Sequential(nn.Conv2d(ch*2,ch,1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.bnd_head    = BoundaryHead(ch)
        self.hires_bnd   = HighResBoundaryStream(self.FPN_OUT)
        self.unc_head    = UncertaintyHead(ch)
        self.decoder     = ProgressiveDecoder(self.FPN_OUT, ch)

    def freeze_backbone(self, n=1):
        self.backbone.freeze_stages(n)

    def forward(self, x):
        tgt = x.shape[2:]
        c1, c2, c3, c4 = self.backbone(x)
        raw  = self.fpn({'0':c1,'1':c2,'2':c3,'3':c4})
        fpn  = {'p2':raw['0'],'p3':raw['1'],'p4':raw['2'],'p5':raw['3']}

        feat   = self.proj(fpn['p5'])
        feat_m = self.mamba(feat)
        feat_r = self.texture_enh(feat)
        fused  = self.fusion(torch.cat([feat_r, feat_m], dim=1))

        bnd_c  = self.bnd_head(fused, tgt)
        bnd_f  = self.hires_bnd(fpn['p2'], bnd_c, tgt)
        unc    = self.unc_head(fused, tgt)
        seg    = F.interpolate(self.decoder(fused, fpn, bnd_c),
                               size=tgt, mode='bilinear', align_corners=False)
        return {'seg': seg, 'boundary': bnd_f,
                'boundary_coarse': bnd_c, 'uncertainty': unc}