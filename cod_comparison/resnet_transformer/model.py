"""
Model: ResNet50 + Transformer
==============================
Backbone  : ResNet-50 (ImageNet pretrained)
Neck      : FPN (4 levels)
Branch    : ResNet features → Multi-head Self-Attention Transformer blocks
Heads     : BoundaryHead, HighResBoundaryStream, UncertaintyHead
Decoder   : Progressive (ASPP + lateral connections)

Transformer branch uses standard MHSA + FFN with positional encoding,
matching parameter count to the Mamba branch for fair comparison.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.ops import FeaturePyramidNetwork


# ── ResNet-50 Backbone (same as ResNet_Mamba) ─────────────────────────────────

class ResNetBackbone(nn.Module):
    OUT_CHANNELS = [256, 512, 1024, 2048]

    def __init__(self, pretrained=True):
        super().__init__()
        base = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

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


# ── Transformer Branch ────────────────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for feature maps."""
    def __init__(self, channels, max_h=64, max_w=64):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, channels // 2)
        self.col_embed = nn.Embedding(max_w, channels // 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        rows = torch.arange(H, device=x.device)
        cols = torch.arange(W, device=x.device)
        row_enc = self.row_embed(rows).unsqueeze(1).expand(H, W, C//2)  # (H,W,C/2)
        col_enc = self.col_embed(cols).unsqueeze(0).expand(H, W, C//2)  # (H,W,C/2)
        pos = torch.cat([row_enc, col_enc], dim=-1)                      # (H,W,C)
        return x + pos.permute(2, 0, 1).unsqueeze(0)                    # (B,C,H,W)


class TransformerBlock(nn.Module):
    """Standard MHSA + FFN block with pre-norm."""
    def __init__(self, dim, heads=8, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # x: (B, L, C)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBranch(nn.Module):
    """
    Applies a stack of TransformerBlocks to a 2-D feature map.
    Flattens (B,C,H,W) → (B,H*W,C), runs transformer, reshapes back.
    """
    def __init__(self, channels=512, depth=4, heads=8):
        super().__init__()
        self.proj_in  = nn.Conv2d(channels, channels, 1)
        self.pos_enc  = PositionalEncoding2D(channels)
        self.blocks   = nn.ModuleList([
            TransformerBlock(channels, heads=heads) for _ in range(depth)])
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.norm     = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x   = self.proj_in(x)
        x   = self.pos_enc(x)
        seq = x.flatten(2).transpose(1, 2)    # (B, H*W, C)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        return self.proj_out(seq.transpose(1,2).reshape(B, C, H, W))


# ── Shared Heads (identical to ResNet_Mamba for fair comparison) ──────────────

class TextureEnhancement(nn.Module):
    def __init__(self, ch=512):
        super().__init__()
        self.hf = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch,ch,3,padding=2,dilation=2,padding_mode='replicate'),
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

    def __init__(self, transformer_depth=4, transformer_heads=8,
                 pretrained=True, **kwargs):
        super().__init__()
        self.backbone    = ResNetBackbone(pretrained=pretrained)
        self.fpn         = FeaturePyramidNetwork(
            in_channels_list=ResNetBackbone.OUT_CHANNELS, out_channels=self.FPN_OUT)
        ch = self.BRANCH_CH
        self.proj        = nn.Sequential(nn.Conv2d(self.FPN_OUT,ch,1),
                                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.transformer = TransformerBranch(channels=ch, depth=transformer_depth,
                                              heads=transformer_heads)
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
        feat_t = self.transformer(feat)
        feat_r = self.texture_enh(feat)
        fused  = self.fusion(torch.cat([feat_r, feat_t], dim=1))

        bnd_c  = self.bnd_head(fused, tgt)
        bnd_f  = self.hires_bnd(fpn['p2'], bnd_c, tgt)
        unc    = self.unc_head(fused, tgt)
        seg    = F.interpolate(self.decoder(fused, fpn, bnd_c),
                               size=tgt, mode='bilinear', align_corners=False)
        return {'seg': seg, 'boundary': bnd_f,
                'boundary_coarse': bnd_c, 'uncertainty': unc}