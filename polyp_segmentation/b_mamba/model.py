"""
Model: Swin-T + Pure Mamba  (Best model — v7 architecture)
============================================================
Backbone  : Swin-Tiny (ImageNet pretrained, patch4 window7)
Neck      : FPN (4 levels)
Branch 1  : Texture Enhancement on p5 features
Branch 2  : Pure Mamba SSM on p5 features
Fusion    : Cross-Attention between the two branches
Heads     : BoundaryHead, HighResBoundaryStream, UncertaintyHead
Decoder   : Progressive (ASPP + lateral connections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
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
        self.in_proj = nn.Linear(dim, d_inner*2, bias=False)
        self.conv1d  = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                 padding=d_conv-1, groups=d_inner, bias=True)
        self.act     = nn.SiLU()
        self.x_proj  = nn.Linear(d_inner, d_state*2+d_inner, bias=False)
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


# ── Swin-T Backbone ───────────────────────────────────────────────────────────

class SwinBackbone(nn.Module):
    OUT_CHANNELS = [96, 192, 384, 768]

    def __init__(self, pretrained=True, swin_weights=None, input_size=384):
        super().__init__()
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained if swin_weights is None else False,
            features_only=True, out_indices=(0,1,2,3),
            img_size=(input_size, input_size))
        if swin_weights:
            print(f"Loading Swin-T weights: {swin_weights}")
            sd = torch.load(swin_weights, map_location='cpu', weights_only=False)
            if isinstance(sd, dict):
                sd = sd.get('model', sd.get('state_dict', sd))
            new_sd = {}
            for k, v in sd.items():
                k = k.replace('module.', '')
                if k.startswith('layers.'):
                    k = k.replace('layers.', 'layers_', 1)
                new_sd[k] = v
            print(self.swin.load_state_dict(new_sd, strict=False))

    def forward(self, x):
        return [f.permute(0,3,1,2).contiguous() for f in self.swin(x)]

    def freeze_stages(self, n=2):
        if hasattr(self.swin, 'patch_embed'):
            for p in self.swin.patch_embed.parameters():
                p.requires_grad = False
        for i in range(n):
            stage = getattr(self.swin, f'layers_{i}', None)
            if stage:
                for p in stage.parameters():
                    p.requires_grad = False


# ── Cross-Attention (Swin+Mamba specific — enables branch fusion) ─────────────

class CrossAttention(nn.Module):
    def __init__(self, channels=512, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        for tag in ('r','m'):
            for proj in ('q','k','v'):
                setattr(self, f'{proj}_{tag}', nn.Conv2d(channels, channels, 1))
            setattr(self, f'out_{tag}',  nn.Conv2d(channels, channels, 1))
            setattr(self, f'norm_{tag}', nn.BatchNorm2d(channels))

    def _reshape(self, t):
        B, C, H, W = t.shape
        return t.reshape(B, self.heads, C//self.heads, H*W).permute(0,1,3,2)

    def _attn(self, q, k, v):
        B, C, H, W = q.shape
        q_ = self._reshape(q); k_ = self._reshape(k); v_ = self._reshape(v)
        a  = torch.softmax(torch.matmul(q_, k_.transpose(-1,-2)) * self.scale, dim=-1)
        return torch.matmul(a, v_).permute(0,1,3,2).reshape(B, C, H, W)

    def forward(self, fr, fm):
        if fm.shape[-2:] != fr.shape[-2:]:
            fm = F.interpolate(fm, size=fr.shape[-2:], mode='bilinear', align_corners=False)
        fr = self.norm_r(fr + self.out_r(self._attn(self.q_r(fr), self.k_m(fm), self.v_m(fm))))
        fm = self.norm_m(fm + self.out_m(self._attn(self.q_m(fm), self.k_r(fr), self.v_r(fr))))
        return fr, fm


# ── Shared Heads ──────────────────────────────────────────────────────────────

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

    def __init__(self, mamba_depth=4, mamba_d_state=16,
                 pretrained=True, swin_weights=None, img_size=384, **kwargs):
        super().__init__()
        self.backbone    = SwinBackbone(pretrained=pretrained,
                                        swin_weights=swin_weights, input_size=img_size)
        self.fpn         = FeaturePyramidNetwork(
            in_channels_list=SwinBackbone.OUT_CHANNELS, out_channels=self.FPN_OUT)
        ch = self.BRANCH_CH
        self.res_proj    = nn.Sequential(nn.Conv2d(self.FPN_OUT,ch,1),
                                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.mamba_proj  = nn.Sequential(nn.Conv2d(self.FPN_OUT,ch,1),
                                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.mamba       = MambaBranch(channels=ch, depth=mamba_depth, d_state=mamba_d_state)
        self.texture_enh = TextureEnhancement(ch=ch)
        self.cross_attn  = CrossAttention(channels=ch, heads=8)
        self.fusion      = nn.Sequential(nn.Conv2d(ch*2,ch,1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.bnd_head    = BoundaryHead(ch)
        self.hires_bnd   = HighResBoundaryStream(self.FPN_OUT)
        self.unc_head    = UncertaintyHead(ch)
        self.decoder     = ProgressiveDecoder(self.FPN_OUT, ch)

    def freeze_backbone(self, n=2):
        self.backbone.freeze_stages(n)

    def forward(self, x):
        tgt = x.shape[2:]
        c1, c2, c3, c4 = self.backbone(x)
        raw  = self.fpn({'0':c1,'1':c2,'2':c3,'3':c4})
        fpn  = {'p2':raw['0'],'p3':raw['1'],'p4':raw['2'],'p5':raw['3']}

        feat_r = self.texture_enh(self.res_proj(fpn['p5']))
        feat_m = self.mamba(self.mamba_proj(fpn['p5']))
        if feat_m.shape[-2:] != feat_r.shape[-2:]:
            feat_m = F.interpolate(feat_m, size=feat_r.shape[-2:],
                                   mode='bilinear', align_corners=False)
        feat_r, feat_m = self.cross_attn(feat_r, feat_m)
        fused  = self.fusion(torch.cat([feat_r, feat_m], dim=1))

        bnd_c  = self.bnd_head(fused, tgt)
        bnd_f  = self.hires_bnd(fpn['p2'], bnd_c, tgt)
        unc    = self.unc_head(fused, tgt)
        seg    = F.interpolate(self.decoder(fused, fpn, bnd_c),
                               size=tgt, mode='bilinear', align_corners=False)
        return {'seg': seg, 'boundary': bnd_f,
                'boundary_coarse': bnd_c, 'uncertainty': unc}