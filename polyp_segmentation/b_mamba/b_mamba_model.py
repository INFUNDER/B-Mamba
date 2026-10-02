"""
Model: Swin-T + Boundary-Aware Mamba (B-Mamba)
============================================================
Specifically designed for Medical Image Segmentation (e.g., Polyps).
The Mamba SSM's dynamic parameters (B, C, Delta) are conditioned
on high-frequency boundary maps to handle fuzzy lesion borders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.ops import FeaturePyramidNetwork

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


class BoundaryAwareMambaBlock(nn.Module):
    """
    Novelty: Condition the SSM parameters on a boundary feature map.
    """
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
        
        # We project the SSM parameters from [x_conv, boundary_features]
        # x_conv: d_inner, boundary: d_inner
        self.x_proj  = nn.Linear(d_inner * 2, d_state*2+d_inner, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log   = nn.Parameter(torch.log(A))
        self.D       = nn.Parameter(torch.ones(d_inner))
        self.out_proj= nn.Linear(d_inner, dim, bias=False)

    def forward(self, x, boundary_seq):
        residual = x
        x = self.norm(x)
        
        B_sz, L, _ = x.shape
        xz      = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv  = self.conv1d(x_in.transpose(1,2))[:,:,:L].transpose(1,2)
        x_conv  = self.act(x_conv)
        
        # Concatenate standard features with boundary features
        # boundary_seq must have same sequence length and d_inner channels
        x_cond  = torch.cat([x_conv, boundary_seq], dim=-1)
        
        x_dbl   = self.x_proj(x_cond)
        B_ssm   = x_dbl[..., :self.d_state]
        C_ssm   = x_dbl[..., self.d_state:self.d_state*2]
        delta   = F.softplus(self.dt_proj(x_dbl[..., self.d_state*2:]))
        
        y = selective_scan_pytorch(x_conv, delta, self.A_log, B_ssm, C_ssm, self.D)
        out = self.out_proj(y * self.act(z))
        
        return out + residual


class BMambaBranch(nn.Module):
    def __init__(self, channels=512, depth=4, d_state=16):
        super().__init__()
        self.proj_in  = nn.Conv2d(channels, channels, 1)
        # Expand boundary features to match d_inner (channels * expand)
        self.bnd_proj = nn.Conv2d(channels, channels * 2, 1) 
        
        self.blocks   = nn.ModuleList([BoundaryAwareMambaBlock(channels, d_state=d_state, expand=2)
                                       for _ in range(depth)])
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.norm     = nn.LayerNorm(channels)

    def forward(self, x, boundary_feat):
        B_sz, C, H, W = x.shape
        x   = self.proj_in(x)
        
        # Project and flatten boundary features
        bnd = self.bnd_proj(boundary_feat) # (B, channels*2, H, W)
        bnd_seq = bnd.flatten(2).transpose(1, 2)
        
        seq = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            seq = blk(seq, bnd_seq)
            
        seq = self.norm(seq)
        return self.proj_out(seq.transpose(1,2).reshape(B_sz, C, H, W))


# --- The rest can reuse components from model.py, just updating the full model class ---

class SwinBackbone(nn.Module):
    OUT_CHANNELS = [96, 192, 384, 768]
    def __init__(self, pretrained=True, input_size=384):
        super().__init__()
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True, out_indices=(0,1,2,3),
            img_size=(input_size, input_size))

    def forward(self, x):
        return [f.permute(0,3,1,2).contiguous() for f in self.swin(x)]

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
        return self.fuse(torch.cat([hf*(1+edge), edge], dim=1)), hf*(1+edge) # Return enhanced edge features too

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
        self.ref2 = nn.Sequential(nn.Conv2d(64+32, 32,3,padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True))
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, fused, fpn):
        x = self.aspp(fused)
        x = F.interpolate(x, size=fpn['p4'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.ref4(torch.cat([x, self.lat4(fpn['p4'])], dim=1))
        x = F.interpolate(x, size=fpn['p3'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.ref3(torch.cat([x, self.lat3(fpn['p3'])], dim=1))
        x = F.interpolate(x, size=fpn['p2'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.ref2(torch.cat([x, self.lat2(fpn['p2'])], dim=1))
        return self.head(x)


class BMambaModel(nn.Module):
    FPN_OUT   = 256
    BRANCH_CH = 512

    def __init__(self, mamba_depth=4, mamba_d_state=16, pretrained=True, img_size=352):
        super().__init__()
        self.backbone    = SwinBackbone(pretrained=pretrained, input_size=img_size)
        self.fpn         = FeaturePyramidNetwork(
            in_channels_list=SwinBackbone.OUT_CHANNELS, out_channels=self.FPN_OUT)
        
        ch = self.BRANCH_CH
        self.res_proj    = nn.Sequential(nn.Conv2d(self.FPN_OUT,ch,1),
                                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.mamba_proj  = nn.Sequential(nn.Conv2d(self.FPN_OUT,ch,1),
                                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        
        self.texture_enh = TextureEnhancement(ch=ch)
        self.b_mamba     = BMambaBranch(channels=ch, depth=mamba_depth, d_state=mamba_d_state)
        
        self.fusion      = nn.Sequential(nn.Conv2d(ch*2,ch,1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.decoder     = ProgressiveDecoder(self.FPN_OUT, ch)

    def forward(self, x):
        tgt = x.shape[2:]
        c1, c2, c3, c4 = self.backbone(x)
        raw  = self.fpn({'0':c1,'1':c2,'2':c3,'3':c4})
        fpn  = {'p2':raw['0'],'p3':raw['1'],'p4':raw['2'],'p5':raw['3']}

        # 1. Extract texture & explicit boundaries
        feat_r, bnd_feat = self.texture_enh(self.res_proj(fpn['p5']))
        
        # 2. Mamba conditioned on boundaries
        feat_m = self.b_mamba(self.mamba_proj(fpn['p5']), bnd_feat)
        
        if feat_m.shape[-2:] != feat_r.shape[-2:]:
            feat_m = F.interpolate(feat_m, size=feat_r.shape[-2:],
                                   mode='bilinear', align_corners=False)
                                   
        # 3. Concatenate and decode
        fused  = self.fusion(torch.cat([feat_r, feat_m], dim=1))
        seg    = F.interpolate(self.decoder(fused, fpn),
                               size=tgt, mode='bilinear', align_corners=False)
                               
        return {'seg': seg}

# ==============================================================================
# BMambaModel_Pro (SOTA Version with Deep Supervision & Swin-Base)
# ==============================================================================

class SwinBaseBackbone(nn.Module):
    OUT_CHANNELS = [128, 256, 512, 1024]
    def __init__(self, pretrained=True, input_size=352):
        super().__init__()
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            features_only=True, out_indices=(0,1,2,3),
            img_size=(input_size, input_size))

    def forward(self, x):
        return [f.permute(0,3,1,2).contiguous() for f in self.swin(x)]

class ProgressiveDecoder_Pro(nn.Module):
    def __init__(self, fpn_ch=256, fused_ch=512):
        super().__init__()
        self.aspp = ASPP(fused_ch, 256)
        self.lat4 = nn.Sequential(nn.Conv2d(fpn_ch,128,1), nn.ReLU(inplace=True))
        self.lat3 = nn.Sequential(nn.Conv2d(fpn_ch, 64,1), nn.ReLU(inplace=True))
        self.lat2 = nn.Sequential(nn.Conv2d(fpn_ch, 32,1), nn.ReLU(inplace=True))
        self.ref4 = nn.Sequential(nn.Conv2d(256+128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.ref3 = nn.Sequential(nn.Conv2d(128+64,  64,3,padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True))
        self.ref2 = nn.Sequential(nn.Conv2d(64+32, 32,3,padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True))
        
        # Deep Supervision Heads
        self.head4 = nn.Conv2d(128, 1, 1)
        self.head3 = nn.Conv2d(64, 1, 1)
        self.head2 = nn.Conv2d(32, 1, 1)

    def forward(self, fused, fpn):
        x4 = self.aspp(fused)
        x4 = F.interpolate(x4, size=fpn['p4'].shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.ref4(torch.cat([x4, self.lat4(fpn['p4'])], dim=1))
        out4 = self.head4(x4)
        
        x3 = F.interpolate(x4, size=fpn['p3'].shape[-2:], mode='bilinear', align_corners=False)
        x3 = self.ref3(torch.cat([x3, self.lat3(fpn['p3'])], dim=1))
        out3 = self.head3(x3)
        
        x2 = F.interpolate(x3, size=fpn['p2'].shape[-2:], mode='bilinear', align_corners=False)
        x2 = self.ref2(torch.cat([x2, self.lat2(fpn['p2'])], dim=1))
        out2 = self.head2(x2)
        
        return out2, out3, out4

class BMambaModel_Pro(nn.Module):
    FPN_OUT   = 256
    BRANCH_CH = 512

    def __init__(self, mamba_depth=4, mamba_d_state=16, pretrained=True, img_size=352):
        super().__init__()
        self.backbone    = SwinBaseBackbone(pretrained=pretrained, input_size=img_size)
        self.fpn         = FeaturePyramidNetwork(
            in_channels_list=SwinBaseBackbone.OUT_CHANNELS, out_channels=self.FPN_OUT)
        
        ch = self.BRANCH_CH
        self.res_proj    = nn.Sequential(nn.Conv2d(self.FPN_OUT,ch,1),
                                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.mamba_proj  = nn.Sequential(nn.Conv2d(self.FPN_OUT,ch,1),
                                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        
        self.texture_enh = TextureEnhancement(ch=ch)
        self.b_mamba     = BMambaBranch(channels=ch, depth=mamba_depth, d_state=mamba_d_state)
        
        self.fusion      = nn.Sequential(nn.Conv2d(ch*2,ch,1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.decoder     = ProgressiveDecoder_Pro(self.FPN_OUT, ch)

    def forward(self, x):
        tgt = x.shape[2:]
        c1, c2, c3, c4 = self.backbone(x)
        raw  = self.fpn({'0':c1,'1':c2,'2':c3,'3':c4})
        fpn  = {'p2':raw['0'],'p3':raw['1'],'p4':raw['2'],'p5':raw['3']}

        feat_r, bnd_feat = self.texture_enh(self.res_proj(fpn['p5']))
        feat_m = self.b_mamba(self.mamba_proj(fpn['p5']), bnd_feat)
        
        if feat_m.shape[-2:] != feat_r.shape[-2:]:
            feat_m = F.interpolate(feat_m, size=feat_r.shape[-2:], mode='bilinear', align_corners=False)
                                   
        fused  = self.fusion(torch.cat([feat_r, feat_m], dim=1))
        out2, out3, out4 = self.decoder(fused, fpn)
        
        seg = F.interpolate(out2, size=tgt, mode='bilinear', align_corners=False)
        seg3 = F.interpolate(out3, size=tgt, mode='bilinear', align_corners=False)
        seg4 = F.interpolate(out4, size=tgt, mode='bilinear', align_corners=False)
                               
        if self.training:
            return {'seg': seg, 'seg3': seg3, 'seg4': seg4}
        else:
            return {'seg': seg}
