"""
Shared COD Evaluation Metrics — Fixed v7
=========================================
Same for ALL 3 models. Ensures fair comparison.

Fixes vs original v1-v3:
  1. Em  — correct scalar normalisation + threshold sweep (Em_mean, Em_max)
           + degenerate guard (Em now always in [0,1])
  2. wFm — distance-from-boundary weights (not distance-from-centroid)
  3. Sm  — correct centre-of-mass centroid via argwhere
  4. Fm  — full threshold sweep (Fm_mean, Fm_max)
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, convolve


class CODMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mae_list     = []
        self.sm_list      = []
        self.em_mean_list = []
        self.em_max_list  = []
        self.wfm_list     = []
        self.fm_list      = []

    def update(self, pred_logits, gt_mask):
        pred = torch.sigmoid(pred_logits).squeeze(1).cpu().numpy()
        gt   = gt_mask.squeeze(1).cpu().numpy()
        for p, g in zip(pred, gt):
            self.mae_list.append(self._mae(p, g))
            self.sm_list.append(self._s_measure(p, g))
            em_m, em_x = self._e_measure_sweep(p, g)
            self.em_mean_list.append(em_m)
            self.em_max_list.append(em_x)
            self.wfm_list.append(self._weighted_f(p, g))
            fm_m, fm_x = self._f_measure_sweep(p, g)
            self.fm_list.append((fm_m, fm_x))

    def compute(self):
        if not self.mae_list:
            return {}
        return {
            'MAE':     float(np.mean(self.mae_list)),
            'Sm':      float(np.mean(self.sm_list)),
            'Em_mean': float(np.mean(self.em_mean_list)),
            'Em_max':  float(np.mean(self.em_max_list)),
            'wFm':     float(np.mean(self.wfm_list)),
            'Fm_mean': float(np.mean([f[0] for f in self.fm_list])),
            'Fm_max':  float(np.mean([f[1] for f in self.fm_list])),
        }

    # ── MAE ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _mae(pred, gt):
        return float(np.abs(pred - gt).mean())

    # ── S-measure ─────────────────────────────────────────────────────────────
    @staticmethod
    def _s_measure(pred, gt, alpha=0.5):
        y = gt.mean()
        if y == 0: return float(1.0 - pred.mean())
        if y == 1: return float(pred.mean())

        def object_score(p, g):
            u    = g.mean()
            s_fg = (p * g).sum() / (g.sum() + 1e-8)
            s_bg = ((1-p)*(1-g)).sum() / ((1-g).sum() + 1e-8)
            return u * s_fg + (1 - u) * s_bg

        def region_score(p, g):
            h, w   = g.shape
            coords = np.argwhere(g > 0.5)
            if len(coords) == 0: return 0.0
            cy, cx = coords.mean(axis=0)
            cx = int(np.clip(cx, 1, w-1))
            cy = int(np.clip(cy, 1, h-1))
            quads   = [(p[:cy,:cx],g[:cy,:cx]),(p[:cy,cx:],g[:cy,cx:]),
                       (p[cy:,:cx],g[cy:,:cx]),(p[cy:,cx:],g[cy:,cx:])]
            s, tg   = 0.0, g.sum() + 1e-8
            for pq, gq in quads:
                if gq.size == 0: continue
                s += (gq.sum()/tg) * (2*(pq*gq).sum()+1e-8) / (pq.sum()+gq.sum()+1e-8)
            return s

        return alpha * object_score(pred, gt) + (1-alpha) * region_score(pred, gt)

    # ── E-measure ─────────────────────────────────────────────────────────────
    @staticmethod
    def _e_measure_at(pred_b, gt):
        h, w = gt.shape
        gt_size = h * w
        gt_fg_num = np.count_nonzero(gt)
        
        if gt_fg_num == 0:
            return float(np.count_nonzero(pred_b == 0) / (gt_size + 1e-8))
        elif gt_fg_num == gt_size:
            return float(np.count_nonzero(pred_b == 1) / (gt_size + 1e-8))
            
        mu_p = pred_b.mean()
        mu_g = gt.mean()
        
        align_p = pred_b - mu_p
        align_g = gt - mu_g
        
        align_mat = 2.0 * align_p * align_g / (align_p**2 + align_g**2 + 1e-8)
        enhanced = (align_mat + 1.0)**2 / 4.0
        return float(enhanced.mean())

    @classmethod
    def _e_measure_sweep(cls, pred, gt, num_thresh=255):
        thresholds = np.linspace(1/255, 254/255, num_thresh)
        scores = np.clip([cls._e_measure_at((pred>=t).astype(np.float32), gt)
                          for t in thresholds], 0.0, 1.0)
        return float(scores.mean()), float(scores.max())

    # ── Weighted F-measure ────────────────────────────────────────────────────
    @staticmethod
    def _weighted_f(pred, gt, beta_sq=0.3):
        gt_b = (gt > 0.5).astype(np.float32)
        if gt_b.sum() == 0:
            return 0.0

        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = distance_transform_edt(gt_b == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt_b)
        
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt_b == 0] = Et[Idxt[0][gt_b == 0], Idxt[1][gt_b == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        shape = (7, 7)
        sigma = 5
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        K = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        K[K < np.finfo(K.dtype).eps * K.max()] = 0
        sumh = K.sum()
        if sumh != 0:
            K /= sumh

        EA = convolve(Et, weights=K, mode='constant', cval=0.0)

        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt_b > 0.5, np.minimum(E, EA), E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt_b == 0, 2.0 - np.exp(np.log(0.5) / 5.0 * Dst), np.ones_like(gt_b))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt_b) - np.sum(Ew[gt_b > 0.5])
        FPw = np.sum(Ew[gt_b == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        R = 1.0 - np.mean(Ew[gt_b > 0.5])
        P = TPw / (TPw + FPw + 1e-8)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1.0 + beta_sq) * R * P / (R + beta_sq * P + 1e-8)

        return float(Q)

    # ── F-measure sweep ───────────────────────────────────────────────────────
    @staticmethod
    def _f_measure_sweep(pred, gt, beta_sq=0.3, num_thresh=255):
        gt_b   = (gt > 0.5).astype(np.float32)
        scores = []
        for t in np.linspace(1/255, 254/255, num_thresh):
            pb   = (pred >= t).astype(np.float32)
            tp   = (pb * gt_b).sum()
            fp   = (pb * (1-gt_b)).sum()
            fn   = ((1-pb) * gt_b).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            scores.append((1+beta_sq)*prec*rec / (beta_sq*prec+rec+1e-8))
        scores = np.array(scores)
        return float(scores.mean()), float(scores.max())