import torch
try:
    from thop import profile
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
    from thop import profile

from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro

def compute_overhead():
    model = BMambaModel_Pro(pretrained=False, img_size=352)
    
    # We measure for a single image 1x3x352x352
    dummy_input = torch.randn(1, 3, 352, 352)
    
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    # GMacs (Giga MACs = 1e9)
    # Params in Millions
    print("=== Computational Overhead ===")
    print(f"Parameters: {params / 1e6:.2f} M")
    print(f"GFLOPs (MACs): {macs / 1e9:.2f} G")
    print("==============================")

if __name__ == '__main__':
    compute_overhead()
