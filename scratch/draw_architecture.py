import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# Setup output directory and file path
output_dir = "/home/ronit.28010/COD_Project/main2fig"
os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, "fig7_architecture_schematic.png")

# Configure plot
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=300)
ax.set_xlim(0, 18)
ax.set_ylim(0, 10.5)
ax.axis('off')

# Color Palette (Premium Muted)
color_img = '#f2f2f2'        # Gray for images
color_backbone = '#d6e4f0'   # Soft blue for backbone
color_fpn = '#e2f0d9'        # Soft green for FPN
color_texture = '#fff2cc'    # Soft yellow for texture branch
color_mamba = '#e1d5e7'      # Soft purple for mamba branch
color_attn = '#ffe6cc'       # Soft orange for attention
color_decoder = '#f8cecc'    # Soft red for progressive decoder
color_output = '#e8eaed'     # Light gray for outputs

# Helper function to draw a 3D-like box representing a feature map tensor
def draw_tensor_3d(ax, x, y, w, h, d, label, color, channels=""):
    # Main front face
    rect = patches.Rectangle((x, y), w, h, linewidth=0.8, edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    # Top face
    top_pts = [[x, y+h], [x+d, y+h+d], [x+w+d, y+h+d], [x+w, y+h]]
    top_poly = patches.Polygon(top_pts, linewidth=0.8, edgecolor='black', facecolor=color, alpha=0.7)
    ax.add_patch(top_poly)
    # Right face
    right_pts = [[x+w, y], [x+w+d, y+d], [x+w+d, y+h+d], [x+w, y+h]]
    right_poly = patches.Polygon(right_pts, linewidth=0.8, edgecolor='black', facecolor=color, alpha=0.8)
    ax.add_patch(right_poly)
    # Text labels
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontweight='bold', fontsize=8.5)
    if channels:
        ax.text(x + w + d + 0.1, y + h/2 + d/2, channels, ha='left', va='center', fontsize=7.5, fontstyle='italic')

# Helper to draw a standard process block
def draw_block(ax, x, y, w, h, label, color, border_style='-', text_weight='normal', font_size=8.5):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                                 linewidth=1, edgecolor='black', facecolor=color, linestyle=border_style)
    ax.add_patch(rect)
    # Handle multi-line labels
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontweight=text_weight, fontsize=font_size)

# Helper to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, label="", label_pos='top', color='black', style='-|>', line_w=1.2, is_dashed=False):
    ls = '--' if is_dashed else '-'
    arrow = ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=line_w, ls=ls,
                        shrinkA=0, shrinkB=0, patchA=None, patchB=None,
                        connectionstyle="arc3")
    )
    if label:
        lx, ly = (x1+x2)/2, (y1+y2)/2
        if label_pos == 'top':
            ax.text(lx, ly + 0.1, label, ha='center', va='bottom', fontsize=8, color=color)
        elif label_pos == 'bottom':
            ax.text(lx, ly - 0.1, label, ha='center', va='top', fontsize=8, color=color)
        elif label_pos == 'left':
            ax.text(lx - 0.1, ly, label, ha='right', va='center', fontsize=8, color=color)
        elif label_pos == 'right':
            ax.text(lx + 0.1, ly, label, ha='left', va='center', fontsize=8, color=color)

# ----------------- Left Section: Input & Backbone & FPN -----------------

# 1. Input Image
draw_block(ax, 0.4, 5.0, 1.4, 1.4, "Input Image\n(384x384, 3ch)", color_img, text_weight='bold')
draw_arrow(ax, 1.8, 5.7, 2.4, 5.7)

# 2. Hierarchical Backbone
draw_block(ax, 2.4, 4.2, 1.6, 3.0, "Hierarchical\nBackbone\n\nSwin-Tiny\nor\nResNet-50", color_backbone, text_weight='bold')

# Arrow from Backbone to FPN Neck
draw_arrow(ax, 4.0, 5.7, 4.6, 5.7)

# 3. FPN Neck Vertical Stack (256 channels)
# Background bounding box for FPN
fpn_bg = patches.Rectangle((4.6, 1.0), 2.2, 8.5, linewidth=1, edgecolor='#999999', facecolor='#fbfbfb', linestyle='--', alpha=0.5)
ax.add_patch(fpn_bg)
ax.text(5.7, 9.25, "FPN Neck (256ch)", ha='center', va='bottom', fontweight='bold', color='#555555')

# Draw FPN levels: p2, p3, p4, p5
draw_tensor_3d(ax, 5.1, 1.5, 1.0, 0.5, 0.25, "p2", color_fpn, "192x192")
draw_tensor_3d(ax, 5.1, 3.2, 1.0, 0.5, 0.25, "p3", color_fpn, "96x96")
draw_tensor_3d(ax, 5.1, 4.9, 1.0, 0.5, 0.25, "p4", color_fpn, "48x48")
draw_tensor_3d(ax, 5.1, 6.6, 1.0, 0.5, 0.25, "p5", color_fpn, "24x24")

# Upward fusion arrows inside FPN (from shallower to deeper)
draw_arrow(ax, 5.6, 2.3, 5.6, 3.2, style='->', color='#777777', line_w=1.0)
draw_arrow(ax, 5.6, 4.0, 5.6, 4.9, style='->', color='#777777', line_w=1.0)
draw_arrow(ax, 5.6, 5.7, 5.6, 6.6, style='->', color='#777777', line_w=1.0)

# ----------------- Middle Section: Parallel Branches on p5 -----------------

# Arrow from p5 to split point
draw_arrow(ax, 6.5, 7.0, 7.3, 7.0)

# Project and Split point
draw_block(ax, 7.3, 6.6, 1.0, 0.8, "Project\n512ch", '#d6e4f0')
# Split arrows
draw_arrow(ax, 7.8, 7.4, 7.8, 8.8)  # up to texture branch
draw_arrow(ax, 7.8, 6.6, 7.8, 5.2)  # down to mamba branch

# Branch 1: Texture Enhancement (Top)
tex_bg = patches.Rectangle((8.6, 7.8), 2.8, 2.2, linewidth=1, edgecolor='#999999', facecolor='#fffdf5', linestyle='-', alpha=0.95)
ax.add_patch(tex_bg)
ax.text(10.0, 9.75, "Texture Enhancement Branch", ha='center', va='bottom', fontweight='bold', color='#c49a45')

draw_block(ax, 8.8, 8.8, 1.1, 0.6, "Dilated\nConvolutions", color_texture)
draw_block(ax, 10.1, 8.8, 1.1, 0.6, "Sobel Edge\nFilter", color_texture)
draw_block(ax, 9.4, 8.0, 1.2, 0.5, "Feature Fusion\nhf * (1 + edge)", color_texture)

draw_arrow(ax, 7.8, 8.8, 8.8, 9.1)
draw_arrow(ax, 7.8, 8.8, 10.1, 9.1)
draw_arrow(ax, 9.35, 8.8, 9.7, 8.5)
draw_arrow(ax, 10.65, 8.8, 10.3, 8.5)

# Branch 2: Mamba Sequence Modeling (Bottom)
mamba_bg = patches.Rectangle((8.6, 4.0), 2.8, 2.2, linewidth=1, edgecolor='#999999', facecolor='#faf5ff', linestyle='-', alpha=0.95)
ax.add_patch(mamba_bg)
ax.text(10.0, 5.95, "Mamba Sequence Branch", ha='center', va='bottom', fontweight='bold', color='#764a8c')

draw_block(ax, 8.8, 5.0, 1.1, 0.6, "1D Flattening\n(L = HxW)", color_mamba)
draw_block(ax, 10.1, 5.0, 1.1, 0.6, "4 Mamba Blocks\n(Selective Scan)", color_mamba)
draw_block(ax, 9.4, 4.2, 1.2, 0.5, "2D Reshape\n(B, 512, H, W)", color_mamba)

draw_arrow(ax, 7.8, 5.2, 8.8, 5.3)
draw_arrow(ax, 9.9, 5.3, 10.1, 5.3)
draw_arrow(ax, 10.6, 5.0, 10.2, 4.7)

# ----------------- Middle-Right: Cross-Attention Fusion -----------------

# Arrow from Texture Branch to Cross-Attention
draw_arrow(ax, 11.4, 8.2, 12.1, 8.2, label="feat_r", label_pos='top')
# Arrow from Mamba Branch to Cross-Attention
draw_arrow(ax, 11.4, 4.4, 12.1, 4.4, label="feat_m", label_pos='bottom')

# Cross-Attention Bounding Box
attn_bg = patches.Rectangle((12.1, 4.0), 1.8, 4.7, linewidth=1.2, edgecolor='#dd8452', facecolor='#fff7f0', alpha=0.95)
ax.add_patch(attn_bg)
ax.text(13.0, 8.45, "Cross-Attention\nFusion", ha='center', va='bottom', fontweight='bold', color='#b25424')

# Inside Cross-Attention: Q, K, V Projections
draw_block(ax, 12.3, 7.2, 0.6, 0.4, "Q_r, K_r\nV_r", color_attn, font_size=7)
draw_block(ax, 12.3, 5.0, 0.6, 0.4, "Q_m, K_m\nV_m", color_attn, font_size=7)

# Intersecting Query/Key/Value exchange arrows
draw_arrow(ax, 12.9, 7.2, 13.5, 5.4, style='->', color='#b25424', line_w=0.8, is_dashed=True)
draw_arrow(ax, 12.9, 5.4, 13.5, 7.2, style='->', color='#b25424', line_w=0.8, is_dashed=True)

# Outputs of Cross-Attention to Concatenation & Fusion
draw_arrow(ax, 13.9, 7.4, 14.5, 7.4)
draw_arrow(ax, 13.9, 5.2, 14.5, 5.2)

# Concatenation and 1x1 Fusion block
draw_block(ax, 14.5, 4.6, 1.2, 3.4, "Feature Concat\n&\n1x1 Conv + BN\n(Fused: 512ch)", color_decoder, text_weight='bold')

# Arrow from Fused features
draw_arrow(ax, 15.7, 6.3, 16.5, 6.3, label="fused", label_pos='top')

# ----------------- Progressive Decoder cascade (Bottom) -----------------
dec_bg = patches.Rectangle((7.3, 0.8), 8.4, 2.7, linewidth=1.2, edgecolor='#999999', facecolor='#faf0f0', linestyle='-', alpha=0.95)
ax.add_patch(dec_bg)
ax.text(11.5, 3.15, "Progressive Decoder Cascade", ha='center', va='bottom', fontweight='bold', color='#9c3d3d')

# ASPP Module
draw_block(ax, 7.5, 1.4, 1.1, 1.5, "ASPP\n(Rates: 1, 6,\n12, 18)", color_decoder, text_weight='bold')
# Arrow from Fused Features down to Progressive Decoder (ASPP)
draw_arrow(ax, 15.1, 4.6, 15.1, 2.5)
draw_arrow(ax, 15.1, 2.5, 8.6, 2.5)

# Cascaded refinement blocks
draw_block(ax, 9.0, 1.5, 1.1, 0.6, "Refine Stage 4\nConcat (ASPP, p4)", color_decoder, font_size=7.5)
draw_block(ax, 10.5, 1.5, 1.1, 0.6, "Refine Stage 3\nConcat (R4, p3)", color_decoder, font_size=7.5)
draw_block(ax, 12.0, 1.5, 1.1, 0.6, "Refine Stage 2\nConcat (R3, p2, bnd)", color_decoder, font_size=7.5)
draw_block(ax, 13.5, 1.5, 1.0, 0.6, "Segmentation\nHead (1x1 Conv)", color_decoder, font_size=7.5)

# Arrows in the cascade
draw_arrow(ax, 8.6, 1.8, 9.0, 1.8)
draw_arrow(ax, 10.1, 1.8, 10.5, 1.8)
draw_arrow(ax, 11.6, 1.8, 12.0, 1.8)
draw_arrow(ax, 13.1, 1.8, 13.5, 1.8)

# Skip connections from FPN (p4, p3, p2) to the Progressive Decoder
# Connect p4 to Refine Stage 4
draw_arrow(ax, 6.1, 5.15, 9.55, 5.15, is_dashed=True, color='#555555')
draw_arrow(ax, 9.55, 5.15, 9.55, 2.1, is_dashed=True, color='#555555', label="p4", label_pos='left')

# Connect p3 to Refine Stage 3
draw_arrow(ax, 6.1, 3.45, 11.05, 3.45, is_dashed=True, color='#555555')
draw_arrow(ax, 11.05, 3.45, 11.05, 2.1, is_dashed=True, color='#555555', label="p3", label_pos='left')

# Connect p2 to Refine Stage 2
draw_arrow(ax, 6.1, 1.75, 12.55, 1.75, is_dashed=True, color='#555555')
# Arrow for p2 skip label
ax.text(6.8, 1.85, "p2 skip", fontsize=7.5, color='#555555')

# ----------------- Right Section: Heads & Multi-head Outputs -----------------

# Background bounding box for Outputs
out_bg = patches.Rectangle((16.5, 0.5), 1.3, 9.5, linewidth=1.2, edgecolor='#777777', facecolor='#f7f7f7', alpha=0.9)
ax.add_patch(out_bg)
ax.text(17.15, 9.7, "Multi-head\nOutputs", ha='center', va='bottom', fontweight='bold', color='#333333')

# 1. Coarse Boundary Head
draw_block(ax, 16.6, 8.0, 1.1, 0.7, "Coarse\nBoundary\n(bnd_c)", color_output, text_weight='bold', font_size=8)
draw_arrow(ax, 15.7, 6.3, 16.1, 6.3)
draw_arrow(ax, 16.1, 6.3, 16.1, 8.3)
draw_arrow(ax, 16.1, 8.3, 16.6, 8.3)

# Feed Coarse Boundary back to Decoder Stage 2 skip
draw_arrow(ax, 17.15, 8.0, 17.15, 7.8)
draw_arrow(ax, 17.15, 7.8, 16.1, 7.8)
draw_arrow(ax, 16.1, 7.8, 16.1, 2.8)
draw_arrow(ax, 16.1, 2.8, 12.8, 2.8)
draw_arrow(ax, 12.8, 2.8, 12.8, 2.1, label="bnd_c", label_pos='left')

# 2. High-Res Fine Boundary Stream
draw_block(ax, 16.6, 5.5, 1.1, 0.8, "High-Res Stream\nFine Boundary\n(bnd_f)", color_output, text_weight='bold', font_size=7.5)
# Input from p2 to Fine Boundary Stream (long skip connection)
draw_arrow(ax, 5.6, 1.5, 5.6, 0.4, is_dashed=True, color='#555555')
draw_arrow(ax, 5.6, 0.4, 16.3, 0.4, is_dashed=True, color='#555555')
draw_arrow(ax, 16.3, 0.4, 16.3, 5.9, is_dashed=True, color='#555555', label="p2 skip", label_pos='left')
draw_arrow(ax, 16.3, 5.9, 16.6, 5.9)
# Also takes coarse boundary as input
draw_arrow(ax, 17.15, 8.0, 17.15, 6.3)

# 3. Uncertainty Head
draw_block(ax, 16.6, 3.2, 1.1, 0.7, "Uncertainty\nHead\n(unc)", color_output, text_weight='bold', font_size=8)
draw_arrow(ax, 15.9, 6.3, 15.9, 3.5)
draw_arrow(ax, 15.9, 3.5, 16.6, 3.5)

# 4. Final Segmentation Map
draw_block(ax, 16.6, 0.9, 1.1, 0.7, "Segmentation\nMap\n(seg)", color_output, text_weight='bold', font_size=8)
draw_arrow(ax, 14.5, 1.8, 16.6, 1.8) # from progressive decoder to output

plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()
print("Saved 100% accurate vector schematic diagram to", fig_path)
