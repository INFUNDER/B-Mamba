import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Create directories
output_dir = "/home/ronit.28010/COD_Project/main2fig"
os.makedirs(output_dir, exist_ok=True)

# Copy existing files
# 1. Qualitative CAM predictions
shutil.copy2(
    "/home/ronit.28010/COD_Project/visualizations/comparison_CAM.png",
    os.path.join(output_dir, "comparison_CAM.png")
)
# 2. Qualitative NonCAM predictions
shutil.copy2(
    "/home/ronit.28010/COD_Project/visualizations/comparison_NonCAM.png",
    os.path.join(output_dir, "comparison_NonCAM.png")
)
# 3. Model Architecture schematic
shutil.copy2(
    "/home/ronit.28010/.gemini/antigravity-ide/brain/54f780f6-1cbf-4534-a9d1-24591cd0acd7/model_architecture_1780569917563.png",
    os.path.join(output_dir, "fig7_architecture_schematic.png")
)
print("Copied qualitative predictions and architecture diagram.")

# ----------------- Fig 5: Best Checkpoint Bar Chart -----------------
# Let's use a nice professional styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={'width_ratios': [1.8, 1]}, dpi=300)

models = ['ResNet + Transformer', 'ResNet + Mamba', 'Swin + Mamba']
colors = ['#5b73a7', '#d88354', '#57a868'] # Premium muted colors

# Left subplot: Sm, wFm, Em_mean (higher is better)
metrics_left = ['S$_m$', 'wF$_m$', 'E$_m$ (mean)']
resnet_trans_left = [0.8045, 0.6268, 0.8348]
resnet_mamba_left = [0.8097, 0.6408, 0.8387]
swin_mamba_left = [0.8401, 0.6979, 0.8742]

x = np.arange(len(metrics_left))
width = 0.24

axes[0].bar(x - width, resnet_trans_left, width, label='ResNet + Transformer', color=colors[0], edgecolor='black', linewidth=0.7)
axes[0].bar(x, resnet_mamba_left, width, label='ResNet + Mamba', color=colors[1], edgecolor='black', linewidth=0.7)
axes[0].bar(x + width, swin_mamba_left, width, label='Swin + Mamba', color=colors[2], edgecolor='black', linewidth=0.7)

axes[0].set_title(r'Higher is Better ($\uparrow$)', pad=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_left)
axes[0].set_ylim(0.5, 1.0)
axes[0].set_ylabel('Score', fontweight='bold')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend(loc='lower left', frameon=True)

# Right subplot: MAE (lower is better)
mae_values = [0.0435, 0.0424, 0.0341]
axes[1].bar(models, mae_values, width=0.5, color=colors, edgecolor='black', linewidth=0.7)
axes[1].set_title(r'Mean Absolute Error ($\downarrow$)', pad=12, fontweight='bold')
axes[1].set_ylabel('MAE', fontweight='bold')
axes[1].set_ylim(0, 0.05)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

# Rotate labels slightly for readability
axes[1].set_xticklabels(models, rotation=15, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig5_best_checkpoint_bar.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated Fig 5.")

# ----------------- Fig 6: Gain Decomposition -----------------
# Purple for decoder gain (ResNet+Mamba - ResNet+Transformer)
# Red/Coral for backbone gain (Swin+Mamba - ResNet+Mamba)
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

metrics = ['S$_m$', 'wF$_m$', 'MAE Improvement']
decoder_gains = [0.8097 - 0.8045, 0.6408 - 0.6268, 0.0435 - 0.0424]
backbone_gains = [0.8401 - 0.8097, 0.6979 - 0.6408, 0.0424 - 0.0341]

x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, decoder_gains, width, label=r'Decoder Gain (Transformer $\rightarrow$ Mamba)', color='#8064a2', edgecolor='black', linewidth=0.7)
ax.bar(x + width/2, backbone_gains, width, label=r'Backbone Gain (ResNet $\rightarrow$ Swin)', color='#c0504d', edgecolor='black', linewidth=0.7)

ax.set_title('Architectural Gain Decomposition (CAM-only)', pad=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Performance Gain', fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper left', frameon=True)

# Add values above bars
for i in range(len(metrics)):
    ax.text(i - width/2, decoder_gains[i] + 0.001, f"+{decoder_gains[i]:.4f}", ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, backbone_gains[i] + 0.001, f"+{backbone_gains[i]:.4f}", ha='center', va='bottom', fontsize=9)

ax.set_ylim(0, 0.07)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig6_gain_decomposition.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated Fig 6.")
