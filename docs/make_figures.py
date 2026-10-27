#!/usr/bin/env python3
"""
Figure generation scripts for PGDA paper.
Run from the docs/ directory: python make_figures.py
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def fig_training_curves():
    """Figure: Training curves for all agents."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Mock data - replace with actual training logs
    steps = np.linspace(0, 1_000_000, 50)
    
    agents = {
        'Baseline (A)': {'color': '#1f77b4', 'style': '-'},
        'PGDA (B)': {'color': '#ff7f0e', 'style': '-'},
        'No-Cue (E)': {'color': '#2ca02c', 'style': '--'},
        'Oracle (C)': {'color': '#d62728', 'style': ':'},
        'No-DR (D)': {'color': '#9467bd', 'style': '-.'},
    }
    
    for name, style in agents.items():
        # Logistic-like curve
        final = {'Baseline (A)': 0.79, 'PGDA (B)': 0.79, 'No-Cue (E)': 0.84, 
                 'Oracle (C)': 0.81, 'No-DR (D)': 0.92}[name]
        curve = final / (1 + np.exp(-(steps - 400_000) / 100_000))
        ax.plot(steps / 1e6, curve, label=name, color=style['color'], 
                linestyle=style['style'], linewidth=2)
    
    ax.set_xlabel('Training Steps (millions)')
    ax.set_ylabel('Success Rate')
    ax.set_title('Training Curves: PGDA vs Ablations')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'training_curves.png'}")


def fig_mismatch_test():
    """Figure: Mismatch test results."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    conditions = ['Matched', 'Mismatched', 'No-Texture']
    sim_means = [0.79, 0.68, 0.15]
    sim_stds = [0.03, 0.04, 0.08]
    real_means = [0.967, 0.20, 0.033]
    real_stds = [0.02, 0.05, 0.02]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax.bar(x - width/2, sim_means, width, yerr=sim_stds, 
           label='Simulation (5 seeds)', color='#1f77b4', capsize=5)
    ax.bar(x + width/2, real_means, width, yerr=real_stds,
           label='Real UR3e (30 eps each)', color='#ff7f0e', capsize=5)
    
    ax.set_ylabel('Success Rate')
    ax.set_title('Mismatch Test: PGDA (Agent B) Relies on Material Cue')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mismatch_test.png', dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'mismatch_test.png'}")


def fig_ablation():
    """Figure: Ablation study comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    agents = ['No-DR (D)', 'Baseline (A)', 'No-Cue (E)', 'PGDA (B)', 'Oracle (C)']
    means = [0.92, 0.79, 0.84, 0.79, 0.81]
    stds = [0.024, 0.032, 0.028, 0.032, 0.031]
    colors = ['#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    bars = ax.barh(agents, means, xerr=stds, color=colors, capsize=5, edgecolor='black')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean:.0%} ± {std:.0%}', va='center', fontsize=10)
    
    ax.set_xlabel('Success Rate')
    ax.set_title('Ablation Study: Does PGDA Need the Visual Cue?')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Annotate key finding
    ax.annotate('No-Cue ≥ PGDA\n→ cue not needed', xy=(0.84, 2), xytext=(0.55, 2.5),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_study.png', dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'ablation_study.png'}")


def fig_task2():
    """Figure: Task 2 sliding results."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    policies = ['Reactive\n(react to contact)', 'Committed\n(push before contact)']
    gains = [5.1, 11.8]
    cis = [[3.8, 6.4], [10.3, 13.4]]  # 95% bootstrap CI
    
    colors = ['#2ca02c', '#d62728']
    bars = ax.bar(policies, gains, color=colors, edgecolor='black', width=0.6)
    
    for bar, gain, ci in zip(bars, gains, cis):
        ax.errorbar(bar.get_x() + bar.get_width()/2, gain,
                    yerr=[[gain - ci[0]], [ci[1] - gain]],
                    fmt='none', color='black', capsize=8, linewidth=2)
        ax.text(bar.get_x() + bar.get_width()/2, gain + 1.5,
                f'+{gain} pp', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Perfect Cue Gain (percentage points)')
    ax.set_title('Task 2: Conflicting Pushes — Cue Value Depends on Commitment')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 16)
    
    # Annotate key insight
    ax.annotate('First contact reveals\nmaterial in 95% of episodes',
                xy=(0.5, 8), xytext=(0.5, 13),
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'task2_sliding.png', dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'task2_sliding.png'}")


def fig_sim2real():
    """Figure: Sim-to-real transfer."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conditions = ['PGDA + correct\nlabel', 'PGDA + wrong\nlabel', 'PGDA + no\nlabel', 
                  'Baseline + correct\nlabel']
    heavy_rust = [0.967, 0.20, 0.033, 0.70]
    sandpaper = [0.967, 0.20, 0.033, 0.633]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax.bar(x - width/2, heavy_rust, width, label='Heavy Rust', color='#8c564b', edgecolor='black')
    ax.bar(x + width/2, sandpaper, width, label='Coarse Sandpaper', color='#7f7f7f', edgecolor='black')
    
    ax.set_ylabel('Success Rate')
    ax.set_title('Sim-to-Real Transfer on UR3e (30 episodes each)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sim2real_transfer.png', dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'sim2real_transfer.png'}")


def fig_encoder_cm():
    """Figure: Encoder confusion matrix (sim vs real)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Simulation confusion matrix
    cm_sim = np.array([
        [0.72, 0.14, 0.14],
        [0.18, 0.64, 0.18],
        [0.20, 0.38, 0.42],
    ])
    classes = ['Heavy Rust', 'Wet Ice', 'Coarse Sandpaper']
    
    im1 = ax1.imshow(cm_sim, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.set_yticklabels(classes)
    ax1.set_title('Synthetic Held-Out (68.3% acc)')
    ax1.set_ylabel('True')
    ax1.set_xlabel('Predicted')
    
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{cm_sim[i,j]:.0%}', ha='center', va='center',
                     color='white' if cm_sim[i,j] > 0.5 else 'black')
    
    # Real confusion matrix
    cm_real = np.array([
        [0.567, 0.133, 0.300],
        [0.233, 0.500, 0.267],
        [0.400, 0.267, 0.333],
    ])
    
    im2 = ax2.imshow(cm_real, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_yticklabels(classes)
    ax2.set_title('Real Objects (46.7% acc)')
    ax2.set_xlabel('Predicted')
    
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{cm_real[i,j]:.0%}', ha='center', va='center',
                     color='white' if cm_real[i,j] > 0.5 else 'black')
    
    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'encoder_confusion.png', dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'encoder_confusion.png'}")


def main():
    """Generate all figures."""
    print("Generating paper figures...")
    fig_training_curves()
    fig_mismatch_test()
    fig_ablation()
    fig_task2()
    fig_sim2real()
    fig_encoder_cm()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
