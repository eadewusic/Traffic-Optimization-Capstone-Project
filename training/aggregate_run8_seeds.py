#!/usr/bin/env python3
"""
Aggregate Results Across Multiple Seeds (Run 8)
This script aggregates and analyzes training results from multiple seeds
for Run 8 of the Traffic Optimization Capstone Project. It computes statistical
measures, generates visualizations, and creates a detailed text report.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats
from datetime import datetime


def find_project_root():
    """Find project root directory"""
    current = Path(__file__).parent
    
    # Look for characteristic directories
    while current != current.parent:
        if (current / 'results').exists() and (current / 'models').exists():
            return current
        current = current.parent
    
    # If not found, use parent of training directory
    return Path(__file__).parent.parent


def load_seed_results(results_run8_dir):
    """Load training summaries from all seed folders"""
    
    if not results_run8_dir.exists():
        print(f" Error: {results_run8_dir} does not exist")
        return None
    
    seed_folders = [f for f in results_run8_dir.iterdir() if f.is_dir() and f.name.startswith('seed_')]
    
    if not seed_folders:
        print(f" No seed folders found in {results_run8_dir}")
        return None
    
    results = []
    
    for seed_folder in sorted(seed_folders):
        summary_json = seed_folder / 'training_summary.json'
        
        if not summary_json.exists():
            print(f"  No training_summary.json in {seed_folder.name}, skipping...")
            continue
        
        with open(summary_json, 'r') as f:
            summary = json.load(f)
            results.append(summary)
            seed = summary['seed']
            final_reward = summary['training_statistics']['final_reward']
            print(f" Loaded: {seed_folder.name} (seed {seed}, final reward: {final_reward:.1f})")
    
    return results


def compute_statistics(results):
    """Compute aggregate statistics across all seeds"""
    
    seeds = [r['seed'] for r in results]
    final_rewards = [r['training_statistics']['final_reward'] for r in results]
    best_rewards = [r['training_statistics']['best_reward'] for r in results]
    improvements = [r['training_statistics']['improvement'] for r in results]
    
    stats_dict = {
        'n_seeds': len(results),
        'seeds': seeds,
        'final_reward_mean': np.mean(final_rewards),
        'final_reward_std': np.std(final_rewards),
        'final_reward_min': np.min(final_rewards),
        'final_reward_max': np.max(final_rewards),
        'best_reward_mean': np.mean(best_rewards),
        'best_reward_std': np.std(best_rewards),
        'improvement_mean': np.mean(improvements),
        'improvement_std': np.std(improvements),
        'all_final_rewards': final_rewards,
        'all_best_rewards': best_rewards,
        'all_improvements': improvements
    }
    
    return stats_dict


def create_aggregate_visualization(results, stats, output_dir):
    """Create comprehensive multi-seed comparison visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Run 8: Multi-Seed Training Analysis', fontsize=16, weight='bold')
    
    seeds = stats['seeds']
    colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))
    
    # Plot 1: Final Rewards Comparison
    ax1 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(seeds))
    bars = ax1.bar(x_pos, stats['all_final_rewards'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=stats['final_reward_mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {stats['final_reward_mean']:.1f}")
    ax1.fill_between(x_pos, 
                     stats['final_reward_mean'] - stats['final_reward_std'],
                     stats['final_reward_mean'] + stats['final_reward_std'],
                     alpha=0.2, color='red', label=f"±1 Std: {stats['final_reward_std']:.1f}")
    
    ax1.set_xlabel('Seed', fontsize=12)
    ax1.set_ylabel('Final Reward', fontsize=12)
    ax1.set_title('Final Reward Across Seeds', fontsize=14, weight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, stats['all_final_rewards'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Best Rewards
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(x_pos, stats['all_best_rewards'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=stats['best_reward_mean'], color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('Best Reward')
    ax2.set_title('Best Reward During Training')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{s}" for s in seeds], rotation=45, fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Improvement
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(x_pos, stats['all_improvements'], color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=stats['improvement_mean'], color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Seed')
    ax3.set_ylabel('Improvement')
    ax3.set_title('Learning Improvement')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{s}" for s in seeds], rotation=45, fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Distribution Box Plot
    ax4 = fig.add_subplot(gs[1, 2])
    data_to_plot = [stats['all_final_rewards'], stats['all_best_rewards']]
    bp = ax4.boxplot(data_to_plot, labels=['Final', 'Best'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax4.set_ylabel('Reward')
    ax4.set_title('Reward Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Statistics Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    table_data = [
        ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
        ['Final Reward', 
         f"{stats['final_reward_mean']:.1f}", 
         f"±{stats['final_reward_std']:.1f}",
         f"{stats['final_reward_min']:.1f}",
         f"{stats['final_reward_max']:.1f}"],
        ['Best Reward',
         f"{stats['best_reward_mean']:.1f}",
         f"±{stats['best_reward_std']:.1f}",
         f"{np.min(stats['all_best_rewards']):.1f}",
         f"{np.max(stats['all_best_rewards']):.1f}"],
        ['Improvement',
         f"{stats['improvement_mean']:.1f}",
         f"±{stats['improvement_std']:.1f}",
         f"{np.min(stats['all_improvements']):.1f}",
         f"{np.max(stats['all_improvements']):.1f}"]
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, 4):
        for j in range(5):
            if j == 0:
                table[(i, j)].set_facecolor('#E8F5E9')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    ax5.set_title('Summary Statistics Across All Seeds', 
                 fontsize=14, weight='bold', pad=20)
    
    plot_path = output_dir / 'multiseed_analysis.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n Visualization saved: {plot_path}")
    return plot_path


def create_text_report(results, stats, output_dir):
    """Create detailed text report"""
    
    report_path = output_dir / 'multiseed_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RUN 8: MULTI-SEED TRAINING - STATISTICAL ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Seeds: {stats['n_seeds']} ({', '.join(map(str, stats['seeds']))})\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("AGGREGATE RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"Final Reward: {stats['final_reward_mean']:.2f} ± {stats['final_reward_std']:.2f}\n")
        f.write(f"Best Reward:  {stats['best_reward_mean']:.2f} ± {stats['best_reward_std']:.2f}\n")
        f.write(f"Improvement:  {stats['improvement_mean']:.2f} ± {stats['improvement_std']:.2f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("INDIVIDUAL SEEDS\n")
        f.write("-"*70 + "\n\n")
        
        sorted_results = sorted(results, key=lambda x: x['training_statistics']['final_reward'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            seed = result['seed']
            final = result['training_statistics']['final_reward']
            f.write(f"{i}. Seed {seed}: {final:.2f}")
            if i == 1:
                f.write(" BEST")
            f.write("\n")
        
        f.write("\n")
        
        if stats['n_seeds'] >= 3:
            t_stat, p_value = scipy_stats.ttest_1samp(stats['all_final_rewards'], 0)
            f.write("-"*70 + "\n")
            f.write("STATISTICAL SIGNIFICANCE\n")
            f.write("-"*70 + "\n\n")
            f.write(f"t-statistic: {t_stat:.3f}\n")
            f.write(f"p-value: {p_value:.6f}\n")
            f.write(f"Result: {'Significant (p < 0.05)' if p_value < 0.05 else 'Not significant'}\n\n")
        
        best_seed = sorted_results[0]['seed']
        f.write("-"*70 + "\n")
        f.write("DEPLOYMENT\n")
        f.write("-"*70 + "\n\n")
        f.write(f"Best Model: Seed {best_seed}\n")
        f.write(f"Location: models/hardware_ppo/run_8/seed_{best_seed}/best_model.zip\n\n")
        
        f.write("="*70 + "\n")
    
    print(f" Text report saved: {report_path}")
    return report_path


def main():
    print("="*70)
    print(" RUN 8: AGGREGATE MULTI-SEED RESULTS")
    print("="*70)
    
    project_root = find_project_root()
    results_run8_dir = project_root / 'results' / 'run_8'
    viz_run8_dir = project_root / 'visualizations' / 'run_8'
    
    print(f"\n Project: {project_root}")
    print(f" Scanning: {results_run8_dir}\n")
    
    if not results_run8_dir.exists():
        print(f" Error: {results_run8_dir} not found!")
        return
    
    results = load_seed_results(results_run8_dir)
    
    if not results or len(results) < 2:
        print("\n Need at least 2 seed results!")
        return
    
    print(f"\n Found {len(results)} seeds\n")
    
    stats = compute_statistics(results)
    
    results_aggregate = results_run8_dir / 'aggregate_analysis'
    viz_aggregate = viz_run8_dir / 'aggregate_analysis'
    
    results_aggregate.mkdir(parents=True, exist_ok=True)
    viz_aggregate.mkdir(parents=True, exist_ok=True)
    
    create_aggregate_visualization(results, stats, viz_aggregate)
    create_text_report(results, stats, results_aggregate)
    
    print("\n" + "="*70)
    print(" AGGREGATION COMPLETE")
    print("="*70)
    print(f"\n Final Reward: {stats['final_reward_mean']:.2f} ± {stats['final_reward_std']:.2f}")
    
    best_seed = sorted(results, key=lambda x: x['training_statistics']['final_reward'], reverse=True)[0]['seed']
    print(f" Best Seed: {best_seed}")
    print(f"\n Results: {results_aggregate}")


if __name__ == "__main__":
    main()
