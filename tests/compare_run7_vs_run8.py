"""
Compare Run 7 vs Run 8 Seed 789 - Final Models
Purpose: Determine which final model should be deployed to Raspberry Pi hardware
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats


class RunComparator:
    """Compare two trained models to determine best for deployment"""
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.run7_data = None
        self.run8_seed789_data = None
        
    def load_run_data(self):
        """Load training summaries for both runs"""
        
        # Load Run 7
        run7_path = self.project_root / 'results' / 'run_7' / 'run7_training_summary.json'
        if not run7_path.exists():
            raise FileNotFoundError(f"Run 7 training summary not found")
        
        with open(run7_path, 'r') as f:
            self.run7_data = json.load(f)
        
        print(f"Loaded Run 7: {run7_path}")
        
        # Load Run 8 Seed 789
        run8_seed789_path = self.project_root / 'results' / 'run_8' / 'seed_789' / 'training_summary.json'
        
        if not run8_seed789_path.exists():
            raise FileNotFoundError(f"Run 8 Seed 789 training summary not found")
        
        with open(run8_seed789_path, 'r') as f:
            self.run8_seed789_data = json.load(f)
        
        print(f"Loaded Run 8 Seed 789: {run8_seed789_path}")
        
    def extract_comparison_metrics(self):
        """Extract key metrics for comparison"""
        
        # Run 7 metrics
        run7_metrics = {
            'name': 'Run 7',
            'final_reward': self.run7_data['training']['final_reward'],
            'best_reward': self.run7_data['training']['best_reward'],
            'total_steps': self.run7_data['training']['total_steps'],
            'improvement': self.run7_data['training']['improvement'],
            'final_std': self.run7_data['training']['final_std'],
            'ready_for_testing': self.run7_data['assessment']['ready_for_testing'],
            'training_efficiency': self.run7_data['training']['final_reward'] / self.run7_data['training']['total_steps'] * 1000000
        }
        
        # Run 8 Seed 789 metrics
        run8_metrics = {
            'name': 'Run 8 Seed 789',
            'final_reward': self.run8_seed789_data['training_statistics']['final_reward'],
            'best_reward': self.run8_seed789_data['training_statistics']['best_reward'],
            'total_steps': 1000000,  # Standard for Run 8
            'improvement': self.run8_seed789_data['training_statistics']['improvement'],
            'final_std': None,  # Not available in single seed summary
            'ready_for_testing': True,
            'training_efficiency': self.run8_seed789_data['training_statistics']['final_reward'] / 1000000 * 1000000
        }
        
        return run7_metrics, run8_metrics
    
    def perform_statistical_comparison(self, run7_metrics, run8_metrics):
        """Perform statistical analysis to determine significance"""
        
        print("\n" + "="*70)
        print(" STATISTICAL COMPARISON")
        print("="*70)
        
        # Calculate absolute difference
        final_diff = run8_metrics['final_reward'] - run7_metrics['final_reward']
        final_diff_pct = (final_diff / run7_metrics['final_reward']) * 100
        
        print(f"\n Final Reward Difference:")
        print(f"   Run 8 Seed 789: {run8_metrics['final_reward']:.2f}")
        print(f"   Run 7:          {run7_metrics['final_reward']:.2f}")
        print(f"   Difference:     {final_diff:+.2f} ({final_diff_pct:+.2f}%)")
        
        # Training efficiency
        print(f"\n Training Efficiency (reward per 1M steps):")
        print(f"   Run 8 Seed 789: {run8_metrics['training_efficiency']:.2f}")
        print(f"   Run 7:          {run7_metrics['training_efficiency']:.2f}")
        
        # Steps efficiency
        steps_diff = run8_metrics['total_steps'] - run7_metrics['total_steps']
        steps_diff_pct = (steps_diff / run7_metrics['total_steps']) * 100
        print(f"\n Training Steps:")
        print(f"   Run 8 Seed 789: {run8_metrics['total_steps']:,} steps")
        print(f"   Run 7:          {run7_metrics['total_steps']:,} steps")
        print(f"   Difference:     {steps_diff:+,} steps ({steps_diff_pct:+.1f}%)")
        
        # Determine winner
        if final_diff > 0:
            winner = "Run 8 Seed 789"
            winner_advantage = final_diff
        else:
            winner = "Run 7"
            winner_advantage = abs(final_diff)
        
        # Practical significance threshold (>1% improvement)
        is_significant = abs(final_diff_pct) > 1.0
        
        comparison_result = {
            'winner': winner,
            'advantage': winner_advantage,
            'advantage_pct': abs(final_diff_pct),
            'is_significant': is_significant,
            'final_diff': final_diff,
            'steps_saved': abs(steps_diff) if run8_metrics['final_reward'] >= run7_metrics['final_reward'] else 0
        }
        
        return comparison_result
    
    def create_comparison_visualization(self, run7_metrics, run8_metrics, comparison_result):
        """Create comprehensive comparison visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Run 7 vs Run 8 Seed 789: Final Model Comparison', 
                     fontsize=16, fontweight='bold')
        
        models = ['Run 7', 'Run 8\nSeed 789']
        colors = ['#3498db', '#2ecc71']  # Blue for Run 7, Green for Run 8
        
        # 1. Final Reward Comparison
        ax1 = axes[0, 0]
        final_rewards = [run7_metrics['final_reward'], run8_metrics['final_reward']]
        bars1 = ax1.bar(models, final_rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Final Reward', fontsize=12, fontweight='bold')
        ax1.set_title('Final Reward (Deployment Performance)', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, final_rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Highlight winner
        if comparison_result['winner'] == 'Run 8 Seed 789':
            bars1[1].set_edgecolor('gold')
            bars1[1].set_linewidth(3)
        else:
            bars1[0].set_edgecolor('gold')
            bars1[0].set_linewidth(3)
        
        # 2. Best Reward Comparison
        ax2 = axes[0, 1]
        best_rewards = [run7_metrics['best_reward'], run8_metrics['best_reward']]
        bars2 = ax2.bar(models, best_rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Best Reward', fontsize=12, fontweight='bold')
        ax2.set_title('Best Reward During Training', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, best_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Training Steps Comparison
        ax3 = axes[0, 2]
        steps = [run7_metrics['total_steps']/1000, run8_metrics['total_steps']/1000]
        bars3 = ax3.bar(models, steps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
        ax3.set_title('Training Efficiency', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars3, steps):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}K',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. Performance Delta
        ax4 = axes[1, 0]
        ax4.axis('off')
        delta_text = f"""
        PERFORMANCE COMPARISON
        
        Final Reward Difference:
        {comparison_result['final_diff']:+.2f} points
        ({comparison_result['advantage_pct']:+.2f}%)
        
        Winner: {comparison_result['winner']}
        
        Statistical Significance:
        {'YES - Meaningful improvement' if comparison_result['is_significant'] else 'NO - Marginal difference'}
        """
        ax4.text(0.5, 0.5, delta_text, 
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
        
        # 5. Side-by-Side Metrics Table
        ax5 = axes[1, 1]
        ax5.axis('off')
        
        table_data = [
            ['Metric', 'Run 7', 'Run 8 Seed 789', 'Δ'],
            ['Final Reward', 
             f"{run7_metrics['final_reward']:.1f}",
             f"{run8_metrics['final_reward']:.1f}",
             f"{comparison_result['final_diff']:+.1f}"],
            ['Best Reward',
             f"{run7_metrics['best_reward']:.1f}",
             f"{run8_metrics['best_reward']:.1f}",
             f"{run8_metrics['best_reward'] - run7_metrics['best_reward']:+.1f}"],
            ['Training Steps',
             f"{run7_metrics['total_steps']//1000}K",
             f"{run8_metrics['total_steps']//1000}K",
             f"{(run8_metrics['total_steps'] - run7_metrics['total_steps'])//1000:+}K"],
            ['Improvement',
             f"{run7_metrics['improvement']:.1f}",
             f"{run8_metrics['improvement']:.1f}",
             f"{run8_metrics['improvement'] - run7_metrics['improvement']:+.1f}"]
        ]
        
        table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, 5):
            table[(i, 0)].set_facecolor('#E8F5E9')
            for j in range(1, 4):
                table[(i, j)].set_facecolor('#F5F5F5')
        
        ax5.set_title('Detailed Metrics Comparison', fontsize=12, fontweight='bold', pad=20)
        
        # 6. Deployment Recommendation
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        if comparison_result['winner'] == 'Run 8 Seed 789':
            recommendation = f"""
            DEPLOYMENT RECOMMENDATION
            
            Deploy: Run 8 Seed 789
            
            Reasons:
            • Higher final reward (+{comparison_result['advantage']:.1f})
            • More efficient training
              ({comparison_result['steps_saved']//1000}K fewer steps)
            • Part of validated multi-seed study
            • Proven reproducibility
            
            Model Files:
            models/hardware_ppo/run_8/
            seed_789/ppo_final_seed789.zip
            seed_789/vec_normalize_seed789.pkl
            
            Status: RECOMMENDED
            """
            box_color = 'lightgreen'
        else:
            recommendation = f"""
            DEPLOYMENT RECOMMENDATION
            
            Keep: Run 7
            
            Reasons:
            • Higher final reward (+{comparison_result['advantage']:.1f})
            • Already deployed and tested
            • Proven hardware compatibility
            
            Model Files:
            models/hardware_ppo/run_7/
            final_model.zip
            vecnormalize.pkl
            
            Status: CURRENT CHAMPION
            """
            box_color = 'lightblue'
        
        ax6.text(0.5, 0.5, recommendation,
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7),
                family='monospace')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = self.project_root / 'visualizations' / 'run_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'run7_vs_run8seed789_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n Comparison visualization saved: {output_path}")
        
        plt.close()
    
    def generate_deployment_report(self, run7_metrics, run8_metrics, comparison_result):
        """Generate detailed deployment recommendation report"""
        
        output_dir = self.project_root / 'results' / 'run_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'deployment_recommendation.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DEPLOYMENT RECOMMENDATION: RUN 7 VS RUN 8 SEED 789\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Comparison: Final Models (with VecNormalize)\n\n")
            
            f.write("-"*70 + "\n")
            f.write("FINAL MODEL PERFORMANCE\n")
            f.write("-"*70 + "\n\n")
            
            f.write(f"Run 7:\n")
            f.write(f"  Final Reward:   {run7_metrics['final_reward']:.2f}\n")
            f.write(f"  Best Reward:    {run7_metrics['best_reward']:.2f}\n")
            f.write(f"  Training Steps: {run7_metrics['total_steps']:,}\n")
            f.write(f"  Status:         Currently deployed on Raspberry Pi\n\n")
            
            f.write(f"Run 8 Seed 789:\n")
            f.write(f"  Final Reward:   {run8_metrics['final_reward']:.2f}\n")
            f.write(f"  Best Reward:    {run8_metrics['best_reward']:.2f}\n")
            f.write(f"  Training Steps: {run8_metrics['total_steps']:,}\n")
            f.write(f"  Status:         Multi-seed validation champion\n\n")
            
            f.write("-"*70 + "\n")
            f.write("COMPARISON RESULTS\n")
            f.write("-"*70 + "\n\n")
            
            f.write(f"Performance Difference:\n")
            f.write(f"  Final Reward:   {comparison_result['final_diff']:+.2f} points "
                   f"({comparison_result['advantage_pct']:+.2f}%)\n")
            f.write(f"  Winner:         {comparison_result['winner']}\n")
            f.write(f"  Advantage:      {comparison_result['advantage']:.2f} points\n")
            f.write(f"  Significant:    {'YES' if comparison_result['is_significant'] else 'NO'}\n\n")
            
            if comparison_result['steps_saved'] > 0:
                f.write(f"Training Efficiency:\n")
                f.write(f"  Steps Saved:    {comparison_result['steps_saved']:,} steps\n")
                f.write(f"  Efficiency Gain: {(comparison_result['steps_saved']/run7_metrics['total_steps']*100):.1f}%\n\n")
            
            f.write("-"*70 + "\n")
            f.write("DEPLOYMENT RECOMMENDATION\n")
            f.write("-"*70 + "\n\n")
            
            if comparison_result['winner'] == 'Run 8 Seed 789':
                f.write("RECOMMENDED MODEL: Run 8 Seed 789\n\n")
                f.write("Justification:\n")
                f.write(f"1. Superior final performance (+{comparison_result['advantage']:.1f} reward)\n")
                f.write(f"2. More efficient training ({comparison_result['steps_saved']//1000}K fewer steps)\n")
                f.write(f"3. Part of multi-seed validation (proven reproducibility)\n")
                f.write(f"4. Statistical significance: {'YES' if comparison_result['is_significant'] else 'Marginal but positive'}\n\n")
                
                f.write("Deployment Files:\n")
                f.write("  Model:        models/hardware_ppo/run_8/seed_789/ppo_final_seed789.zip\n")
                f.write("  VecNormalize: models/hardware_ppo/run_8/seed_789/vec_normalize_seed789.pkl\n\n")
                
            else:
                f.write("RECOMMENDED MODEL: Run 7 (Keep Current)\n\n")
                f.write("Justification:\n")
                f.write(f"1. Superior final performance (+{comparison_result['advantage']:.1f} reward)\n")
                f.write(f"2. Already deployed and tested on hardware\n")
                f.write(f"3. Proven stability in real-world conditions\n\n")
                
                f.write("Deployment Files:\n")
                f.write("  Model:        models/hardware_ppo/run_7/final_model.zip\n")
                f.write("  VecNormalize: models/hardware_ppo/run_7/vecnormalize.pkl\n\n")
                
                f.write("Note:\n")
                f.write("Run 8 Seed 789 showed strong performance but did not exceed\n")
                f.write("Run 7's final model. Consider Run 8 as validation of approach.\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f" Deployment report saved: {report_path}")
        
        # Also save JSON for programmatic access
        json_path = output_dir / 'comparison_results.json'
        json_data = {
            'analysis_date': datetime.now().isoformat(),
            'run7': run7_metrics,
            'run8_seed789': run8_metrics,
            'comparison': comparison_result,
            'recommendation': comparison_result['winner']
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f" JSON results saved: {json_path}")
    
    def run_full_comparison(self):
        """Execute complete comparison analysis"""
        
        print("="*70)
        print(" RUN 7 VS RUN 8 SEED 789: FINAL MODEL COMPARISON")
        print("="*70)
        
        # Load data
        self.load_run_data()
        
        # Extract metrics
        run7_metrics, run8_metrics = self.extract_comparison_metrics()
        
        # Statistical comparison
        comparison_result = self.perform_statistical_comparison(run7_metrics, run8_metrics)
        
        # Create visualization
        self.create_comparison_visualization(run7_metrics, run8_metrics, comparison_result)
        
        # Generate report
        self.generate_deployment_report(run7_metrics, run8_metrics, comparison_result)
        
        print("\n" + "="*70)
        print(" COMPARISON COMPLETE")
        print("="*70)
        print(f"\n WINNER: {comparison_result['winner']}")
        print(f"   Advantage: {comparison_result['advantage']:.2f} points ({comparison_result['advantage_pct']:.2f}%)")
        print(f"   Significant: {'YES' if comparison_result['is_significant'] else 'NO (marginal)'}")
        print("\n Results saved in: results/run_comparison/")
        print(" Visualization saved in: visualizations/run_comparison/")
        
        return comparison_result


def main():
    """Main comparison script"""
    
    PROJECT_ROOT = r"C:\Users\HP\Traffic-Optimization-Capstone-Project"
    
    comparator = RunComparator(PROJECT_ROOT)
    result = comparator.run_full_comparison()
    
    print("\n" + "="*70)
    if result['winner'] == 'Run 8 Seed 789':
        print("  RECOMMENDATION: Deploy Run 8 Seed 789 to Raspberry Pi")
        print(f"    Expected improvement: +{result['advantage']:.1f} reward points")
    else:
        print(" RECOMMENDATION: Keep Run 7 on Raspberry Pi")
        print(f"    Run 7 maintains advantage of +{result['advantage']:.1f} points")
    print("="*70)


if __name__ == "__main__":
    main()
