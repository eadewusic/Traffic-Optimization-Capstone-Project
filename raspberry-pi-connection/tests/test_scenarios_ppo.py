"""
PPO Agent Testing Scenarios - Run 8 Seed 789
Comprehensive scenario testing to demonstrate decision-making logic

This script tests the PPO agent across diverse traffic scenarios to answer:
- How does the agent prioritize when queues are imbalanced? (e.g., W=3, S=10)
- Does it handle balanced traffic efficiently?
- Can it recover from extreme congestion in a single lane?
- How does it respond to random, stochastic arrivals?
- What's the agent's logic for deciding phase changes?

Each scenario:
1. Sets up a specific traffic pattern
2. Runs the PPO agent
3. Logs all decisions with explanations
4. Compares against baseline fixed-timing
5. Generates detailed analysis report

This testing is useful for:
- Thesis defense demos
- Understanding agent behavior
- Validating adaptive control
- Comparing different traffic patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os
from datetime import datetime
import json

# Add path for environments
sys.path.append('/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/environments')
from run7_env import Run7TrafficEnv


# Testing Scenarios
class ScenarioTester:
    """
    Comprehensive scenario testing for PPO agent
    
    Tests agent across predefined scenarios and analyzes decision-making logic
    """
    
    def __init__(self, model_path, vecnorm_path, output_dir='results/scenario_tests'):
        """
        Initialize scenario tester
        
        Args:
            model_path: Path to PPO model
            vecnorm_path: Path to VecNormalize
            output_dir: Directory for test results
        """
        print(" PPO AGENT SCENARIO TESTING")
        print(" Run 8 Seed 789 - Multi-Seed Champion")
        print()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"test_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        print(f"[LOADING MODEL]")
        print(f"  Model: {model_path}")
        print(f"  VecNormalize: {vecnorm_path}")
        
        self.model = PPO.load(model_path)
        dummy_env = DummyVecEnv([lambda: Run7TrafficEnv()])
        self.vec_env = VecNormalize.load(vecnorm_path, dummy_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        
        print("  Model loaded\n")
        
        self.max_queue = 20
        self.results = []
    
    def predict_action(self, queues, explain=False):
        """
        Get PPO action for given queue state
        
        Args:
            queues: Array [N, S, E, W]
            explain: If True, print explanation of decision
        
        Returns:
            Tuple of (action, explanation_text)
        """
        # Normalize observation
        obs = np.array(queues, dtype=np.float32) / self.max_queue
        obs_norm = self.vec_env.normalize_obs(obs)
        
        # Get action
        action, _ = self.model.predict(obs_norm, deterministic=True)
        action = int(action)
        
        # Generate explanation
        n, s, e, w = queues
        ns_demand = n + s
        ew_demand = e + w
        
        explanation = []
        explanation.append(f"State: [N={n}, S={s}, E={e}, W={w}]")
        explanation.append(f"N/S demand: {ns_demand} cars")
        explanation.append(f"E/W demand: {ew_demand} cars")
        
        if action == 0:
            explanation.append("Decision: North/South GREEN")
            if ns_demand > ew_demand:
                explanation.append(f"Reason: N/S has higher demand ({ns_demand} > {ew_demand})")
            elif ns_demand == ew_demand:
                explanation.append("Reason: Equal demand, maintaining current phase")
            else:
                explanation.append(f"Reason: Agent prioritizing N/S despite lower demand")
                explanation.append("       (possibly due to phase switching cost)")
        else:
            explanation.append("Decision: East/West GREEN")
            if ew_demand > ns_demand:
                explanation.append(f"Reason: E/W has higher demand ({ew_demand} > {ns_demand})")
            elif ew_demand == ns_demand:
                explanation.append("Reason: Equal demand, agent chose E/W")
            else:
                explanation.append(f"Reason: Agent prioritizing E/W despite lower demand")
                explanation.append("       (possibly due to phase switching cost)")
        
        if explain:
            for line in explanation:
                print(f"  {line}")
            print()
        
        return action, "\n".join(explanation)
    
    def run_scenario(self, name, initial_queues, description, max_steps=30):
        """
        Run a single test scenario
        
        Args:
            name: Scenario name
            initial_queues: Initial queue state [N, S, E, W]
            description: Scenario description
            max_steps: Maximum steps to run
        
        Returns:
            Dictionary with scenario results
        """
        print(f"\n SCENARIO: {name}")
        print(f"Description: {description}")
        print(f"Initial state: N={initial_queues[0]}, S={initial_queues[1]}, "
              f"E={initial_queues[2]}, W={initial_queues[3]}")
        print()
        
        queues = np.array(initial_queues, dtype=np.float32)
        step = 0
        decisions = []
        current_phase = 0
        phase_changes = 0
        total_cleared = 0
        
        print("Step-by-step decisions:")
        print("-" * 70)
        
        while np.sum(queues) > 0 and step < max_steps:
            step += 1
            
            # Get PPO decision
            action, explanation = self.predict_action(queues, explain=False)
            
            # Check phase change
            phase_change = (action != current_phase)
            if phase_change:
                phase_changes += 1
                current_phase = action
            
            # Clear vehicles
            cleared = 0
            if action == 0:  # N/S green
                for lane in [0, 1]:
                    if queues[lane] > 0:
                        clear_amount = min(2, queues[lane])  # Clear up to 2 cars
                        queues[lane] -= clear_amount
                        cleared += clear_amount
            else:  # E/W green
                for lane in [2, 3]:
                    if queues[lane] > 0:
                        clear_amount = min(2, queues[lane])
                        queues[lane] -= clear_amount
                        cleared += clear_amount
            
            total_cleared += cleared
            
            # Log decision
            decisions.append({
                'step': step,
                'queues_before': queues.copy() + cleared,  # Queues before clearing
                'action': action,
                'phase': 'N/S' if action == 0 else 'E/W',
                'phase_change': phase_change,
                'cleared': cleared,
                'queues_after': queues.copy(),
                'explanation': explanation
            })
            
            # Print step summary
            phase_marker = "→ SWITCH" if phase_change else ""
            print(f"[Step {step:2d}] {['N/S', 'E/W'][action]} GREEN {phase_marker:10s} | "
                  f"Cleared: {int(cleared):2d} | "
                  f"Remaining: [N={int(queues[0])} S={int(queues[1])} E={int(queues[2])} W={int(queues[3])}]")
        
        print("-" * 70)
        print(f"Scenario complete in {step} steps")
        print(f"Total cleared: {int(total_cleared)} cars")
        print(f"Phase changes: {phase_changes}")
        print(f"Final queues: N={int(queues[0])}, S={int(queues[1])}, "
              f"E={int(queues[2])}, W={int(queues[3])}")
        
        # Compute metrics
        result = {
            'name': name,
            'description': description,
            'initial_queues': initial_queues,
            'final_queues': queues.tolist(),
            'steps': step,
            'total_cleared': int(total_cleared),
            'phase_changes': phase_changes,
            'efficiency': total_cleared / max(phase_changes, 1),  # Cars per phase change
            'decisions': decisions
        }
        
        self.results.append(result)
        return result
    
    def compare_with_baseline(self, initial_queues, max_steps=30):
        """
        Compare PPO agent with fixed-timing baseline
        
        Args:
            initial_queues: Initial queue state
            max_steps: Maximum steps
        
        Returns:
            Comparison metrics
        """
        # PPO agent
        queues_ppo = np.array(initial_queues, dtype=np.float32)
        steps_ppo = 0
        cleared_ppo = 0
        current_phase = 0
        phases_ppo = 0
        
        while np.sum(queues_ppo) > 0 and steps_ppo < max_steps:
            steps_ppo += 1
            
            # PPO decision
            action, _ = self.predict_action(queues_ppo, explain=False)
            
            if action != current_phase:
                phases_ppo += 1
                current_phase = action
            
            # Clear vehicles
            if action == 0:
                for lane in [0, 1]:
                    if queues_ppo[lane] > 0:
                        clear_amount = min(2, queues_ppo[lane])
                        queues_ppo[lane] -= clear_amount
                        cleared_ppo += clear_amount
            else:
                for lane in [2, 3]:
                    if queues_ppo[lane] > 0:
                        clear_amount = min(2, queues_ppo[lane])
                        queues_ppo[lane] -= clear_amount
                        cleared_ppo += clear_amount
        
        # Fixed-timing baseline (10-step cycles)
        queues_fixed = np.array(initial_queues, dtype=np.float32)
        steps_fixed = 0
        cleared_fixed = 0
        current_phase = 0
        phases_fixed = 0
        steps_in_phase = 0
        
        while np.sum(queues_fixed) > 0 and steps_fixed < max_steps:
            steps_fixed += 1
            steps_in_phase += 1
            
            # Switch every 10 steps
            if steps_in_phase > 10:
                current_phase = 1 - current_phase
                phases_fixed += 1
                steps_in_phase = 0
            
            # Clear vehicles
            if current_phase == 0:
                for lane in [0, 1]:
                    if queues_fixed[lane] > 0:
                        clear_amount = min(2, queues_fixed[lane])
                        queues_fixed[lane] -= clear_amount
                        cleared_fixed += clear_amount
            else:
                for lane in [2, 3]:
                    if queues_fixed[lane] > 0:
                        clear_amount = min(2, queues_fixed[lane])
                        queues_fixed[lane] -= clear_amount
                        cleared_fixed += clear_amount
        
        return {
            'ppo': {
                'steps': steps_ppo,
                'cleared': int(cleared_ppo),
                'phase_changes': phases_ppo,
                'efficiency': cleared_ppo / max(phases_ppo, 1),
                'final_queue': int(np.sum(queues_ppo))
            },
            'fixed_timing': {
                'steps': steps_fixed,
                'cleared': int(cleared_fixed),
                'phase_changes': phases_fixed,
                'efficiency': cleared_fixed / max(phases_fixed, 1),
                'final_queue': int(np.sum(queues_fixed))
            }
        }
    
    def generate_report(self):
        """
        Generate comprehensive test report
        """
        report_path = os.path.join(self.output_dir, "scenario_test_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("\n PPO AGENT SCENARIO TESTING REPORT\n")
            f.write(" Run 8 Seed 789 - Multi-Seed Champion\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Scenarios: {len(self.results)}\n\n")
            
            # Summary table
            f.write("\n SCENARIO SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"{'Scenario':<30} {'Steps':<8} {'Cleared':<10} {'Phases':<8} {'Efficiency':<12}\n")
            f.write("-" * 70 + "\n")
            
            for result in self.results:
                f.write(f"{result['name']:<30} {result['steps']:<8} "
                       f"{result['total_cleared']:<10} {result['phase_changes']:<8} "
                       f"{result['efficiency']:.2f}\n")
            
            f.write("\n")
            
            # Detailed results
            for i, result in enumerate(self.results, 1):
                f.write("\n" + "="*70 + "\n")
                f.write(f" SCENARIO {i}: {result['name']}\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Description: {result['description']}\n\n")
                
                f.write("Initial State:\n")
                f.write(f"  North: {result['initial_queues'][0]} cars\n")
                f.write(f"  South: {result['initial_queues'][1]} cars\n")
                f.write(f"  East: {result['initial_queues'][2]} cars\n")
                f.write(f"  West: {result['initial_queues'][3]} cars\n")
                f.write(f"  Total: {sum(result['initial_queues'])} cars\n\n")
                
                f.write("Results:\n")
                f.write(f"  Steps to clear: {result['steps']}\n")
                f.write(f"  Total cleared: {result['total_cleared']}\n")
                f.write(f"  Phase changes: {result['phase_changes']}\n")
                f.write(f"  Efficiency: {result['efficiency']:.2f} cars/phase\n\n")
                
                f.write("Final State:\n")
                f.write(f"  North: {int(result['final_queues'][0])} cars\n")
                f.write(f"  South: {int(result['final_queues'][1])} cars\n")
                f.write(f"  East: {int(result['final_queues'][2])} cars\n")
                f.write(f"  West: {int(result['final_queues'][3])} cars\n")
                f.write(f"  Remaining: {int(sum(result['final_queues']))} cars\n\n")
                
                f.write("Decision Log:\n")
                f.write("-" * 70 + "\n")
                for decision in result['decisions'][:10]:  # First 10 steps
                    f.write(f"\nStep {decision['step']}:\n")
                    f.write(f"  {decision['explanation']}\n")
                    f.write(f"  Cleared: {int(decision['cleared'])} cars\n")
                    if decision['phase_change']:
                        f.write(f"  → Phase change occurred\n")
                
                if len(result['decisions']) > 10:
                    f.write(f"\n... ({len(result['decisions']) - 10} more steps)\n")
        
        print(f"\n[SAVED] Report: {report_path}")
        return report_path
    
    def visualize_results(self):
        """
        Create visualization of all scenarios
        """
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PPO Agent Scenario Testing Results', fontsize=14, fontweight='bold')
        
        # Plot 1: Steps to clear
        names = [r['name'][:20] for r in self.results]
        steps = [r['steps'] for r in self.results]
        axes[0, 0].barh(names, steps, color='steelblue')
        axes[0, 0].set_xlabel('Steps to Clear')
        axes[0, 0].set_title('Completion Time by Scenario')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Efficiency (cars/phase)
        efficiency = [r['efficiency'] for r in self.results]
        axes[0, 1].barh(names, efficiency, color='green')
        axes[0, 1].set_xlabel('Cars Cleared per Phase Change')
        axes[0, 1].set_title('Control Efficiency')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Phase changes
        phases = [r['phase_changes'] for r in self.results]
        axes[1, 0].barh(names, phases, color='orange')
        axes[1, 0].set_xlabel('Phase Changes')
        axes[1, 0].set_title('Adaptability (Fewer = More Stable)')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Initial vs Final queue
        initial_totals = [sum(r['initial_queues']) for r in self.results]
        final_totals = [sum(r['final_queues']) for r in self.results]
        
        x = np.arange(len(names))
        width = 0.35
        axes[1, 1].bar(x - width/2, initial_totals, width, label='Initial', color='red', alpha=0.7)
        axes[1, 1].bar(x + width/2, final_totals, width, label='Final', color='blue', alpha=0.7)
        axes[1, 1].set_ylabel('Total Cars')
        axes[1, 1].set_title('Queue Reduction')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        viz_path = os.path.join(self.output_dir, "scenario_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVED] Visualization: {viz_path}")


# Predefined Scenarios
def run_all_scenarios():
    """
    Run all predefined test scenarios
    """
    MODEL_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/ppo_final_seed789.zip"
    VECNORM_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/vec_normalize_seed789.pkl"
    
    # Check files exist
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return
    if not os.path.exists(VECNORM_PATH):
        print(f"[ERROR] VecNormalize not found: {VECNORM_PATH}")
        return
    
    # Initialize tester
    tester = ScenarioTester(MODEL_PATH, VECNORM_PATH)
    
    # Scenario 1: Imbalanced queues (supervisor's example)
    tester.run_scenario(
        name="Imbalanced: W=3, S=10",
        initial_queues=[0, 10, 0, 3],
        description="South has 10 cars, West has 3. Does agent prioritize high demand?"
    )
    
    # Scenario 2: Balanced traffic
    tester.run_scenario(
        name="Balanced: All=5",
        initial_queues=[5, 5, 5, 5],
        description="Equal demand from all directions. How does agent handle?"
    )
    
    # Scenario 3: Single lane spike
    tester.run_scenario(
        name="Single Lane Spike: N=15",
        initial_queues=[15, 0, 0, 0],
        description="Extreme congestion in single direction. Can agent clear efficiently?"
    )
    
    # Scenario 4: Two-direction congestion
    tester.run_scenario(
        name="N/S Heavy: N=8, S=8",
        initial_queues=[8, 8, 2, 2],
        description="North/South demand much higher than East/West"
    )
    
    # Scenario 5: Opposite imbalance
    tester.run_scenario(
        name="E/W Heavy: E=8, W=8",
        initial_queues=[2, 2, 8, 8],
        description="East/West demand much higher than North/South"
    )
    
    # Scenario 6: Diagonal traffic
    tester.run_scenario(
        name="Diagonal: N=6, W=6",
        initial_queues=[6, 0, 0, 6],
        description="Opposite corners have traffic. Tests phase selection logic."
    )
    
    # Scenario 7: Light traffic
    tester.run_scenario(
        name="Light Traffic: Total=4",
        initial_queues=[1, 1, 1, 1],
        description="Minimal traffic. How many phase changes needed?"
    )
    
    # Scenario 8: Staircase pattern
    tester.run_scenario(
        name="Staircase: 1,3,5,7",
        initial_queues=[1, 3, 5, 7],
        description="Increasing demand pattern. Does agent clear in optimal order?"
    )
    
    # Print comparisons
    print("\n\n COMPARISON WITH FIXED-TIMING BASELINE")
    print("="*70)
    
    comparison_scenarios = [
        ([0, 10, 0, 3], "Imbalanced: W=3, S=10"),
        ([5, 5, 5, 5], "Balanced: All=5"),
        ([15, 0, 0, 0], "Single Lane: N=15")
    ]
    
    for queues, name in comparison_scenarios:
        print(f"\n{name}:")
        print("-" * 70)
        comparison = tester.compare_with_baseline(queues)
        
        print(f"{'Metric':<20} {'PPO Agent':<20} {'Fixed-Timing':<20} {'Winner':<15}")
        print("-" * 70)
        
        ppo = comparison['ppo']
        fixed = comparison['fixed_timing']
        
        # Steps
        winner = 'PPO' if ppo['steps'] < fixed['steps'] else 'Fixed' if ppo['steps'] > fixed['steps'] else 'Tie'
        print(f"{'Steps':<20} {ppo['steps']:<20} {fixed['steps']:<20} {winner:<15}")
        
        # Efficiency
        winner = 'PPO' if ppo['efficiency'] > fixed['efficiency'] else 'Fixed' if ppo['efficiency'] < fixed['efficiency'] else 'Tie'
        print(f"{'Efficiency':<20} {ppo['efficiency']:.2f}{' '*17} {fixed['efficiency']:.2f}{' '*17} {winner:<15}")
        
        # Phase changes
        winner = 'PPO' if ppo['phase_changes'] < fixed['phase_changes'] else 'Fixed' if ppo['phase_changes'] > fixed['phase_changes'] else 'Tie'
        print(f"{'Phase Changes':<20} {ppo['phase_changes']:<20} {fixed['phase_changes']:<20} {winner:<15}")
    
    # Generate outputs
    tester.generate_report()
    tester.visualize_results()
    
    print("\n TESTING COMPLETE")
    print(f"Results saved to: {tester.output_dir}")
    print()
    
    # Save JSON
    json_path = os.path.join(tester.output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(tester.results, f, indent=2)
    print(f"[SAVED] JSON: {json_path}")


# Interactive Testing
def interactive_test():
    """
    Interactive mode - user can specify custom scenarios
    """
    MODEL_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/ppo_final_seed789.zip"
    VECNORM_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/vec_normalize_seed789.pkl"
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECNORM_PATH):
        print("[ERROR] Model files not found")
        return
    
    tester = ScenarioTester(MODEL_PATH, VECNORM_PATH)
    
    print("\n INTERACTIVE SCENARIO TESTING")
    print("="*70)
    print("\nEnter queue lengths for each direction (or 'q' to quit)\n")
    
    while True:
        try:
            north = input("North queue (0-20): ").strip()
            if north.lower() == 'q':
                break
            north = int(north)
            
            south = int(input("South queue (0-20): ").strip())
            east = int(input("East queue (0-20): ").strip())
            west = int(input("West queue (0-20): ").strip())
            
            if not all(0 <= q <= 20 for q in [north, south, east, west]):
                print("[ERROR] Queues must be between 0 and 20\n")
                continue
            
            name = input("Scenario name: ").strip()
            if not name:
                name = f"Custom: N={north}, S={south}, E={east}, W={west}"
            
            tester.run_scenario(
                name=name,
                initial_queues=[north, south, east, west],
                description="User-defined custom scenario"
            )
            
            print("\nTest another scenario? (Enter 'q' to quit, any key to continue)")
            
        except KeyboardInterrupt:
            print("\n\n[STOPPED] Testing stopped by user")
            break
        except ValueError:
            print("[ERROR] Invalid input. Please enter numbers 0-20\n")
    
    if tester.results:
        tester.generate_report()
        tester.visualize_results()
        print(f"\nResults saved to: {tester.output_dir}")


# Main Execution

def main():
    """
    Main execution - choose mode
    """
    print("\n PPO AGENT SCENARIO TESTING")
    print(" Run 8 Seed 789 - Multi-Seed Champion")
    print("="*70)
    print()
    print("Select mode:")
    print("  1. Run all predefined scenarios")
    print("  2. Interactive testing (custom scenarios)")
    print("  q. Quit")
    print()
    
    choice = input("Enter choice (1-2) or 'q': ").strip()
    
    if choice == '1':
        run_all_scenarios()
    elif choice == '2':
        interactive_test()
    elif choice.lower() in ['q', 'quit', 'exit']:
        print("[EXIT] Goodbye!")
    else:
        print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()
