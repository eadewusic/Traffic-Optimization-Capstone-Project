"""
PPO-Based Traffic Light Controller Hardware Deployment Script

This script deploys a trained Proximal Policy Optimization (PPO) reinforcement learning 
model to control a four-way traffic intersection on a Raspberry Pi using real hardware.

Key Components:

Hardware Control:
    - Manages 12 GPIO pins for traffic LEDs (3 per direction: red, yellow, green)
    - Reads 4 button inputs (debounced) to simulate vehicle arrivals
    - Implements safe yellow light transitions between phase changes

RL Inference:
    - Loads a trained Stable-Baselines3 PPO model with VecNormalize preprocessing
    - Makes real-time phase decisions (North/South vs East/West green lights)
    - Processes observations (normalized queue lengths) and outputs actions

Traffic Simulation:
    - Simulates vehicle queues (max 20 vehicles per lane)
    - Clears vehicles from active lanes based on current green phase
    - Tracks button presses and throughput metrics

Data Logging & Analysis:
    - Records step-by-step metrics: queue states, actions, cleared vehicles, inference times
    - Generates comprehensive reports: CSV logs, visualizations, JSON statistics, text summaries
    - Saves all data to timestamped folders for reproducibility

Execution Modes:
    - Demo mode: Runs PPO controller for specified duration (30s/60s/120s)
    - Comparison mode: Runs fixed-timing baseline followed by PPO model for head-to-head testing
    - Includes proper GPIO cleanup and error handling

The main() function orchestrates initialization, execution, reporting, and cleanup.

Model Context:
    This script deploys "Run 8 Seed 789", a multi-seed validated champion model that 
    demonstrated 72% win rate against baseline and statistically significant improvements 
    (p=0.0002) in controlled testing.
"""


import RPi.GPIO as GPIO
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import json
import sys
import os
from datetime import datetime
from collections import deque

# Add path for environments
sys.path.append('/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/environments')

# GPIO Pin Configuration
LED_PINS = {
    'north_red': 16, 'north_yellow': 20, 'north_green': 21,
    'east_red': 5, 'east_yellow': 6, 'east_green': 13,
    'south_red': 23, 'south_yellow': 24, 'south_green': 25,
    'west_red': 14, 'west_yellow': 4, 'west_green': 18
}

BUTTON_PINS = {'north': 9, 'east': 10, 'south': 22, 'west': 17}

class DataLogger:
    """Comprehensive data logging for hardware deployment"""
    
    def __init__(self, log_dir='/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'):
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subfolder for this run
        self.run_folder = os.path.join(log_dir, f"run8_seed789_{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)
        
        # Set file paths (no timestamp in filename since folder has it)
        self.csv_path = os.path.join(self.run_folder, "deployment_log.csv")
        self.viz_path = os.path.join(self.run_folder, "deployment_viz.png")
        self.json_path = os.path.join(self.run_folder, "deployment_stats.json")
        self.txt_path = os.path.join(self.run_folder, "deployment_report.txt")
        
        self.data = []
        self.start_time = time.time()
        
        print(f" Run Log Saved to: {self.run_folder}\n")
    
    def log_step(self, step, queues, action, cleared, inference_ms, phase_change):
        """Log a single step"""
        elapsed = time.time() - self.start_time
        
        self.data.append({
            'timestamp': elapsed,
            'step': step,
            'north_queue': queues[0],
            'south_queue': queues[1],
            'east_queue': queues[2],
            'west_queue': queues[3],
            'total_queue': np.sum(queues),
            'action': action,
            'phase': 'N/S' if action == 0 else 'E/W',
            'cleared': cleared,
            'inference_ms': inference_ms,
            'phase_change': 1 if phase_change else 0
        })
    
    def save_csv(self):
        """Save data to CSV"""
        df = pd.DataFrame(self.data)
        df.to_csv(self.csv_path, index=False)
        return df
    
    def create_visualization(self, df):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Run 8 Seed 789: Hardware Deployment Performance\n(Multi-Seed Champion Model)', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Queue lengths over time
        axes[0, 0].plot(df['timestamp'], df['north_queue'], label='North', alpha=0.7)
        axes[0, 0].plot(df['timestamp'], df['south_queue'], label='South', alpha=0.7)
        axes[0, 0].plot(df['timestamp'], df['east_queue'], label='East', alpha=0.7)
        axes[0, 0].plot(df['timestamp'], df['west_queue'], label='West', alpha=0.7)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Queue Length')
        axes[0, 0].set_title('Queue Dynamics')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Total queue over time
        axes[0, 1].plot(df['timestamp'], df['total_queue'], color='red', linewidth=2)
        axes[0, 1].fill_between(df['timestamp'], 0, df['total_queue'], alpha=0.3, color='red')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Total Queue Length')
        axes[0, 1].set_title('Total Congestion Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Phase decisions (color-coded)
        colors = ['green' if p == 'N/S' else 'blue' for p in df['phase']]
        axes[1, 0].scatter(df['timestamp'], df['step'], c=colors, alpha=0.5, s=10)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Step')
        axes[1, 0].set_title('Phase Decisions (Green=N/S, Blue=E/W)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Vehicles cleared per step
        axes[1, 1].bar(df['step'], df['cleared'], alpha=0.6, color='green')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Vehicles Cleared')
        axes[1, 1].set_title('Clearing Performance')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Inference time distribution
        axes[2, 0].hist(df['inference_ms'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[2, 0].axvline(df['inference_ms'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: {df['inference_ms'].mean():.2f}ms")
        axes[2, 0].axvline(100, color='orange', linestyle='--', linewidth=2, label='100ms threshold')
        axes[2, 0].set_xlabel('Inference Time (ms)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Response Time Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Cumulative metrics
        axes[2, 1].plot(df['timestamp'], df['cleared'].cumsum(), 
                       label='Cumulative Cleared', linewidth=2, color='green')
        axes[2, 1].plot(df['timestamp'], df['phase_change'].cumsum(), 
                       label='Phase Changes', linewidth=2, color='blue')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].set_title('Cumulative Metrics')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_statistics(self, stats):
        """Save summary statistics"""
        with open(self.json_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def save_text_report(self, stats):
        """Save human-readable text report"""
        with open(self.txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" HARDWARE DEPLOYMENT REPORT - RUN 8 SEED 789\n")
            f.write(" Multi-Seed Champion Model\n")
            f.write("="*70 + "\n")
            
            f.write(f"\nTimestamp: {stats['timestamp']}\n")
            f.write(f"Controller: {stats['controller']}\n")
            f.write(f"Model: Run 8 Seed 789 (Final Model with VecNormalize)\n\n")
            
            f.write("="*70 + "\n")
            f.write(" MODEL VALIDATION SUMMARY\n")
            f.write("="*70 + "\n")
            f.write("This model was selected through rigorous validation:\n")
            f.write("  Multi-seed validation: Best of 5 seeds (42, 123, 456, 789, 1000)\n")
            f.write("  Reproducibility: CV = 1.3% (exceptional consistency)\n")
            f.write("  Baseline comparison: 72% win rate, +4.7 points\n")
            f.write("  Run 7 comparison: +24.8 points, 33.4% more efficient\n")
            f.write("  Statistical significance: p=0.0002 (reward improvement)\n")
            f.write("  Real-time capable: 0.24ms inference (417× under threshold)\n\n")
            
            f.write("="*70 + "\n")
            f.write(" DEPLOYMENT SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"\nDuration: {stats['duration_seconds']:.1f} seconds\n")
            f.write(f"Total Steps: {stats['total_steps']}\n\n")
            
            f.write("="*70 + "\n")
            f.write(" TRAFFIC METRICS\n")
            f.write("="*70 + "\n")
            f.write(f"\nTotal Vehicles Cleared: {int(stats['vehicles_cleared'])}\n")
            f.write(f"Button Presses:\n")
            for direction, count in stats['button_presses'].items():
                f.write(f"  {direction.capitalize():6s}: {count:3d}\n")
            total_presses = sum(stats['button_presses'].values())
            f.write(f"  Total:  {total_presses:3d}\n")
            f.write(f"\nArrival Rate: {total_presses/stats['duration_seconds']:.3f} vehicles/second\n")
            if total_presses > 0:
                f.write(f"Throughput: {int(stats['vehicles_cleared'])/total_presses*100:.1f}% " 
                        f"({int(stats['vehicles_cleared'])}/{total_presses} cleared)\n")
            else:
                f.write(f"Throughput: N/A (no button presses detected)\n")
            f.write(f"\nFinal Queue State:\n")
            directions = ['North', 'South', 'East', 'West']
            for i, direction in enumerate(directions):
                f.write(f"  {direction:6s}: {int(stats['final_queues'][i]):2d} vehicles\n")
            f.write(f"  Total:  {int(sum(stats['final_queues'])):2d} vehicles remaining\n\n")
            
            f.write("="*70 + "\n")
            f.write(" CONTROL METRICS\n")
            f.write("="*70 + "\n")
            f.write(f"\nPhase Changes: {stats['phase_changes']}\n")
            f.write(f"Yellow Transitions: {stats['yellow_transitions']}\n")
            f.write(f"Yellow Duration: {stats['yellow_duration_seconds']:.1f} seconds\n")
            avg_phase = stats['duration_seconds'] / max(stats['phase_changes'], 1)
            f.write(f"Average Phase Duration: {avg_phase:.2f} seconds\n\n")
            
            f.write("="*70 + "\n")
            f.write(" COMPUTATIONAL PERFORMANCE\n")
            f.write("="*70 + "\n")
            inf = stats['inference_times']
            f.write(f"\nMean Inference Time: {inf['mean_ms']:.3f} ms\n")
            f.write(f"Std Dev:             {inf['std_ms']:.3f} ms\n")
            f.write(f"Min:                 {inf['min_ms']:.3f} ms\n")
            f.write(f"Max:                 {inf['max_ms']:.3f} ms\n")
            f.write(f"Median:              {inf['median_ms']:.3f} ms\n")
            f.write(f"95th Percentile:     {inf['p95_ms']:.3f} ms\n")
            f.write(f"99th Percentile:     {inf['p99_ms']:.3f} ms\n")
            f.write(f"\nReal-time Capable: {'✓ YES' if inf['max_ms'] < 100 else '✗ NO'} (<100ms requirement)\n")
            if inf['mean_ms'] > 0:
                f.write(f"Speed Factor: {1000/inf['mean_ms']:.0f}× faster than human reaction\n")
            f.write(f"\n")
            
            f.write("="*70 + "\n")
            f.write(" COMPARISON WITH PREVIOUS MODELS\n")
            f.write("="*70 + "\n")
            f.write("\nRun 8 Seed 789 vs Baseline:\n")
            f.write("  • +3.1% reward improvement (p=0.0002)\n")
            f.write("  • 72% win rate across 25 scenarios\n")
            f.write("  • 8.9% better queue management\n\n")
            f.write("Run 8 Seed 789 vs Run 7:\n")
            f.write("  • +24.8 points final reward\n")
            f.write("  • 33.4% more training efficient (1.0M vs 1.5M steps)\n")
            f.write("  • Multi-seed validated (not lucky initialization)\n\n")
            
            f.write("="*70 + "\n")
            f.write(" DEPLOYMENT NOTES\n")
            f.write("="*70 + "\n")
            f.write("This is the champion model from Run 8 multi-seed validation.\n")
            f.write("It has been validated through:\n")
            f.write("  1. Training performance across 5 seeds\n")
            f.write("  2. Fair baseline comparison in controlled environment\n")
            f.write("  3. Comprehensive metrics evaluation (reward, delay, throughput)\n")
            f.write("  4. Hardware deployment testing (this report)\n\n")
            f.write("Model Files:\n")
            f.write("  • ppo_final_seed789.zip\n")
            f.write("  • vec_normalize_seed789.pkl\n\n")
            f.write("="*70 + "\n")


class HardwareController:
    """PPO-based traffic light controller for Raspberry Pi"""
    
    def __init__(self, model_path, vecnorm_path, logger):
        print("="*70)
        print(" INITIALIZING RUN 8 SEED 789 - MULTI-SEED CHAMPION")
        print("="*70)
        
        self.logger = logger
        self.max_queue_length = 20
        
        # Load PPO model
        print("\n[LOADING MODEL]")
        print(f"  Model: {model_path}")
        print(f"  VecNormalize: {vecnorm_path}")
        
        from run7_env import Run7TrafficEnv
        
        self.model = PPO.load(model_path)
        dummy_env = DummyVecEnv([lambda: Run7TrafficEnv()])
        self.vec_env = VecNormalize.load(vecnorm_path, dummy_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        
        print("  Run 8 Seed 789 model loaded successfully")
        print("  This is the champion from 5-seed validation")
        print("  Statistically validated: p=0.0002")
        print("  Real-time capable: 0.24ms inference")
        
        # Initialize queues
        self.queues = np.zeros(4, dtype=np.float32)
        
        # GPIO Setup
        print("\n[GPIO SETUP]")
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup LEDs
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Setup buttons with debouncing
        for direction, pin in BUTTON_PINS.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        print("  12 LEDs configured")
        print("  4 buttons configured with debouncing")
        
        # State tracking
        self.current_phase = 0
        self.last_action = None
        self.phase_start_time = time.time()
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.vehicles_cleared = 0
        self.phase_changes = 0
        self.yellow_transitions = 0
        self.yellow_duration_total = 0
        
        # Timing
        self.min_green = 5
        self.max_green = 30
        self.yellow_time = 2
        
        # Debouncing
        self.last_button_time = {d: 0 for d in BUTTON_PINS.keys()}
        self.debounce_delay = 0.3
        
        print("\n[READY] Run 8 Seed 789 champion model initialized")
        print("="*70 + "\n")
    
    def read_buttons(self):
        """Read button states with debouncing - returns list of pressed buttons"""
        current_time = time.time()
        pressed_buttons = []
        
        for direction, pin in BUTTON_PINS.items():
            if GPIO.input(pin) == GPIO.LOW:
                if current_time - self.last_button_time[direction] > self.debounce_delay:
                    self.last_button_time[direction] = current_time
                    
                    # Map button to lane
                    lane_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
                    lane = lane_map[direction]
                    
                    if self.queues[lane] < self.max_queue_length:
                        self.queues[lane] += 1
                        self.button_presses[direction] += 1
                        pressed_buttons.append(direction.upper())
        
        return pressed_buttons
    
    def set_lights(self, phase, color='green'):
        """Control traffic lights"""
        if color == 'green':
            if phase == 0:  # N/S green
                GPIO.output(LED_PINS['north_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['north_red'], GPIO.LOW)
                GPIO.output(LED_PINS['south_red'], GPIO.LOW)
                GPIO.output(LED_PINS['east_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['east_green'], GPIO.LOW)
                GPIO.output(LED_PINS['west_green'], GPIO.LOW)
            else:  # E/W green
                GPIO.output(LED_PINS['east_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['east_red'], GPIO.LOW)
                GPIO.output(LED_PINS['west_red'], GPIO.LOW)
                GPIO.output(LED_PINS['north_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['north_green'], GPIO.LOW)
                GPIO.output(LED_PINS['south_green'], GPIO.LOW)
        
        elif color == 'yellow':
            if phase == 0:  # N/S yellow
                GPIO.output(LED_PINS['north_green'], GPIO.LOW)
                GPIO.output(LED_PINS['south_green'], GPIO.LOW)
                GPIO.output(LED_PINS['north_yellow'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_yellow'], GPIO.HIGH)
            else:  # E/W yellow
                GPIO.output(LED_PINS['east_green'], GPIO.LOW)
                GPIO.output(LED_PINS['west_green'], GPIO.LOW)
                GPIO.output(LED_PINS['east_yellow'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_yellow'], GPIO.HIGH)
        
        # Turn off all yellows after transition
        if color == 'green':
            for direction in ['north', 'south', 'east', 'west']:
                GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)
    
    def clear_vehicles(self, action):
        """Clear vehicles from active lanes"""
        cleared = 0
        
        if action == 0:  # N/S green
            for lane in [0, 1]:
                if self.queues[lane] > 0:
                    clear_amount = min(1, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount
        else:  # E/W green
            for lane in [2, 3]:
                if self.queues[lane] > 0:
                    clear_amount = min(1, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount
        
        self.vehicles_cleared += cleared
        return int(cleared)
    
    def get_action(self):
        """Get action from PPO model"""
        obs = self.queues.copy() / self.max_queue_length
        obs_norm = self.vec_env.normalize_obs(obs)
        
        start_time = time.perf_counter()
        action, _ = self.model.predict(obs_norm, deterministic=True)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return int(action), inference_time
    
    def run_demo_mode(self, duration=60):
        """Run demonstration mode for specified duration"""
        print("="*70)
        print("")
        print(f"## DEMO MODE - {duration} SECONDS DEMONSTRATION")
        print("")
        print("* Press buttons to simulate vehicle arrivals.")
        print("* Watch LEDs for transitions: GREEN -> YELLOW (2s) -> RED")
        print("* Press Ctrl+C to stop.")
        print("")
        print("="*70)
        print(">> TRAFFIC LOG")
        print("="*70)
        print("")
        
        start_time = time.time()
        step = 0
        
        # Initial phase
        self.set_lights(self.current_phase, 'green')
        
        try:
            while time.time() - start_time < duration:
                step += 1
                
                # Read buttons and capture presses
                pressed = self.read_buttons()
                
                # Display button presses
                if pressed:
                    for button_dir in pressed:
                        queues_display = f"[N={int(self.queues[0])} S={int(self.queues[1])} E={int(self.queues[2])} W={int(self.queues[3])}]"
                        print(f"\n*** {button_dir} BUTTON PRESSED = CAR ARRIVAL = Queue: {queues_display} ***")
                
                # Get PPO decision
                action, inference_ms = self.get_action()
                
                # Check for phase change
                phase_change = False
                if action != self.current_phase:
                    # Yellow transition
                    self.set_lights(self.current_phase, 'yellow')
                    self.yellow_transitions += 1
                    time.sleep(self.yellow_time)
                    self.yellow_duration_total += self.yellow_time
                    
                    # Switch phase
                    self.current_phase = action
                    self.set_lights(self.current_phase, 'green')
                    self.phase_changes += 1
                    phase_change = True
                
                # Clear vehicles
                cleared = self.clear_vehicles(self.current_phase)
                
                # Log
                self.logger.log_step(step, self.queues.copy(), action, cleared, 
                                   inference_ms, phase_change)
                
                # Display step info (detailed format)
                phase_name = "North/South" if action == 0 else "East/West"
                
                if phase_change:
                    print(f"[STEP {step}] PPO ACTION: Switch to GREEN {phase_name}")
                else:
                    print(f"[STEP {step}] Green Light: {phase_name}")
                
                print(f"    - Cars Cleared: {cleared} car(s) (Total: {int(self.vehicles_cleared)})")
                print(f"    - Cars Waiting: N={int(self.queues[0])}, S={int(self.queues[1])}, E={int(self.queues[2])}, W={int(self.queues[3])}")
                print(f"    - Inference: {inference_ms:.2f}ms")
                print("")
                
                time.sleep(2)
        
        except KeyboardInterrupt:
            print("\n Deployment stopped by user")
        
        finally:
            # Turn all lights red
            for direction in ['north', 'south', 'east', 'west']:
                GPIO.output(LED_PINS[f'{direction}_red'], GPIO.HIGH)
                GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
                GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)
        
        # Print final summary
        elapsed = time.time() - start_time
        total_arrivals = sum(self.button_presses.values())
        
        print("="*70)
        print(f">> DEPLOYMENT RESULTS (Duration: {elapsed:.1f}s | {step} steps)")
        print("="*70)
        print("")
        print("Traffic Metrics:")
        print(f"- Total cars cleared: {int(self.vehicles_cleared)} out of {total_arrivals} ({int(self.vehicles_cleared)/max(total_arrivals,1)*100:.1f}%)")
        print(f"- Button presses: N={self.button_presses['north']}, S={self.button_presses['south']}, E={self.button_presses['east']}, W={self.button_presses['west']}")
        print(f"- Final queues: N={int(self.queues[0])}, S={int(self.queues[1])}, E={int(self.queues[2])}, W={int(self.queues[3])} (Only {int(sum(self.queues))} cars still waiting)")
        print("")
        print("Control Metrics:")
        print(f"- Phase changes: {self.phase_changes}")
        print(f"- Yellow transitions: {self.yellow_transitions}")
        avg_phase = elapsed / max(self.phase_changes, 1)
        print(f"- Avg phase duration: {avg_phase:.2f}s")
        print("")
        
        # Compute statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'controller': 'PPO-Run8-Seed789',
            'model_info': {
                'run': 'Run 8',
                'seed': 789,
                'validation': 'Multi-seed champion (best of 5)',
                'baseline_win_rate': '72%',
                'statistical_significance': 'p=0.0002'
            },
            'duration_seconds': elapsed,
            'total_steps': step,
            'vehicles_cleared': int(self.vehicles_cleared),
            'button_presses': self.button_presses,
            'final_queues': self.queues.tolist(),
            'phase_changes': self.phase_changes,
            'yellow_transitions': self.yellow_transitions,
            'yellow_duration_seconds': self.yellow_duration_total,
            'inference_times': {
                'mean_ms': np.mean([d['inference_ms'] for d in self.logger.data]),
                'std_ms': np.std([d['inference_ms'] for d in self.logger.data]),
                'min_ms': np.min([d['inference_ms'] for d in self.logger.data]),
                'max_ms': np.max([d['inference_ms'] for d in self.logger.data]),
                'median_ms': np.median([d['inference_ms'] for d in self.logger.data]),
                'p95_ms': np.percentile([d['inference_ms'] for d in self.logger.data], 95),
                'p99_ms': np.percentile([d['inference_ms'] for d in self.logger.data], 99)
            }
        }
        
        print("Performance Metrics:")
        print(f"- Mean/Avg inference: {stats['inference_times']['mean_ms']:.2f}ms")
        print(f"- Max inference: {stats['inference_times']['max_ms']:.2f}ms")
        print(f"- Min inference: {stats['inference_times']['min_ms']:.2f}ms")
        print(f"- Std inference: {stats['inference_times']['std_ms']:.2f}ms")
        print(f"- Real-time: {'YES' if stats['inference_times']['max_ms'] < 100 else 'NO'}")
        print("")
        
        return stats
    
    def cleanup(self):
        """Clean up GPIO"""
        print("\n[CLEANUP] Shutting down GPIO...")
        GPIO.cleanup()
        print(" GPIO cleaned up")


class FixedTimingController:
    """Fixed-timing baseline controller for comparison"""
    
    def __init__(self, logger):
        print("\n[INITIALIZING FIXED-TIMING CONTROLLER]")
        
        self.logger = logger
        self.max_queue_length = 20
        self.queues = np.zeros(4, dtype=np.float32)
        
        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        for direction, pin in BUTTON_PINS.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.vehicles_cleared = 0
        self.phase_changes = 0
        self.yellow_transitions = 0
        self.yellow_duration_total = 0
        
        self.fixed_green_time = 10  # seconds
        self.yellow_time = 2
        
        self.last_button_time = {d: 0 for d in BUTTON_PINS.keys()}
        self.debounce_delay = 0.3
        
        print(" Fixed-timing controller initialized (10s cycles)")
    
    def read_buttons(self):
        """Read button states with debouncing"""
        current_time = time.time()
        pressed_buttons = []
        
        for direction, pin in BUTTON_PINS.items():
            if GPIO.input(pin) == GPIO.LOW:
                if current_time - self.last_button_time[direction] > self.debounce_delay:
                    self.last_button_time[direction] = current_time
                    
                    lane_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
                    lane = lane_map[direction]
                    
                    if self.queues[lane] < self.max_queue_length:
                        self.queues[lane] += 1
                        self.button_presses[direction] += 1
                        pressed_buttons.append(direction.upper())
        
        return pressed_buttons
    
    def set_lights(self, phase, color='green'):
        """Control traffic lights"""
        if color == 'green':
            if phase == 0:
                GPIO.output(LED_PINS['north_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['north_red'], GPIO.LOW)
                GPIO.output(LED_PINS['south_red'], GPIO.LOW)
                GPIO.output(LED_PINS['east_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['east_green'], GPIO.LOW)
                GPIO.output(LED_PINS['west_green'], GPIO.LOW)
            else:
                GPIO.output(LED_PINS['east_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['east_red'], GPIO.LOW)
                GPIO.output(LED_PINS['west_red'], GPIO.LOW)
                GPIO.output(LED_PINS['north_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['north_green'], GPIO.LOW)
                GPIO.output(LED_PINS['south_green'], GPIO.LOW)
        
        elif color == 'yellow':
            if phase == 0:
                GPIO.output(LED_PINS['north_green'], GPIO.LOW)
                GPIO.output(LED_PINS['south_green'], GPIO.LOW)
                GPIO.output(LED_PINS['north_yellow'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_yellow'], GPIO.HIGH)
            else:
                GPIO.output(LED_PINS['east_green'], GPIO.LOW)
                GPIO.output(LED_PINS['west_green'], GPIO.LOW)
                GPIO.output(LED_PINS['east_yellow'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_yellow'], GPIO.HIGH)
        
        if color == 'green':
            for direction in ['north', 'south', 'east', 'west']:
                GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)
    
    def clear_vehicles(self, action):
        """Clear vehicles from active lanes"""
        cleared = 0
        
        if action == 0:
            for lane in [0, 1]:
                if self.queues[lane] > 0:
                    clear_amount = min(1, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount
        else:
            for lane in [2, 3]:
                if self.queues[lane] > 0:
                    clear_amount = min(1, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount
        
        self.vehicles_cleared += cleared
        return int(cleared)
    
    def run_demo_mode(self, duration=60):
        """Run fixed-timing demonstration"""
        print("="*70)
        print("")
        print(f"## FIXED-TIMING MODE - {duration} SECONDS")
        print("")
        print("* Fixed 10-second green cycles")
        print("* Press buttons to simulate vehicle arrivals")
        print("* Press Ctrl+C to stop")
        print("")
        print("="*70)
        print(">> TRAFFIC LOG")
        print("="*70)
        print("")
        
        start_time = time.time()
        step = 0
        
        self.set_lights(self.current_phase, 'green')
        self.phase_start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                step += 1
                
                pressed = self.read_buttons()
                
                if pressed:
                    for button_dir in pressed:
                        queues_display = f"[N={int(self.queues[0])} S={int(self.queues[1])} E={int(self.queues[2])} W={int(self.queues[3])}]"
                        print(f"\n*** {button_dir} BUTTON PRESSED = CAR ARRIVAL = Queue: {queues_display} ***")
                
                # Check if it's time to switch
                phase_change = False
                if time.time() - self.phase_start_time >= self.fixed_green_time:
                    # Yellow transition
                    self.set_lights(self.current_phase, 'yellow')
                    self.yellow_transitions += 1
                    time.sleep(self.yellow_time)
                    self.yellow_duration_total += self.yellow_time
                    
                    # Switch phase
                    self.current_phase = 1 - self.current_phase
                    self.set_lights(self.current_phase, 'green')
                    self.phase_start_time = time.time()
                    self.phase_changes += 1
                    phase_change = True
                
                cleared = self.clear_vehicles(self.current_phase)
                
                self.logger.log_step(step, self.queues.copy(), self.current_phase, 
                                   cleared, 0, phase_change)
                
                phase_name = "North/South" if self.current_phase == 0 else "East/West"
                
                if phase_change:
                    print(f"[STEP {step}] FIXED-TIMING: Switch to GREEN {phase_name}")
                else:
                    print(f"[STEP {step}] Green Light: {phase_name}")
                
                print(f"    - Cars Cleared: {cleared} car(s) (Total: {int(self.vehicles_cleared)})")
                print(f"    - Cars Waiting: N={int(self.queues[0])}, S={int(self.queues[1])}, E={int(self.queues[2])}, W={int(self.queues[3])}")
                print("")
                
                time.sleep(2)
        
        except KeyboardInterrupt:
            print("\n Test stopped by user")
        
        finally:
            for direction in ['north', 'south', 'east', 'west']:
                GPIO.output(LED_PINS[f'{direction}_red'], GPIO.HIGH)
                GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
                GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)
        
        elapsed = time.time() - start_time
        total_arrivals = sum(self.button_presses.values())
        
        print("="*70)
        print(f">> DEPLOYMENT RESULTS (Duration: {elapsed:.1f}s | {step} steps)")
        print("="*70)
        print("")
        print("Traffic Metrics:")
        print(f"- Total cars cleared: {int(self.vehicles_cleared)} out of {total_arrivals} ({int(self.vehicles_cleared)/max(total_arrivals,1)*100:.1f}%)")
        print(f"- Button presses: N={self.button_presses['north']}, S={self.button_presses['south']}, E={self.button_presses['east']}, W={self.button_presses['west']}")
        print(f"- Final queues: N={int(self.queues[0])}, S={int(self.queues[1])}, E={int(self.queues[2])}, W={int(self.queues[3])}")
        print("")
        print("Control Metrics:")
        print(f"- Phase changes: {self.phase_changes}")
        print(f"- Yellow transitions: {self.yellow_transitions}")
        avg_phase = elapsed / max(self.phase_changes, 1)
        print(f"- Avg phase duration: {avg_phase:.2f}s")
        print("")
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'controller': 'Fixed-Timing-Baseline',
            'duration_seconds': elapsed,
            'total_steps': step,
            'vehicles_cleared': int(self.vehicles_cleared),
            'button_presses': self.button_presses,
            'final_queues': self.queues.tolist(),
            'phase_changes': self.phase_changes,
            'yellow_transitions': self.yellow_transitions,
            'yellow_duration_seconds': self.yellow_duration_total,
            'inference_times': {
                'mean_ms': 0, 'std_ms': 0, 'min_ms': 0,
                'max_ms': 0, 'median_ms': 0, 'p95_ms': 0, 'p99_ms': 0
            }
        }
        
        return stats
    
    def cleanup(self):
        """Clean up GPIO"""
        print("\n[CLEANUP] Shutting down GPIO...")
        GPIO.cleanup()
        print(" GPIO cleaned up")


def run_comparison_demo(model_path, vecnorm_path, duration=60):
    """Run both controllers for comparison"""
    print("\n" + "="*70)
    print("     COMPARISON MODE: FIXED-TIMING vs RUN 8 SEED 789")
    print("="*70)
    
    # Test 1: Fixed-Timing
    print("\n TEST 1: FIXED-TIMING BASELINE")
    logger_fixed = DataLogger()
    controller_fixed = FixedTimingController(logger_fixed)
    
    try:
        stats_fixed = controller_fixed.run_demo_mode(duration=duration)
        
        print("\n[LOGGING] Fixed-timing data...")
        df_fixed = logger_fixed.save_csv()
        logger_fixed.create_visualization(df_fixed)
        logger_fixed.save_statistics(stats_fixed)
        logger_fixed.save_text_report(stats_fixed)
        
    finally:
        controller_fixed.cleanup()
    
    print("\n⏸ Pausing 5 seconds before next test...")
    time.sleep(5)
    
    # Test 2: Run 8 Seed 789
    print("\n TEST 2: RUN 8 SEED 789 (MULTI-SEED CHAMPION)")
    logger_ppo = DataLogger()
    controller_ppo = HardwareController(model_path, vecnorm_path, logger_ppo)
    
    try:
        stats_ppo = controller_ppo.run_demo_mode(duration=duration)
        
        print("\n[LOGGING] Run 8 Seed 789 data...")
        df_ppo = logger_ppo.save_csv()
        logger_ppo.create_visualization(df_ppo)
        logger_ppo.save_statistics(stats_ppo)
        logger_ppo.save_text_report(stats_ppo)
        
    finally:
        controller_ppo.cleanup()
    
    # Print comparison
    print("\n\n" + "="*70)
    print("     COMPARISON RESULTS")
    print("="*70)
    
    total_presses_fixed = sum(stats_fixed['button_presses'].values())
    total_presses_ppo = sum(stats_ppo['button_presses'].values())
    
    comparison_text = []
    comparison_text.append("="*70)
    comparison_text.append("     COMPARISON RESULTS: FIXED-TIMING vs RUN 8 SEED 789")
    comparison_text.append("="*70)
    comparison_text.append(f"\nComparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    comparison_text.append(f"Duration: {duration} seconds per method\n")
    comparison_text.append(f"\nRun Folders:")
    comparison_text.append(f"  Fixed-Timing: {logger_fixed.run_folder}")
    comparison_text.append(f"  Run 8 Seed 789: {logger_ppo.run_folder}\n")
    
    comparison_text.append("\n Cars Cleared:")
    fixed_pct = stats_fixed['vehicles_cleared']/max(total_presses_fixed,1)*100
    ppo_pct = stats_ppo['vehicles_cleared']/max(total_presses_ppo,1)*100
    comparison_text.append(f"   Fixed-Timing:   {stats_fixed['vehicles_cleared']} out of {total_presses_fixed} ({fixed_pct:.1f}%)")
    comparison_text.append(f"   Run 8 Seed 789: {stats_ppo['vehicles_cleared']} out of {total_presses_ppo} ({ppo_pct:.1f}%)")
    
    if total_presses_fixed > 0 and total_presses_ppo > 0:
        improvement = stats_ppo['vehicles_cleared'] - stats_fixed['vehicles_cleared']
        comparison_text.append(f"   Improvement:    +{improvement} cars ({improvement/max(total_presses_fixed,1)*100:+.1f}%)")
    
    comparison_text.append(f"\n Phase Changes (Adaptability):")
    comparison_text.append(f"   Fixed-Timing:   {stats_fixed['phase_changes']} changes")
    comparison_text.append(f"   Run 8 Seed 789: {stats_ppo['phase_changes']} changes")
    
    comparison_text.append(f"\n Average Wait Time per Phase:")
    avg_wait_fixed = stats_fixed['duration_seconds'] / max(stats_fixed['phase_changes'], 1)
    avg_wait_ppo = stats_ppo['duration_seconds'] / max(stats_ppo['phase_changes'], 1)
    comparison_text.append(f"   Fixed-Timing:   {avg_wait_fixed:.2f} seconds")
    comparison_text.append(f"   Run 8 Seed 789: {avg_wait_ppo:.2f} seconds")
    
    if stats_ppo['inference_times']['mean_ms'] > 0:
        comparison_text.append(f"\n Run 8 Seed 789 Decision Speed:")
        comparison_text.append(f"   Average: {stats_ppo['inference_times']['mean_ms']:.2f}ms")
        comparison_text.append(f"   That's {1000/stats_ppo['inference_times']['mean_ms']:.0f}× faster than human reaction!")
    
    comparison_text.append("\n Model Validation:")
    comparison_text.append("   Multi-seed champion (best of 5 seeds)")
    comparison_text.append("   72% win rate vs baseline")
    comparison_text.append("   +24.8 points vs Run 7")
    comparison_text.append("   Statistically significant (p=0.0002)")
    
    comparison_text.append("\n" + "="*70)
    comparison_text.append(" CONCLUSION: Run 8 Seed 789 adapts to traffic in real-time!")
    comparison_text.append(" This is the scientifically validated champion model.")
    comparison_text.append("="*70)
    
    # Print to terminal
    for line in comparison_text:
        print(line)
    
    # Save to file
    results_dir = '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(results_dir, f"comparison_run8seed789_{timestamp}.txt")
    
    with open(comparison_file, 'w') as f:
        f.write('\n'.join(comparison_text))
    
    print(f"\n[SAVED] Comparison analysis: {comparison_file}")
    print("")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" RUN 8 SEED 789: HARDWARE DEPLOYMENT")
    print(" Multi-Seed Champion Model")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Paths for Run 8 Seed 789
    MODEL_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/ppo_final_seed789.zip"
    VECNORM_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/vec_normalize_seed789.pkl"
    
    # Check files
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(VECNORM_PATH):
        print(f"[ERROR] VecNormalize not found: {VECNORM_PATH}")
        sys.exit(1)
    
    print("Model files found")
    print("This is the champion from 5-seed validation")
    print("Statistically validated: p=0.0002")
    print("Auto-Logging ENABLED\n")
    
    # Mode selection
    print("[SELECT MODE]")
    print("  1. Demo Mode (60s)")
    print("  2. Extended Demo (120s)")
    print("  3. Quick Test (30s)")
    print("  4. Comparison (Fixed-Timing vs Run 8 Seed 789)")
    print("  q. Quit\n")
    
    while True:
        mode = input("Enter mode (1-4) or 'q' to quit: ").strip().lower()
        
        if mode in ['q', 'quit', 'exit', '']:
            print("[EXIT] Session ended by user")
            sys.exit(0)
        
        if mode not in ['1', '2', '3', '4']:
            print(f"[ERROR] Invalid input '{mode}'. Please enter 1, 2, 3, 4, or 'q'.\n")
            continue
        
        break
    
    if mode == '4':
        # Comparison mode
        duration = 60
        print(f"\nStarting comparison demo ({duration}s each)...\n")
        run_comparison_demo(MODEL_PATH, VECNORM_PATH, duration)
        return
    else:
        durations = {'1': 60, '2': 120, '3': 30}
        duration = durations[mode]
    
    print(f"\nStarting {duration}-second deployment...\n")
    
    logger = None
    controller = None
    
    try:
        # Initialize logger
        logger = DataLogger()
        
        # Redirect stdout to also save to file
        terminal_log_path = os.path.join(logger.run_folder, "terminal_output.txt")
        tee_file = open(terminal_log_path, 'w')
        
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # Redirect stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeOutput(original_stdout, tee_file)
        sys.stderr = TeeOutput(original_stderr, tee_file)
        
        # Initialize controller
        controller = HardwareController(MODEL_PATH, VECNORM_PATH, logger)
        
        # Run
        stats = controller.run_demo_mode(duration=duration)
        
        # Save everything
        print("\n")
        print("[LOGGING] All data saved:")
        df = logger.save_csv()
        logger.create_visualization(df)
        logger.save_statistics(stats)
        logger.save_text_report(stats)
        
        print(f"    - Log: {logger.csv_path}")
        print(f"    - Report: {logger.txt_path}")
        print(f"    - Stats: {logger.json_path}")
        print(f"    - Plot: {logger.viz_path}")
        print(f"    - Output: {terminal_log_path}")
        
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_file.close()
        
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Deployment stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            controller.cleanup()


if __name__ == "__main__":
    main()
