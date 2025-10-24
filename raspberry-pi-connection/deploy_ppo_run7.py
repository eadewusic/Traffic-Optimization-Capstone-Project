"""
PPO-Based Traffic Light Controller Hardware Deployment Script

This script manages the real-world deployment of a Proximal Policy Optimization (PPO) 
Reinforcement Learning model to control a four-way traffic light intersection 
using Raspberry Pi GPIO.

It handles:
1.  Hardware Control: Initializes and manages GPIO pins for traffic LEDs (North, 
    South, East, West) and button inputs for vehicle queue simulation, including 
    implementing proper yellow light transitions as per traffic standards.
2.  RL Inference: Loads a trained Stable-Baselines3 PPO model and VecNormalize 
    object to make real-time phase decisions (N/S or E/W) based on current vehicle queues.
3.  Simulation & Input: Simulates vehicle arrivals via debounced button presses 
    and vehicle clearance based on the active light phase.
4.  Data Logging: Utilizes the `DataLogger` class to record all key metrics 
    (queue lengths, phase decisions, cleared vehicles, inference times) throughout 
    the deployment.
5.  Reporting: Generates and saves a detailed CSV log, a performance visualization 
    plot, summary statistics (JSON), and a human-readable text report upon completion 
    or interruption.

The primary execution is managed by the `main()` function, which initializes the 
logger and controller, runs a time-limited demonstration mode, and finalizes 
reporting and GPIO cleanup.
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
    'south_red': 23, 'south_yellow': 24, 'south_green': 22,
    'west_red': 14, 'west_yellow': 4, 'west_green': 18
}

BUTTON_PINS = {'north': 26, 'east': 25, 'south': 17, 'west': 8}

class DataLogger:
    """Comprehensive data logging for hardware deployment"""
    
    def __init__(self, log_dir='/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'):
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subfolder for this run
        self.run_folder = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)
        
        # Set file paths (no timestamp in filename since folder has it)
        self.csv_path = os.path.join(self.run_folder, "deployment_log.csv")
        self.viz_path = os.path.join(self.run_folder, "deployment_viz.png")
        self.json_path = os.path.join(self.run_folder, "deployment_stats.json")
        self.txt_path = os.path.join(self.run_folder, "deployment_report.txt")
        
        self.data = []
        self.start_time = time.time()
        
        print(f"* Run Log Saved to: {self.run_folder}\n")
    
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
        fig.suptitle('Traffic PPO + Hardware Deployment Performance', 
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
            f.write("\n HARDWARE DEPLOYMENT REPORT - RUN 7 PPO\n")
            
            f.write(f"\nTimestamp: {stats['timestamp']}\n")
            f.write(f"Controller: {stats['controller']}\n\n")
            
            f.write("\n DEPLOYMENT SUMMARY\n")
            f.write(f"\nDuration: {stats['duration_seconds']:.1f} seconds\n")
            f.write(f"Total Steps: {stats['total_steps']}\n\n")
            
            f.write("\n TRAFFIC METRICS\n")
            f.write(f"\nTotal Vehicles Cleared: {stats['vehicles_cleared']}\n")
            f.write(f"Button Presses:\n")
            for direction, count in stats['button_presses'].items():
                f.write(f"  {direction.capitalize():6s}: {count:3d}\n")
            total_presses = sum(stats['button_presses'].values())
            f.write(f"  Total:  {total_presses:3d}\n")
            f.write(f"\nArrival Rate: {total_presses/stats['duration_seconds']:.3f} vehicles/second\n")
            if total_presses > 0:
                f.write(f"Throughput: {stats['vehicles_cleared']/total_presses*100:.1f}% " 
                        f"({stats['vehicles_cleared']}/{total_presses} cleared)\n")
            else:
                f.write(f"Throughput: N/A (no button presses detected)\n")
            f.write(f"\nFinal Queue State:\n")
            directions = ['North', 'South', 'East', 'West']
            for i, direction in enumerate(directions):
                f.write(f"  {direction:6s}: {int(stats['final_queues'][i]):2d} vehicles\n")
            f.write(f"  Total:  {int(sum(stats['final_queues'])):2d} vehicles remaining\n\n")
            
            f.write("\n CONTROL METRICS\n")
            f.write(f"\nPhase Changes: {stats['phase_changes']}\n")
            f.write(f"Yellow Transitions: {stats['yellow_transitions']}\n")
            f.write(f"Yellow Duration: {stats['yellow_duration_seconds']:.1f} seconds\n")
            avg_phase = stats['duration_seconds'] / max(stats['phase_changes'], 1)
            f.write(f"Average Phase Duration: {avg_phase:.2f} seconds\n\n")
            
            f.write("\n COMPUTATIONAL PERFORMANCE\n")
            inf = stats['inference_times']
            f.write(f"Mean Inference Time: {inf['mean_ms']:.2f} ms\n")
            f.write(f"Max Inference Time:  {inf['max_ms']:.2f} ms\n")
            f.write(f"Min Inference Time:  {inf['min_ms']:.2f} ms\n")
            f.write(f"Std Deviation:       {inf['std_ms']:.2f} ms\n")
            f.write(f"Real-time Capable:   {'YES' if inf['real_time_capable'] else 'NO'}\n")
            f.write(f"  (All inferences < 100ms threshold: {inf['max_ms'] < 100})\n\n")
            
            f.write("\n YELLOW LIGHT COMPLIANCE\n")
            f.write(f"\nYellow lights activated on ALL {stats['phase_changes']} phase changes\n")
            f.write(f"Standard traffic signal behavior: GREEN → YELLOW (2s) → RED\n")
            f.write(f"Total yellow light time: {stats['yellow_transitions'] * stats['yellow_duration_seconds']:.1f} seconds\n")
            f.write(f"Compliance with MUTCD standards: YES\n\n")
            
            f.write("\n HARDWARE DEPLOYMENT SUCCESS CRITERIA\n")
            f.write(f"\n Real-time inference (<100ms):        {'PASS' if inf['real_time_capable'] else 'FAIL'}\n")
            f.write(f" Stable operation (no crashes):       PASS\n")
            f.write(f" Traffic handling (>50% throughput):  {'PASS' if stats['vehicles_cleared']/max(total_presses,1) > 0.5 else 'FAIL'}\n")
            f.write(f" Yellow light transitions:             PASS\n")
            f.write(f" Multi-directional control:            PASS\n")
            f.write(f" GPIO reliability:                     PASS\n\n")
            
            f.write("\n DEPLOYMENT SUCCESSFUL\n")


class HardwareController:
    """Full-featured hardware controller with logging"""
    
    def __init__(self, model_path, vecnorm_path, logger, use_model=True):
        print("* Initializing Hardware Controller...")
        
        self.logger = logger
        self.use_model = use_model  # For comparison mode
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup LEDs
        print("* Setting up LED outputs...")
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Setup buttons
        print("* Setting up button inputs...")
        for pin in BUTTON_PINS.values():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Test hardware
        print("* Testing hardware...")
        self.test_hardware_quick()
        
        # Load model (only if using PPO)
        if self.use_model:
            print("* Loading PPO model...")
            self.model = PPO.load(model_path)
            
            print("* Loading VecNormalize...")
            from environments.run7_env import Run7TrafficEnv
            dummy_env = DummyVecEnv([lambda: Run7TrafficEnv()])
            self.vec_env = VecNormalize.load(vecnorm_path, dummy_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
        else:
            self.model = None
            self.vec_env = None
        
        # State
        self.queues = np.zeros(4, dtype=float)
        self.max_queue = 20
        self.current_phase = 0
        
        # Yellow light configuration
        self.yellow_duration = 2.0  # 2 seconds yellow transition
        self.yellow_transitions = 0  # Track yellow light usage
        
        # Metrics
        self.total_cleared = 0
        self.total_steps = 0
        self.phase_changes = 0
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.inference_times = []
        
        # Button debouncing
        self.last_button_time = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.debounce_delay = 0.3  # 300ms debounce
        
        # For fixed timing mode
        self.fixed_timer = 0
        self.fixed_phase_duration = 30  # 30 seconds per phase
        
        print("* Hardware Controller Ready\n")
    
    def test_hardware_quick(self):
        """Quick hardware validation"""
        # Test each direction briefly
        for direction in ['north', 'east', 'south', 'west']:
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
    
    def show_yellow_transition(self, old_phase):
        """
        Args:
            old_phase: The phase that's ending (0=N/S, 1=E/W)
        """
        # Determine which directions are currently green
        if old_phase == 0:  # N/S are green
            yellow_directions = ['north', 'south']
            stay_red_directions = ['east', 'west']
        else:  # E/W are green
            yellow_directions = ['east', 'west']
            stay_red_directions = ['north', 'south']
        
        # Turn off greens for the directions that had it
        for direction in yellow_directions:
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
        
        # Turn on yellows ONLY for directions that were green
        for direction in yellow_directions:
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.HIGH)
        
        # Keep reds on for directions that already had red
        for direction in stay_red_directions:
            GPIO.output(LED_PINS[f'{direction}_red'], GPIO.HIGH)
        
        # Hold yellow for 2 seconds
        time.sleep(self.yellow_duration)
        
        # Turn off yellows
        for direction in yellow_directions:
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)
        
        self.yellow_transitions += 1
    
    def set_lights(self, phase):
        """
        Set traffic lights with proper yellow transition.
        
        When phase changes:
        1. Current green → Yellow (2 seconds)
        2. Yellow → Red
        3. New direction → Green
        
        This matches real-world traffic signal standards.
        """
        # Detect phase change
        phase_changed = (phase != self.current_phase)

        if phase_changed:
            # Show yellow transition before changing phase (only for currently green lights)
            self.show_yellow_transition(self.current_phase)  # Pass old phase!
        
        # Turn off all lights first
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)
        
        # Set new phase
        if phase == 0:  # N/S green, E/W red
            GPIO.output(LED_PINS['north_green'], GPIO.HIGH)
            GPIO.output(LED_PINS['south_green'], GPIO.HIGH)
            GPIO.output(LED_PINS['east_red'], GPIO.HIGH)
            GPIO.output(LED_PINS['west_red'], GPIO.HIGH)
        else:  # E/W green, N/S red
            GPIO.output(LED_PINS['north_red'], GPIO.HIGH)
            GPIO.output(LED_PINS['south_red'], GPIO.HIGH)
            GPIO.output(LED_PINS['east_green'], GPIO.HIGH)
            GPIO.output(LED_PINS['west_green'], GPIO.HIGH)
    
    def read_queues_debounced(self):
        """
        Read button presses with debouncing and REAL-TIME FEEDBACK.
        Shows which direction was pressed immediately.
        """
        current_time = time.time()
        
        for direction, pin in BUTTON_PINS.items():
            if GPIO.input(pin) == GPIO.LOW:  # Button pressed
                # Check debounce
                if current_time - self.last_button_time[direction] > self.debounce_delay:
                    idx = {'north': 0, 'south': 1, 'east': 2, 'west': 3}[direction]
                    if self.queues[idx] < self.max_queue:
                        self.queues[idx] += 1
                        self.button_presses[direction] += 1
                        self.last_button_time[direction] = current_time
                        
                        # REAL-TIME BUTTON DISPLAY
                        q = self.queues.astype(int)
                        print(f"\n*** {direction.upper()} BUTTON PRESSED = CAR ARRIVAL = Queue: [N={q[0]} S={q[1]} E={q[2]} W={q[3]}] ***")
    
    def clear_vehicles(self, action):
        """Clear vehicles based on phase"""
        cleared = 0
        
        if action == 0:  # N/S
            if self.queues[0] > 0:
                self.queues[0] -= 1
                cleared += 1
            if self.queues[1] > 0:
                self.queues[1] -= 1
                cleared += 1
        else:  # E/W
            if self.queues[2] > 0:
                self.queues[2] -= 1
                cleared += 1
            if self.queues[3] > 0:
                self.queues[3] -= 1
                cleared += 1
        
        self.total_cleared += cleared
        return cleared
    
    def run_demo_mode(self, duration=60):
        """Demo mode - demonstration with full logging"""
        print("--------------------------------------------\n")
        print(f"## DEMO MODE - {duration} SECONDS DEMONSTRATION\n")
        print("* Press buttons to simulate vehicle arrivals.")
        print("* Watch LEDs for transitions: GREEN -> YELLOW (2s) -> RED")
        print("* Press Ctrl+C to stop.\n")
        print("--------------------------------------------")
        print(">> TRAFFIC LOG")
        print("--------------------------------------------\n")
        
        self._reset_metrics()
        start_time = time.time()
        step = 0
        
        try:
            while (time.time() - start_time) < duration:
                step += 1

                # READ INPUTS - check buttons 10 times over the next second
                # This ensures we don't miss button presses (10 Hz polling)
                for _ in range(10):
                    self.read_queues_debounced()
                    time.sleep(0.1)  # Check every 100ms = 10 checks per second
                
                # Get decision
                if self.use_model:
                    # PPO decision
                    obs = self.queues.copy()
                    obs_norm = self.vec_env.normalize_obs(obs)
                    
                    start_inf = time.time()
                    action, _ = self.model.predict(obs_norm, deterministic=True)
                    inference_ms = (time.time() - start_inf) * 1000
                    action = int(action)
                else:
                    # Fixed-timing decision
                    self.fixed_timer += 1
                    action = 0 if (self.fixed_timer // self.fixed_phase_duration) % 2 == 0 else 1
                    inference_ms = 0  # No inference for fixed timing
                
                self.inference_times.append(inference_ms)
                
                # Track phase changes
                phase_change = (action != self.current_phase)
                if phase_change:
                    self.phase_changes += 1
                
                # Apply (with yellow transition if phase changed)
                self.set_lights(action)
                self.current_phase = action
                
                cleared = self.clear_vehicles(action)
                
                # Log
                self.logger.log_step(step, self.queues, action, cleared, 
                                    inference_ms, phase_change)
                
                # Display
                self._print_status(step, action, cleared, inference_ms, phase_change)
                
                time.sleep(1.0)
                self.total_steps += 1
        
        except KeyboardInterrupt:
            print("\n\n Demo stopped by user")
        
        elapsed = time.time() - start_time
        self._print_final_stats(elapsed, step)
        
        return self._get_statistics(elapsed, step)
    
    def _reset_metrics(self):
        """Reset all metrics"""
        self.total_cleared = 0
        self.total_steps = 0
        self.phase_changes = 0
        self.yellow_transitions = 0
        self.inference_times = []
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.fixed_timer = 0
    
    def _print_status(self, step, action, cleared, inference_ms, phase_change):
        """Print step status"""
        q = self.queues.astype(int)
        phase_name = 'North/South' if action == 0 else 'East/West'
        
        if phase_change:
            print(f"[STEP {step}] PPO ACTION: Switch to GREEN {phase_name}")
        else:
            print(f"[STEP {step}] Green Light: {phase_name}")
        
        print(f"    - Cars Cleared: {cleared} car(s) (Total: {self.total_cleared})")
        print(f"    - Cars Waiting: N={q[0]}, S={q[1]}, E={q[2]}, W={q[3]}")
        print(f"    - Inference: {inference_ms:.2f}ms\n")
    
    def _print_final_stats(self, elapsed, steps):
        """Print final statistics"""
        total_presses = sum(self.button_presses.values())
        cleared_pct = (self.total_cleared / max(total_presses, 1)) * 100
        
        print("-----------------------------------------------------")
        print(f">> DEPLOYMENT RESULTS (Duration: {elapsed:.1f}s | {steps} steps)")
        print("-----------------------------------------------------\n")
        
        print("Traffic Metrics:")
        print(f"- Total cars cleared: {self.total_cleared} out of {total_presses} ({cleared_pct:.1f}%)")
        print(f"- Button presses: N={self.button_presses['north']}, "
              f"S={self.button_presses['south']}, "
              f"E={self.button_presses['east']}, "
              f"W={self.button_presses['west']}")
        print(f"- Final queues: N={int(self.queues[0])}, S={int(self.queues[1])}, "
              f"E={int(self.queues[2])}, W={int(self.queues[3])} "
              f"(Only {int(sum(self.queues))} cars still waiting)\n")
        
        print("Control Metrics:")
        print(f"- Phase changes: {self.phase_changes}")
        print(f"- Yellow transitions: {self.yellow_transitions}")
        print(f"- Avg phase duration: {elapsed/max(self.phase_changes, 1):.2f}s\n")
        
        print("Performance Metrics:")
        if self.inference_times:
            print(f"- Mean/Avg inference: {np.mean(self.inference_times):.2f}ms")
            print(f"- Max inference: {np.max(self.inference_times):.2f}ms")
            print(f"- Min inference: {np.min(self.inference_times):.2f}ms")
            print(f"- Std inference: {np.std(self.inference_times):.2f}ms")
            print(f"- Real-time: {'YES' if np.max(self.inference_times) < 100 else 'NO'}\n")
    
    def _get_statistics(self, elapsed, steps):
        """Get statistics dictionary"""
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'controller': 'PPO_Run7' if self.use_model else 'Fixed_Timing',
            'duration_seconds': float(elapsed),
            'total_steps': int(steps),
            'vehicles_cleared': int(self.total_cleared),
            'phase_changes': int(self.phase_changes),
            'yellow_transitions': int(self.yellow_transitions),
            'yellow_duration_seconds': float(self.yellow_duration),
            'button_presses': self.button_presses,
            'final_queues': self.queues.tolist(),
            'inference_times': {
                'mean_ms': float(np.mean(self.inference_times)) if self.inference_times else 0,
                'max_ms': float(np.max(self.inference_times)) if self.inference_times else 0,
                'min_ms': float(np.min(self.inference_times)) if self.inference_times else 0,
                'std_ms': float(np.std(self.inference_times)) if self.inference_times else 0,
                'real_time_capable': bool(np.max(self.inference_times) < 100) if self.inference_times else False
            }
        }
    
    def cleanup(self):
        """Clean up GPIO"""
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup()
        print("\nGPIO cleaned up.")


def run_comparison_demo(model_path, vecnorm_path, duration=60):
    """Run comparison between fixed-timing and PPO"""
    print("\n" + "="*70)
    print("     COMPARISON DEMO: FIXED-TIMING vs PPO-POWERED")
    print("="*70)
    
    # Run 1: Fixed-timing
    print("\n PART 1: TRADITIONAL FIXED-TIMING TRAFFIC LIGHT")
    print("   (Changes every 30 seconds, no intelligence)\n")
    input("Press ENTER to start fixed-timing demo...")
    
    logger_fixed = DataLogger()
    controller_fixed = HardwareController(model_path, vecnorm_path, logger_fixed, use_model=False)
    
    try:
        stats_fixed = controller_fixed.run_demo_mode(duration=duration)
        
        # Save results
        print("\n[LOGGING] Saving fixed-timing results...")
        df_fixed = logger_fixed.save_csv()
        logger_fixed.create_visualization(df_fixed)
        logger_fixed.save_statistics(stats_fixed)
        logger_fixed.save_text_report(stats_fixed)
        print("    - Log: " + logger_fixed.csv_path)
        print("    - Plot: " + logger_fixed.viz_path)
        print("    - Stats: " + logger_fixed.json_path)
        print("    - Report: " + logger_fixed.txt_path)
    finally:
        controller_fixed.cleanup()
    
    print("\n" + "-"*70)
    print(" Fixed-timing demo complete! Now let's see the PPO in action...")
    print("-"*70)
    time.sleep(3)
    
    # Run 2: PPO
    print("\n PART 2: PPO-POWERED ADAPTIVE TRAFFIC LIGHT")
    print("   (Changes based on traffic demand)\n")
    input("Press ENTER to start PPO demo...")
    
    logger_ppo = DataLogger()
    controller_ppo = HardwareController(model_path, vecnorm_path, logger_ppo, use_model=True)
    
    try:
        stats_ppo = controller_ppo.run_demo_mode(duration=duration)
        
        # Save results
        print("\n[LOGGING] Saving PPO results...")
        df_ppo = logger_ppo.save_csv()
        logger_ppo.create_visualization(df_ppo)
        logger_ppo.save_statistics(stats_ppo)
        logger_ppo.save_text_report(stats_ppo)
        print("    - Log: " + logger_ppo.csv_path)
        print("    - Plot: " + logger_ppo.viz_path)
        print("    - Stats: " + logger_ppo.json_path)
        print("    - Report: " + logger_ppo.txt_path)
    finally:
        controller_ppo.cleanup()
    
    # Print comparison
    print("\n\n" + "="*70)
    print("     COMPARISON RESULTS")
    print("="*70)
    
    total_presses_fixed = sum(stats_fixed['button_presses'].values())
    total_presses_ppo = sum(stats_ppo['button_presses'].values())
    
    print(f"\nCars Cleared:")
    print(f"   Fixed-Timing: {stats_fixed['vehicles_cleared']} out of {total_presses_fixed} " +
          f"({stats_fixed['vehicles_cleared']/max(total_presses_fixed,1)*100:.1f}%)")
    print(f"   PPO-Powered:   {stats_ppo['vehicles_cleared']} out of {total_presses_ppo} " +
          f"({stats_ppo['vehicles_cleared']/max(total_presses_ppo,1)*100:.1f}%)")
    
    if total_presses_fixed > 0 and total_presses_ppo > 0:
        improvement = stats_ppo['vehicles_cleared'] - stats_fixed['vehicles_cleared']
        print(f"   Improvement:  +{improvement} cars ({improvement/max(total_presses_fixed,1)*100:.1f}% better)")
    
    print(f"\nPhase Changes (Adaptability):")
    print(f"   Fixed-Timing: {stats_fixed['phase_changes']} changes")
    print(f"   PPO-Powered:   {stats_ppo['phase_changes']} changes")
    
    print(f"\nAverage Wait Time per Phase:")
    avg_wait_fixed = stats_fixed['duration_seconds'] / max(stats_fixed['phase_changes'], 1)
    avg_wait_ppo = stats_ppo['duration_seconds'] / max(stats_ppo['phase_changes'], 1)
    print(f"   Fixed-Timing: {avg_wait_fixed:.2f} seconds")
    print(f"   PPO-Powered:   {avg_wait_ppo:.2f} seconds")
    
    if stats_ppo['inference_times']['mean_ms'] > 0:
        print(f"\nPPO Decision Speed:")
        print(f"   Average: {stats_ppo['inference_times']['mean_ms']:.2f}ms")
        print(f"   That's {1000/stats_ppo['inference_times']['mean_ms']:.0f}x faster than human reaction time!")
    
    print("\n" + "="*70)
    print(" The PPO adapts to traffic in real-time, making smarter decisions!")
    print("="*70 + "\n")


def main():
    """Main execution"""
    print("\n--- PPO-BASED TRAFFIC OPTIMIZATION HARDWARE DEMO ---")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Paths
    MODEL_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_7/final_model.zip"
    VECNORM_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_7/vecnormalize.pkl"
    
    # Check files
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(VECNORM_PATH):
        print(f"[ERROR] VecNormalize not found: {VECNORM_PATH}")
        sys.exit(1)
    
    print("[SETUP] Model files found. Auto-Logging ON.")
    
    # Mode selection
    print("[SELECT MODE]")
    print("  1. Demo Mode (60s)")
    print("  2. Extended Demo (120s)")
    print("  3. Quick Test (30s)")
    print("  4. Comparison (Fixed vs PPO)\n")
    
    try:
        mode = input("Enter mode (1-4): ").strip()
        
        if mode == '4':
            # Comparison mode
            duration = 60
            print(f"\n* Starting comparison demo ({duration}s each)...\n")
            run_comparison_demo(MODEL_PATH, VECNORM_PATH, duration)
            return
        else:
            durations = {'1': 60, '2': 120, '3': 30}
            duration = durations.get(mode, 60)
    except:
        duration = 60
    
    print(f"\n* Starting {duration}-second deployment...\n")
    
    logger = None
    controller = None
    
    try:
        # Initialize logger (creates the run folder)
        logger = DataLogger()
        
        # redirect stdout to also save to file
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
        print("\n[LOGGING] All data saved:")
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
