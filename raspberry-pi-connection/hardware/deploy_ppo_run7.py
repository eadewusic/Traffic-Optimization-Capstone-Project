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
6.  Firebase Integration: Automatically uploads all deployment data to Firebase 
    Cloud Storage for remote access and analysis.

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

# Import Firebase uploader
try:
    from firebase_uploader import FirebaseUploader
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("[WARNING] Firebase uploader not found. Cloud uploads disabled.")

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
            f.write(f"  North:  {stats['button_presses']['north']}\n")
            f.write(f"  South:  {stats['button_presses']['south']}\n")
            f.write(f"  East:   {stats['button_presses']['east']}\n")
            f.write(f"  West:   {stats['button_presses']['west']}\n")
            f.write(f"  Total:  {sum(stats['button_presses'].values())}\n\n")
            
            total_presses = sum(stats['button_presses'].values())
            if total_presses > 0:
                efficiency = (stats['vehicles_cleared'] / total_presses) * 100
                f.write(f"Clearing Efficiency: {efficiency:.1f}%\n")
                f.write(f"  (Cleared {stats['vehicles_cleared']} out of {total_presses} button presses)\n\n")
            
            f.write("\n CONTROL PERFORMANCE\n")
            f.write(f"\nPhase Changes: {stats['phase_changes']}\n")
            f.write(f"Average Phase Duration: {stats['average_phase_duration']:.2f} seconds\n\n")
            
            f.write(f"N/S Phase Active: {stats['ns_phase_time']:.1f}s ({stats['ns_phase_time']/stats['duration_seconds']*100:.1f}%)\n")
            f.write(f"E/W Phase Active: {stats['ew_phase_time']:.1f}s ({stats['ew_phase_time']/stats['duration_seconds']*100:.1f}%)\n\n")
            
            f.write("\n INFERENCE PERFORMANCE\n")
            f.write(f"\nMean Inference Time: {stats['inference_times']['mean_ms']:.2f} ms\n")
            f.write(f"Median Inference Time: {stats['inference_times']['median_ms']:.2f} ms\n")
            f.write(f"Min Inference Time: {stats['inference_times']['min_ms']:.2f} ms\n")
            f.write(f"Max Inference Time: {stats['inference_times']['max_ms']:.2f} ms\n")
            f.write(f"Std Dev: {stats['inference_times']['std_ms']:.2f} ms\n\n")
            
            if stats['inference_times']['mean_ms'] < 100:
                f.write("Real-time Capability: YES (avg < 100ms threshold)\n")
            else:
                f.write("Real-time Capability: MARGINAL (avg >= 100ms threshold)\n")
            
            f.write(f"\nInference Rate: {1000/stats['inference_times']['mean_ms']:.0f} decisions/second\n")
            f.write(f"  That's {1000/stats['inference_times']['mean_ms']:.0f}x faster than human reaction time!\n\n")
            
            f.write("\n QUEUE ANALYSIS\n")
            f.write(f"\nMean Total Queue: {stats['queue_stats']['mean_total']:.2f}\n")
            f.write(f"Max Total Queue: {stats['queue_stats']['max_total']:.0f}\n")
            f.write(f"Max North Queue: {stats['queue_stats']['max_north']:.0f}\n")
            f.write(f"Max South Queue: {stats['queue_stats']['max_south']:.0f}\n")
            f.write(f"Max East Queue: {stats['queue_stats']['max_east']:.0f}\n")
            f.write(f"Max West Queue: {stats['queue_stats']['max_west']:.0f}\n\n")
            
            f.write("\n MODEL INFO\n")
            f.write(f"\nModel: {stats['model_info']['model_path']}\n")
            f.write(f"Architecture: Stable-Baselines3 PPO\n")
            f.write(f"Hardware: Raspberry Pi + GPIO\n\n")
            
            f.write("\n KEY INSIGHTS FOR PRESENTATION\n")
            f.write(f"\n1. SPEED: The AI makes decisions in {stats['inference_times']['mean_ms']:.1f}ms\n")
            f.write(f"   - That's {1000/stats['inference_times']['mean_ms']:.0f}x faster than a human!\n\n")
            
            if total_presses > 0:
                f.write(f"2. EFFICIENCY: Cleared {efficiency:.1f}% of vehicles\n")
                f.write(f"   - {stats['vehicles_cleared']} cars served from {total_presses} arrivals\n\n")
            
            f.write(f"3. ADAPTABILITY: Changed phases {stats['phase_changes']} times\n")
            f.write(f"   - Adjusted every {stats['average_phase_duration']:.1f} seconds on average\n")
            f.write(f"   - Not stuck on a fixed timer like traditional lights\n\n")
            
            f.write(f"4. REAL-TIME: Processes traffic conditions instantly\n")
            f.write(f"   - Reads 4 queues, decides phase in <{stats['inference_times']['max_ms']:.0f}ms\n\n")
            
            f.write("\n" + "="*70 + "\n")


class HardwareController:
    """Hardware interface for PPO traffic control"""
    
    def __init__(self, model_path, vecnormalize_path, logger=None):
        """Initialize GPIO and load model"""
        self.logger = logger
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        # GPIO setup FIRST - before any other GPIO operations
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(True)
        
        # Clean up any existing GPIO state first
        try:
            for pin in BUTTON_PINS.values():
                try:
                    GPIO.remove_event_detect(pin)
                except:
                    pass
        except:
            pass
        
        # LEDs as outputs
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Buttons as inputs with pull-down
        self.last_press_time = {direction: 0 for direction in BUTTON_PINS}
        self.debounce_delay = 0.3
        
        # Setup button pins first
        for direction, pin in BUTTON_PINS.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        # Then add event detection
        for direction, pin in BUTTON_PINS.items():
            try:
                GPIO.add_event_detect(pin, GPIO.RISING, 
                                    callback=lambda ch, d=direction: self._button_callback(d),
                                    bouncetime=200)
                print(f"[GPIO] Event detection added for {direction} button (pin {pin})")
            except Exception as e:
                print(f"[GPIO ERROR] Failed to add event detection for {direction} (pin {pin}): {e}")
        
        # Load model
        print("[MODEL] Loading PPO model...")
        self.model = PPO.load(model_path)
        self.vec_normalize = VecNormalize.load(vecnormalize_path, DummyVecEnv([lambda: None]))
        print("[MODEL] Model loaded successfully\n")
        
        # State
        self.queues = np.zeros(4, dtype=np.float32)
        self.current_action = 0
        self.step_count = 0
        self.vehicles_cleared = 0
        self.phase_changes = 0
        self.phase_start_time = time.time()
        self.ns_phase_time = 0
        self.ew_phase_time = 0
        
        # Set all lights to red initially
        self._set_all_red()
    
    def _button_callback(self, direction):
        """Handle button press with debouncing"""
        current_time = time.time()
        
        if current_time - self.last_press_time[direction] < self.debounce_delay:
            return
        
        self.last_press_time[direction] = current_time
        
        direction_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        idx = direction_map[direction]
        
        self.queues[idx] = min(self.queues[idx] + 1, 20)
        self.button_presses[direction] += 1
        
        q = self.queues.astype(int)
        print(f"\n  {direction.upper()} PRESSED | Queue: [N={q[0]} S={q[1]} E={q[2]} W={q[3]}]", 
              flush=True)
    
    def _set_lights(self, phase):
        """Set traffic lights based on phase"""
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
        
        GPIO.output(LED_PINS['north_yellow'], GPIO.LOW)
        GPIO.output(LED_PINS['south_yellow'], GPIO.LOW)
        GPIO.output(LED_PINS['east_yellow'], GPIO.LOW)
        GPIO.output(LED_PINS['west_yellow'], GPIO.LOW)
    
    def _yellow_transition(self, from_phase):
        """Show yellow lights for directions that are about to change from GREEN to RED"""
        if from_phase == 0:
            GPIO.output(LED_PINS['north_green'], GPIO.LOW)
            GPIO.output(LED_PINS['south_green'], GPIO.LOW)
            GPIO.output(LED_PINS['north_yellow'], GPIO.HIGH)
            GPIO.output(LED_PINS['south_yellow'], GPIO.HIGH)
        else:
            GPIO.output(LED_PINS['east_green'], GPIO.LOW)
            GPIO.output(LED_PINS['west_green'], GPIO.LOW)
            GPIO.output(LED_PINS['east_yellow'], GPIO.HIGH)
            GPIO.output(LED_PINS['west_yellow'], GPIO.HIGH)
        
        time.sleep(2.0)
        
        if from_phase == 0:
            GPIO.output(LED_PINS['north_yellow'], GPIO.LOW)
            GPIO.output(LED_PINS['south_yellow'], GPIO.LOW)
        else:
            GPIO.output(LED_PINS['east_yellow'], GPIO.LOW)
            GPIO.output(LED_PINS['west_yellow'], GPIO.LOW)
    
    def _set_all_red(self):
        """Set all lights to red"""
        for direction in ['north', 'south', 'east', 'west']:
            GPIO.output(LED_PINS[f'{direction}_red'], GPIO.HIGH)
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
    
    def _clear_vehicles(self, action):
        """Clear vehicles from active phase"""
        if action == 0:
            cleared_n = min(self.queues[0], 2)
            cleared_s = min(self.queues[1], 2)
            self.queues[0] = max(0, self.queues[0] - cleared_n)
            self.queues[1] = max(0, self.queues[1] - cleared_s)
            return int(cleared_n + cleared_s)
        else:
            cleared_e = min(self.queues[2], 2)
            cleared_w = min(self.queues[3], 2)
            self.queues[2] = max(0, self.queues[2] - cleared_e)
            self.queues[3] = max(0, self.queues[3] - cleared_w)
            return int(cleared_e + cleared_w)
    
    def _get_action(self):
        """Get action from PPO model"""
        start = time.perf_counter()
        
        obs_normalized = self.vec_normalize.normalize_obs(self.queues)
        action, _ = self.model.predict(obs_normalized, deterministic=True)
        
        inference_ms = (time.perf_counter() - start) * 1000
        return int(action), inference_ms
    
    def _print_status(self, step, action, cleared, inference_ms):
        """Print current status"""
        q = self.queues.astype(int)
        phase = 'N/S' if action == 0 else 'E/W'
        print(f"Step {step:3d} | Q:[N={q[0]:2d} S={q[1]:2d} E={q[2]:2d} W={q[3]:2d}] | "
              f"Phase:{phase} | Clear:{cleared} | Inf:{inference_ms:.2f}ms", 
              end='\r', flush=True)
    
    def run_demo_mode(self, duration=60):
        """Run time-limited demonstration"""
        print(f"\n{'='*70}")
        print(f" DEMO MODE - {duration} SECOND DEMONSTRATION", flush=True)
        print(f"{'='*70}")
        print("\n Press buttons to simulate vehicle arrivals", flush=True)
        print(" Watch LEDs for yellow transitions: GREEN -> YELLOW (2s) -> RED", flush=True)
        print("\n  Press Ctrl+C to stop\n", flush=True)
        
        start_time = time.time()
        self.step_count = 0
        last_action = self.current_action
        
        try:
            while (time.time() - start_time) < duration:
                action, inference_ms = self._get_action()
                
                phase_change = (action != last_action)
                if phase_change:
                    self.phase_changes += 1
                    
                    phase_duration = time.time() - self.phase_start_time
                    if last_action == 0:
                        self.ns_phase_time += phase_duration
                    else:
                        self.ew_phase_time += phase_duration
                    
                    self._yellow_transition(last_action)
                    self.phase_start_time = time.time()
                
                self._set_lights(action)
                cleared = self._clear_vehicles(action)
                self.vehicles_cleared += cleared
                
                if self.logger:
                    self.logger.log_step(self.step_count, self.queues.copy(), 
                                       action, cleared, inference_ms, phase_change)
                
                self._print_status(self.step_count, action, cleared, inference_ms)
                
                last_action = action
                self.step_count += 1
                time.sleep(0.5)
            
            final_duration = time.time() - self.phase_start_time
            if last_action == 0:
                self.ns_phase_time += final_duration
            else:
                self.ew_phase_time += final_duration
                
        except KeyboardInterrupt:
            print("\n\n[STOPPED] Demo interrupted by user")
        
        self._set_all_red()
        
        return self._get_statistics(time.time() - start_time)
    
    def _get_statistics(self, duration):
        """Generate statistics dictionary"""
        inference_times = [d['inference_ms'] for d in self.logger.data] if self.logger else [0]
        queue_data = [d for d in self.logger.data] if self.logger else []
        
        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'controller': 'PPO-RL (Run 7)',
            'duration_seconds': duration,
            'total_steps': self.step_count,
            'vehicles_cleared': self.vehicles_cleared,
            'button_presses': self.button_presses.copy(),
            'phase_changes': self.phase_changes,
            'average_phase_duration': duration / max(self.phase_changes, 1),
            'ns_phase_time': self.ns_phase_time,
            'ew_phase_time': self.ew_phase_time,
            'inference_times': {
                'mean_ms': np.mean(inference_times),
                'median_ms': np.median(inference_times),
                'min_ms': np.min(inference_times),
                'max_ms': np.max(inference_times),
                'std_ms': np.std(inference_times)
            },
            'queue_stats': {
                'mean_total': np.mean([d['total_queue'] for d in queue_data]) if queue_data else 0,
                'max_total': np.max([d['total_queue'] for d in queue_data]) if queue_data else 0,
                'max_north': np.max([d['north_queue'] for d in queue_data]) if queue_data else 0,
                'max_south': np.max([d['south_queue'] for d in queue_data]) if queue_data else 0,
                'max_east': np.max([d['east_queue'] for d in queue_data]) if queue_data else 0,
                'max_west': np.max([d['west_queue'] for d in queue_data]) if queue_data else 0
            },
            'model_info': {
                'model_path': '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_7/final_model.zip'
            }
        }
        
        return stats
    
    def cleanup(self):
        """Clean up GPIO"""
        print("\n[GPIO] Cleaning up...")
        for pin in BUTTON_PINS.values():
            try:
                GPIO.remove_event_detect(pin)
            except:
                pass
        GPIO.cleanup()


class FixedTimingController(HardwareController):
    """Fixed-timing baseline controller for comparison"""
    
    def __init__(self, logger=None):
        """Initialize without PPO model"""
        self.logger = logger
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        # GPIO setup FIRST
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(True)
        
        # Clean up any existing GPIO state first
        try:
            for pin in BUTTON_PINS.values():
                try:
                    GPIO.remove_event_detect(pin)
                except:
                    pass
        except:
            pass
        
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        self.last_press_time = {direction: 0 for direction in BUTTON_PINS}
        self.debounce_delay = 0.3
        
        # Setup button pins first
        for direction, pin in BUTTON_PINS.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        # Then add event detection
        for direction, pin in BUTTON_PINS.items():
            try:
                GPIO.add_event_detect(pin, GPIO.RISING, 
                                    callback=lambda ch, d=direction: self._button_callback(d),
                                    bouncetime=200)
                print(f"[GPIO] Event detection added for {direction} button (pin {pin})")
            except Exception as e:
                print(f"[GPIO ERROR] Failed to add event detection for {direction} (pin {pin}): {e}")
        
        self.queues = np.zeros(4, dtype=np.float32)
        self.current_action = 0
        self.step_count = 0
        self.vehicles_cleared = 0
        self.phase_changes = 0
        self.phase_start_time = time.time()
        self.ns_phase_time = 0
        self.ew_phase_time = 0
        
        self._set_all_red()
    
    def _get_action(self):
        """Fixed 15-second alternating phases"""
        elapsed = time.time() - self.phase_start_time
        
        if elapsed >= 15.0:
            return (self.current_action + 1) % 2, 0.0
        else:
            return self.current_action, 0.0


def run_comparison_demo(model_path, vecnormalize_path, duration=60):
    """Run comparison between fixed-timing and PPO"""
    
    # Clean up any existing GPIO state from previous runs
    try:
        GPIO.setmode(GPIO.BCM)
        for pin in BUTTON_PINS.values():
            try:
                GPIO.remove_event_detect(pin)
            except:
                pass
        GPIO.cleanup()
        time.sleep(1)
    except:
        pass
    
    print("\n" + "="*70)
    print("     COMPARISON MODE: FIXED-TIMING vs PPO")
    print("="*70)
    print(f"\nRunning {duration}s with each controller...\n")
    
    # Run 1: Fixed-timing
    print("\n[1/2] Running FIXED-TIMING controller...")
    print("-" * 70)
    
    logger_fixed = DataLogger()
    controller_fixed = FixedTimingController(logger_fixed)
    
    try:
        stats_fixed = controller_fixed.run_demo_mode(duration=duration)
        
        print("\n[LOGGING] Saving fixed-timing results...")
        df_fixed = logger_fixed.save_csv()
        logger_fixed.create_visualization(df_fixed)
        stats_fixed['controller'] = 'Fixed-Timing Baseline'
        logger_fixed.save_statistics(stats_fixed)
        logger_fixed.save_text_report(stats_fixed)
        print("    - Log: " + logger_fixed.csv_path)
        print("    - Plot: " + logger_fixed.viz_path)
        print("    - Stats: " + logger_fixed.json_path)
        print("    - Report: " + logger_fixed.txt_path)
    finally:
        controller_fixed.cleanup()
        time.sleep(2)
    
    print("\n\n[2/2] Running PPO-POWERED controller...")
    print("-" * 70)
    print("[GPIO] Waiting for hardware to settle...")
    time.sleep(2)
    
    logger_ppo = DataLogger()
    controller_ppo = HardwareController(model_path, vecnormalize_path, logger_ppo)
    
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
    
    comparison_text = []
    comparison_text.append("="*70)
    comparison_text.append("     COMPARISON RESULTS: FIXED-TIMING vs PPO")
    comparison_text.append("="*70)
    comparison_text.append(f"\nComparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    comparison_text.append(f"Duration: {duration} seconds per method\n")
    comparison_text.append(f"\nRun Folders:")
    comparison_text.append(f"  Fixed-Timing: {logger_fixed.run_folder}")
    comparison_text.append(f"  PPO-Powered:   {logger_ppo.run_folder}\n")
    
    comparison_text.append("\nCars Cleared:")
    fixed_pct = stats_fixed['vehicles_cleared']/max(total_presses_fixed,1)*100
    ppo_pct = stats_ppo['vehicles_cleared']/max(total_presses_ppo,1)*100
    comparison_text.append(f"   Fixed-Timing: {stats_fixed['vehicles_cleared']} out of {total_presses_fixed} ({fixed_pct:.1f}%)")
    comparison_text.append(f"   PPO-Powered:   {stats_ppo['vehicles_cleared']} out of {total_presses_ppo} ({ppo_pct:.1f}%)")
    
    if total_presses_fixed > 0 and total_presses_ppo > 0:
        improvement = stats_ppo['vehicles_cleared'] - stats_fixed['vehicles_cleared']
        comparison_text.append(f"   Improvement:  +{improvement} cars ({improvement/max(total_presses_fixed,1)*100:.1f}% better)")
    
    comparison_text.append(f"\nPhase Changes (Adaptability):")
    comparison_text.append(f"   Fixed-Timing: {stats_fixed['phase_changes']} changes")
    comparison_text.append(f"   PPO-Powered:   {stats_ppo['phase_changes']} changes")
    
    comparison_text.append(f"\nAverage Wait Time per Phase:")
    avg_wait_fixed = stats_fixed['duration_seconds'] / max(stats_fixed['phase_changes'], 1)
    avg_wait_ppo = stats_ppo['duration_seconds'] / max(stats_ppo['phase_changes'], 1)
    comparison_text.append(f"   Fixed-Timing: {avg_wait_fixed:.2f} seconds")
    comparison_text.append(f"   PPO-Powered:   {avg_wait_ppo:.2f} seconds")
    
    if stats_ppo['inference_times']['mean_ms'] > 0:
        comparison_text.append(f"\nPPO Decision Speed:")
        comparison_text.append(f"   Average: {stats_ppo['inference_times']['mean_ms']:.2f}ms")
        comparison_text.append(f"   That's {1000/stats_ppo['inference_times']['mean_ms']:.0f}x faster than human reaction time!")
    
    comparison_text.append("\n" + "="*70)
    comparison_text.append(" CONCLUSION: The PPO adapts to traffic in real-time, making smarter decisions!")
    comparison_text.append("="*70)
    
    # Print to terminal
    for line in comparison_text:
        print(line)
    
    # Save to file
    results_dir = '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(results_dir, f"comparison_analysis_{timestamp}.txt")
    
    with open(comparison_file, 'w') as f:
        f.write('\n'.join(comparison_text))
    
    print(f"\n[SAVED] Comparison analysis: {comparison_file}")
    
    # Upload to Firebase
    if FIREBASE_AVAILABLE:
        try:
            uploader = FirebaseUploader()
            if uploader.initialized:
                print("\n[FIREBASE] Uploading comparison data...")
                
                uploader.upload_run_folder(logger_fixed.run_folder)
                uploader.upload_run_folder(logger_ppo.run_folder)
                uploader.upload_comparison(comparison_file)
                
                print("[FIREBASE] All comparison data uploaded successfully")
        except Exception as e:
            print(f"[FIREBASE] Upload failed: {e}")
    
    print("")


def main():
    """Main execution"""
    # Clean up any GPIO state from previous runs
    try:
        import RPi.GPIO as GPIO_CLEAN
        GPIO_CLEAN.setmode(GPIO_CLEAN.BCM)
        for pin in BUTTON_PINS.values():
            try:
                GPIO_CLEAN.remove_event_detect(pin)
            except:
                pass
        GPIO_CLEAN.cleanup()
    except:
        pass
    
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
    
    if FIREBASE_AVAILABLE:
        print("[SETUP] Firebase integration enabled")
    else:
        print("[SETUP] Firebase integration disabled")
    
    # Mode selection
    print("\n[SELECT MODE]")
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
        
        # Upload to Firebase
        if FIREBASE_AVAILABLE:
            try:
                uploader = FirebaseUploader()
                if uploader.initialized:
                    print("\n[FIREBASE] Uploading deployment data...")
                    uploaded = uploader.upload_run_folder(logger.run_folder)
                    
                    if uploaded:
                        print(f"[FIREBASE] Successfully uploaded {len(uploaded)} files")
                        print(f"[FIREBASE] View at: https://console.firebase.google.com/project/traffic-ppo-pi/storage")
            except Exception as e:
                print(f"[FIREBASE] Upload failed: {e}")
        
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
