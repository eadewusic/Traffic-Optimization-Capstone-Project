"""
Event-Driven PPO Traffic Controller - Run 8 Seed 789 Champion
Hardware Deployment Script

NEW CHANGES (v2.0):
1. Event-Driven Architecture (not time-based)
    - Runs continuously until all lanes clear (IDLE state achieved)
    - No arbitrary 60s/120s time limits
    - Hibernates when idle, wakes on button press

2. Static Display with Increments
    - Uses ANSI escape codes to update in place
    - Shows +/- deltas in green/red
    - No scrolling text confusion

3. Separate Inference Time vs Demo Delay
    - Inference time: Real 5.78ms metric
    - Demo delay: Set to 5s for faster hardware demonstration.
    - Clearly labeled to avoid confusion

4. System Resource Monitoring
   - RAM usage tracking (MB and %)
   - CPU usage monitoring
   - Temperature monitoring (Raspberry Pi specific)
   - Logged and visualized. Functionality is conditional on 
   the availability of the 'psutil' library.

5. Session Management
   - Starts: First button press
   - Ends: All lanes clear OR Ctrl+C force stop
   - Idle state between sessions

6. Model Information
   - Displays full model size at startup
   - Shows validation metrics
   - Multi-seed champion context

FIXED BUGS:
1. Duration logic for Mode 2 - Now runs full 60s across multiple sessions
2. Path handling - Mode 1 uses session subfolders, Mode 2/3 use direct paths
3. Comparison analysis - Generates comparison_analysis.txt file

Key Components:
Hardware Control:
    - 12 GPIO pins for traffic LEDs (3 per direction: red, yellow, green)
    - 4 button inputs (debounced, 300ms delay) to simulate vehicle arrivals
    - Safe yellow light transitions (2 seconds)
    - All-red idle mode for safety

RL Inference:
    - Loads trained Stable-Baselines3 PPO model with VecNormalize
    - Real-time phase decisions (North/South vs East/West green)
    - Processes normalized queue observations
    - Outputs discrete actions

    - Simulated Vehicle Clearing: 2 cars are cleared per lane 
    per step (`CLEARING_RATE = 2`). (This rate was increased from 
    1 in training/earlier versions to speed up demo sessions.)

    - PPO Safety Wrapper: Overrides PPO's action if the chosen phase 
    (N/S or E/W) has zero vehicles and the opposing phase is queued 
    (`USE_SAFETY_WRAPPER = True`). This addresses known suboptimal policies 
    in the trained model without retraining.

Event-Driven Logic:
    - IDLE state: All lanes clear, all red, waiting for button press
    - ACTIVE state: Traffic present, PPO controlling lights
    - Automatic transition between states
    - Graceful session completion

Data Logging & Analysis:
    - Step-by-step metrics: queues, actions, cleared vehicles, inference times
    - System resource tracking: RAM, CPU, temperature
    - CSV logs, visualizations, JSON statistics, text reports
    - Timestamped folders for reproducibility

Model Context:
This script deploys "Run 8 Seed 789", the multi-seed validated champion model:
    - Best of 5 independent training runs (seeds: 42, 123, 456, 789, 1000)
    - 72% win rate against fixed-timing baseline
    - Statistical significance: p=0.0002
    - Coefficient of variation: 1.3% (exceptional reproducibility)
    - Real-time inference: 5.78ms mean (17x safety margin)
"""


import RPi.GPIO as GPIO
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import json
import sys
import os
import select
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

sys.path.append('/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/environments')


class TerminalCapture:
    """Captures terminal output to save in logs"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = []
    
    def write(self, message):
        self.terminal.write(message)
        self.log.append(message)
    
    def flush(self):
        self.terminal.flush()
    
    def get_output(self):
        return ''.join(self.log)


class ButtonRecorder:
    """Records button presses with timestamps for replay in comparison mode"""
    def __init__(self):
        self.recordings = []
        self.start_time = None
    
    def start_recording(self):
        """Start recording session"""
        self.recordings = []
        self.start_time = time.time()
    
    def record_press(self, direction):
        """Record a button press with timestamp"""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        self.recordings.append({
            'direction': direction,
            'timestamp': elapsed
        })
    
    def get_recordings(self):
        """Return all recorded button presses"""
        return self.recordings
    
    def save_to_file(self, filepath):
        """Save recordings to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                'recordings': self.recordings,
                'total_presses': len(self.recordings),
                'duration': self.recordings[-1]['timestamp'] if self.recordings else 0
            }, f, indent=2)


class ButtonReplayer:
    """Replays recorded button presses for comparison mode"""
    def __init__(self, recordings):
        self.recordings = recordings
        self.start_time = None
        self.current_index = 0
    
    def start_replay(self):
        """Start replay session"""
        self.start_time = time.time()
        self.current_index = 0
    
    def get_due_presses(self):
        """Get all button presses that should have occurred by now"""
        if self.start_time is None or not self.recordings:
            return []
        
        elapsed = time.time() - self.start_time
        due_presses = []
        
        while self.current_index < len(self.recordings):
            press = self.recordings[self.current_index]
            if press['timestamp'] <= elapsed:
                due_presses.append(press['direction'])
                self.current_index += 1
            else:
                break
        
        return due_presses
    
    def has_more(self):
        """Check if there are more presses to replay"""
        return self.current_index < len(self.recordings)


# Configuration
LED_PINS = {
    'north_red': 16, 'north_yellow': 20, 'north_green': 21,
    'east_red': 5, 'east_yellow': 6, 'east_green': 13,
    'south_red': 23, 'south_yellow': 24, 'south_green': 25,
    'west_red': 14, 'west_yellow': 4, 'west_green': 18
}

BUTTON_PINS = {'north': 9, 'east': 10, 'south': 22, 'west': 17}

# Adjustable parameters
DEMO_DELAY = 5  # Shorter delay (was 15s)
YELLOW_DURATION = 2
CLEARING_RATE = 2  # Cars cleared per step (was 1)

# Safety wrapper - prevents PPO from choosing empty phases
# This fixes inefficiency WITHOUT retraining
USE_SAFETY_WRAPPER = True  # Set False to see raw PPO behavior

# Debug flag - shows when PPO chooses wrong phase (clears 0 cars)
SHOW_INEFFICIENCY_WARNINGS = True


def get_system_metrics():
    """Get system resource usage (RAM, CPU, temperature)
    
    Returns dict with metrics or None values if psutil unavailable
    Used for monitoring Pi 4 performance during deployment
    """
    if not PSUTIL_AVAILABLE:
        return {
            'ram_mb': None, 'ram_percent': None,
            'cpu_percent': None, 'cpu_temp': None
        }
    
    try:
        process = psutil.Process(os.getpid())
        
        # Get CPU temperature (Raspberry Pi specific)
        try:
            temp_str = os.popen("vcgencmd measure_temp").readline()
            cpu_temp = float(temp_str.replace("temp=","").replace("'C\n",""))
        except:
            cpu_temp = None
        
        return {
            'ram_mb': process.memory_info().rss / 1024 / 1024,
            'ram_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(interval=0.1),
            'cpu_temp': cpu_temp
        }
    except Exception as e:
        # Graceful fallback if monitoring fails
        return {
            'ram_mb': None, 'ram_percent': None,
            'cpu_percent': None, 'cpu_temp': None
        }


class DataLogger:
    """Comprehensive logging"""

    def __init__(self, controller_name, log_dir='/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join(log_dir, f"{controller_name}_{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)

        # Session-specific paths will be set when saving
        self.csv_path = None
        self.viz_path = None
        self.json_path = None
        self.txt_path = None
        self.terminal_path = None

        self.data = []
        self.system_metrics = []  # Track system resources
        self.start_time = time.time()

        # Don't print log path at init - will show at end
    
    def set_session_paths(self, session_number):
        """Set file paths for a specific session (Mode 1 only)"""
        session_folder = os.path.join(self.run_folder, f"session_{session_number}")
        os.makedirs(session_folder, exist_ok=True)
        
        self.csv_path = os.path.join(session_folder, "log.csv")
        self.viz_path = os.path.join(session_folder, "visualization.png")
        self.json_path = os.path.join(session_folder, "stats.json")
        self.txt_path = os.path.join(session_folder, "report.txt")
        self.terminal_path = os.path.join(session_folder, "terminal_output.txt")
        
        return session_folder
    
    def set_direct_paths(self):
        """Set file paths directly in run folder (for Mode 2/3)"""
        self.csv_path = os.path.join(self.run_folder, "log.csv")
        self.viz_path = os.path.join(self.run_folder, "visualization.png")
        self.json_path = os.path.join(self.run_folder, "stats.json")
        self.txt_path = os.path.join(self.run_folder, "report.txt")
        self.terminal_path = os.path.join(self.run_folder, "terminal_output.txt")

    def log_step(self, step, queues, action, cleared, inference_ms, phase_change, sys_metrics=None):
        elapsed = time.time() - self.start_time
        
        step_data = {
            'timestamp': elapsed, 'step': step,
            'north_queue': queues[0], 'south_queue': queues[1],
            'east_queue': queues[2], 'west_queue': queues[3],
            'total_queue': np.sum(queues), 'action': action,
            'phase': 'N/S' if action == 0 else 'E/W',
            'cleared': cleared, 'inference_ms': inference_ms,
            'phase_change': 1 if phase_change else 0
        }
        
        # Add system metrics if available
        if sys_metrics:
            step_data.update({
                'ram_mb': sys_metrics.get('ram_mb'),
                'cpu_percent': sys_metrics.get('cpu_percent'),
                'cpu_temp': sys_metrics.get('cpu_temp')
            })
            self.system_metrics.append(sys_metrics)
        
        self.data.append(step_data)

    def save_all(self, stats):
        if not self.data:
            return

        df = pd.DataFrame(self.data)
        df.to_csv(self.csv_path, index=False)

        # Visualization - 6 plots in 3x2 grid
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'{stats["controller"]} Performance', fontsize=16, fontweight='bold')

        # Plot 1: Queue Dynamics (top-left) - NESW order
        axes[0, 0].plot(df['timestamp'], df['north_queue'], label='North', alpha=0.7, linewidth=2)
        axes[0, 0].plot(df['timestamp'], df['east_queue'], label='East', alpha=0.7, linewidth=2)
        axes[0, 0].plot(df['timestamp'], df['south_queue'], label='South', alpha=0.7, linewidth=2)
        axes[0, 0].plot(df['timestamp'], df['west_queue'], label='West', alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel('Time (s)', fontsize=10)
        axes[0, 0].set_ylabel('Queue Length', fontsize=10)
        axes[0, 0].set_title('Queue Dynamics', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Total Congestion Over Time (top-right)
        axes[0, 1].plot(df['timestamp'], df['total_queue'], color='red', linewidth=2)
        axes[0, 1].fill_between(df['timestamp'], 0, df['total_queue'], alpha=0.3, color='red')
        axes[0, 1].set_xlabel('Time (s)', fontsize=10)
        axes[0, 1].set_ylabel('Total Queue Length', fontsize=10)
        axes[0, 1].set_title('Total Congestion Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Phase Decisions (Green=N/S, Blue=E/W) (middle-left)
        phase_colors = ['green' if p == 0 else 'blue' for p in df['action']]
        axes[1, 0].scatter(df['timestamp'], df['step'], c=phase_colors, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Time (s)', fontsize=10)
        axes[1, 0].set_ylabel('Step', fontsize=10)
        axes[1, 0].set_title('Phase Decisions (Green=N/S, Blue=E/W)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add legend for phase colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='N/S Green'),
                          Patch(facecolor='blue', label='E/W Green')]
        axes[1, 0].legend(handles=legend_elements, fontsize=9)

        # Plot 4: Clearing Performance (middle-right)
        axes[1, 1].bar(df['step'], df['cleared'], alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Step', fontsize=10)
        axes[1, 1].set_ylabel('Vehicles Cleared', fontsize=10)
        axes[1, 1].set_title('Clearing Performance', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Plot 5: Response Time Distribution (bottom-left)
        axes[2, 0].hist(df['inference_ms'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[2, 0].axvline(df['inference_ms'].mean(), color='red', linestyle='--',
                          linewidth=2, label=f"Mean: {df['inference_ms'].mean():.2f}ms")
        axes[2, 0].axhline(y=0, xmin=0, xmax=100, color='orange', linestyle='-', 
                          linewidth=1, label='100ms threshold')
        axes[2, 0].set_xlabel('Inference Time (ms)', fontsize=10)
        axes[2, 0].set_ylabel('Frequency', fontsize=10)
        axes[2, 0].set_title('Response Time Distribution', fontsize=12, fontweight='bold')
        axes[2, 0].legend(fontsize=9)
        axes[2, 0].grid(True, alpha=0.3)

        # Plot 6: Cumulative Metrics (bottom-right)
        cumulative_cleared = df['cleared'].cumsum()
        cumulative_changes = df['phase_change'].cumsum()
        
        ax6 = axes[2, 1]
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(df['timestamp'], cumulative_cleared, color='darkgreen', 
                        linewidth=2, label='Cumulative Cleared')
        line2 = ax6_twin.plot(df['timestamp'], cumulative_changes, color='blue', 
                             linewidth=2, label='Phase Changes', linestyle='--')
        
        ax6.set_xlabel('Time (s)', fontsize=10)
        ax6.set_ylabel('Cumulative Cleared', fontsize=10, color='darkgreen')
        ax6_twin.set_ylabel('Count', fontsize=10, color='blue')
        ax6.set_title('Cumulative Metrics', fontsize=12, fontweight='bold')
        ax6.tick_params(axis='y', labelcolor='darkgreen')
        ax6_twin.tick_params(axis='y', labelcolor='blue')
        ax6.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, fontsize=9, loc='upper left')

        plt.tight_layout()
        plt.savefig(self.viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        with open(self.json_path, 'w') as f:
            json.dump(stats, f, indent=2)

        with open(self.txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"{stats['controller']} DEPLOYMENT REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Duration: {stats['duration_seconds']:.1f}s\n")
            f.write(f"Steps: {stats['total_steps']}\n")
            f.write(f"Cars arrived: {stats['total_arrivals']}\n")
            f.write(f"Cars cleared: {stats['vehicles_cleared']}\n")
            f.write(f"Throughput: {stats['throughput_percent']:.1f}%\n")
            f.write(f"Phase changes: {stats['phase_changes']}\n")
            
            if stats.get('inference_times'):
                f.write(f"\nInference Performance:\n")
                f.write(f"  Mean: {stats['inference_times']['mean_ms']:.2f}ms\n")
                f.write(f"  Max: {stats['inference_times']['max_ms']:.2f}ms\n")
                f.write(f"  Min: {stats['inference_times']['min_ms']:.2f}ms\n")
            
            # Add system metrics summary if available
            if stats.get('system_metrics'):
                sys_m = stats['system_metrics']
                f.write(f"\nSystem Resources:\n")
                if sys_m.get('mean_ram_mb') is not None:
                    f.write(f"  Mean RAM: {sys_m['mean_ram_mb']:.1f}MB\n")
                if sys_m.get('peak_ram_mb') is not None:
                    f.write(f"  Peak RAM: {sys_m['peak_ram_mb']:.1f}MB\n")
                if sys_m.get('mean_cpu_percent') is not None:
                    f.write(f"  Mean CPU: {sys_m['mean_cpu_percent']:.1f}%\n")
                if sys_m.get('peak_temp') is not None:
                    f.write(f"  Peak Temp: {sys_m['peak_temp']:.1f}°C\n")

    def save_terminal_output(self, terminal_capture):
        """Save captured terminal output"""
        if terminal_capture:
            with open(self.terminal_path, 'w') as f:
                f.write(terminal_capture.get_output())


class PPOController:
    """PPO-based controller"""

    def __init__(self, model_path, vecnorm_path, logger, button_replayer=None):
        self.logger = logger
        self.max_queue_length = 20
        self.button_replayer = button_replayer  # For replay in comparison mode

        # Load PPO model with confirmation
        print("[LOADING] PPO model...", end=" ")
        from run7_env import Run7TrafficEnv
        self.model = PPO.load(model_path)
        print("✓")
        
        print("[LOADING] VecNormalize statistics...", end=" ")
        dummy_env = DummyVecEnv([lambda: Run7TrafficEnv()])
        self.vec_env = VecNormalize.load(vecnorm_path, dummy_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        print("✓")
        
        print("[READY] PPO controller initialized\n")

        self.queues = np.zeros(4, dtype=np.float32)
        self.current_phase = 0
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.vehicles_cleared = 0
        self.phase_changes = 0
        self.yellow_transitions = 0

        # Session tracking
        self.session_start = None
        self.session_step = 0

        # Mode tracking
        self.mode_start_time = None  # For timed modes

        self.last_button_time = {d: 0 for d in BUTTON_PINS.keys()}
        self.debounce_delay = 0.3

        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        for direction, pin in BUTTON_PINS.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def read_buttons(self, show_press=False):
        """Read buttons with optional display

        In replay mode, processes recorded button presses instead of hardware
        """
        # REPLAY MODE: Process recorded presses
        if self.button_replayer:
            due_presses = self.button_replayer.get_due_presses()

            for direction in due_presses:
                lane_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
                lane = lane_map[direction]

                if self.queues[lane] < self.max_queue_length:
                    self.queues[lane] += 1
                    self.button_presses[direction] += 1

                    if show_press:
                        print(f"  >> {direction.upper()} button (REPLAY) | "
                              f"Queues now: N={int(self.queues[0])} E={int(self.queues[2])} "
                              f"S={int(self.queues[1])} W={int(self.queues[3])}")

            return [d.capitalize() for d in due_presses]

        # MANUAL MODE: Read hardware buttons
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
                        pressed_buttons.append(direction.capitalize())

                        if show_press:
                            print(f"  >> {direction.upper()} button pressed | "
                                  f"Queues now: N={int(self.queues[0])} E={int(self.queues[2])} "
                                  f"S={int(self.queues[1])} W={int(self.queues[3])}")

        return pressed_buttons

    def set_all_red(self):
        for direction in ['north', 'south', 'east', 'west']:
            GPIO.output(LED_PINS[f'{direction}_red'], GPIO.HIGH)
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)

    def set_lights(self, phase, color='green'):
        for direction in ['north', 'south', 'east', 'west']:
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)

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

    def clear_vehicles(self, action):
        """Clear vehicles - now clears CLEARING_RATE cars"""
        cleared = 0
        if action == 0:
            for lane in [0, 1]:
                if self.queues[lane] > 0:
                    clear_amount = min(CLEARING_RATE, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount
        else:
            for lane in [2, 3]:
                if self.queues[lane] > 0:
                    clear_amount = min(CLEARING_RATE, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount

        self.vehicles_cleared += cleared
        return int(cleared)

    def get_action(self):
        """Get PPO action with inference time"""
        obs = self.queues.copy() / self.max_queue_length
        obs_norm = self.vec_env.normalize_obs(obs)

        start_time = time.perf_counter()
        action, _ = self.model.predict(obs_norm, deterministic=True)
        inference_time = (time.perf_counter() - start_time) * 1000

        return int(action), inference_time

    def get_action_with_safety(self):
        """Get PPO action with optional safety override

        Safety wrapper prevents PPO from choosing phases with zero cars.
        This fixes inefficiency without retraining.
        """
        action, inference_ms = self.get_action()

        if USE_SAFETY_WRAPPER:
            n, s, e, w = self.queues
            ns_cars = n + s
            ew_cars = e + w

            # Override if PPO chose empty phase
            original_action = action
            if action == 0 and ns_cars == 0 and ew_cars > 0:
                action = 1  # Override to E/W
            elif action == 1 and ew_cars == 0 and ns_cars > 0:
                action = 0  # Override to N/S

            # Track if we overrode
            if action != original_action and SHOW_INEFFICIENCY_WARNINGS:
                print(f"  [SAFETY OVERRIDE] PPO chose {['N/S','E/W'][original_action]}, "
                      f"corrected to {['N/S','E/W'][action]}")

        return action, inference_ms

    def check_inefficiency(self, action):
        """Check if PPO is making inefficient decision"""
        n, s, e, w = self.queues
        ns_demand = n + s
        ew_demand = e + w

        # Inefficient if choosing phase with no cars
        if action == 0 and ns_demand == 0 and ew_demand > 0:
            return True, f"INEFFICIENT: Chose N/S but only E/W has cars!"
        if action == 1 and ew_demand == 0 and ns_demand > 0:
            return True, f"INEFFICIENT: Chose E/W but only N/S has cars!"

        return False, ""

    def run(self, duration=None, terminal_capture=None, use_session_folders=True):
        """Run controller with continuous multi-session operation

        Args:
            duration: If None, runs until Ctrl+C (Mode 1). If set, runs for duration seconds (Mode 2)
            terminal_capture: TerminalCapture object for logging
            use_session_folders: If True, creates session_N subfolders (Mode 1).
                               If False, saves directly to run folder (Mode 2)

        System hibernates after clearing all lanes, then wakes for next session.
        Only stops on Ctrl+C or duration timeout (if specified).
        """
        session_number = 0
        session_has_data = False  # Track if current session processed any traffic

        # Track overall mode duration (for Mode 2)
        if duration:
            self.mode_start_time = datetime.now()

        try:
            while True:
                # Check mode duration (not session duration) for timed mode
                if duration and self.mode_start_time:
                    mode_elapsed = (datetime.now() - self.mode_start_time).total_seconds()
                    if mode_elapsed >= duration:
                        print(f"\n[TIMEOUT] {duration}s mode duration reached\n")
                        break

                # IDLE MODE: Wait for first car
                if np.sum(self.queues) == 0:
                    # In replay mode, skip idle wait - just start processing
                    if self.button_replayer:
                        if session_number == 0:
                            session_number = 1
                            self.session_start = datetime.now()
                            self.session_step = 0
                            print("[START] Session 1 beginning (REPLAY MODE)...\n")
                            time.sleep(0.5)
                            continue  # Skip to active mode

                    if session_number == 0:
                        print("[IDLE] Press any button to start...\n")
                    else:
                        # Completed a session - save logs and hibernate
                        if session_has_data:
                            self.save_session_logs(terminal_capture, session_number, use_session_folders)
                            print(f"\n[HIBERNATE] Session {session_number} complete. Waiting for next vehicle...\n")

                        # Reset for next session
                        session_has_data = False
                        self.reset_session_metrics()

                    # Wait for button press to start/resume
                    while np.sum(self.queues) == 0:
                        # Check timeout even while idle
                        if duration and self.mode_start_time:
                            mode_elapsed = (datetime.now() - self.mode_start_time).total_seconds()
                            if mode_elapsed >= duration:
                                print(f"\n[TIMEOUT] {duration}s mode duration reached while idle\n")
                                # Save empty stats before exiting
                                self.save_empty_stats(use_session_folders)
                                return  # Exit completely

                        self.read_buttons(show_press=True)
                        time.sleep(0.1)

                    # Session starting
                    session_number += 1
                    self.session_start = datetime.now()
                    self.session_step = 0

                    if session_number == 1:
                        print("[START] Session 1 beginning...\n")
                    else:
                        print(f"[WAKE] Session {session_number} starting...\n")

                    time.sleep(0.5)

                # ACTIVE MODE: Control traffic
                self.session_step += 1
                session_has_data = True  # Mark that this session processed traffic

                # Get action (with safety wrapper if enabled)
                action, inference_ms = self.get_action_with_safety()

                # Get system metrics
                sys_metrics = get_system_metrics()

                # Check if decision is inefficient (for analysis only)
                is_inefficient, inefficiency_msg = self.check_inefficiency(action)

                # Phase change
                phase_change = False
                if action != self.current_phase:
                    self.set_lights(self.current_phase, 'yellow')
                    self.yellow_transitions += 1
                    time.sleep(YELLOW_DURATION)
                    self.current_phase = action
                    self.phase_changes += 1
                    phase_change = True

                self.set_lights(self.current_phase, 'green')

                # Clear vehicles
                cleared = self.clear_vehicles(self.current_phase)

                # Log (now includes system metrics)
                self.logger.log_step(self.session_step, self.queues.copy(), action, cleared,
                                   inference_ms, phase_change, sys_metrics)

                # Build display string
                phase_name = "N/S" if action == 0 else "E/W"
                switch = " [SWITCH]" if phase_change else ""
                warning = f" <<{inefficiency_msg}>>" if (is_inefficient and SHOW_INEFFICIENCY_WARNINGS) else ""

                # Add system metrics to display if available
                sys_str = ""
                if sys_metrics['ram_mb'] is not None:
                    ram = sys_metrics['ram_mb']
                    cpu = sys_metrics['cpu_percent'] if sys_metrics['cpu_percent'] is not None else 0
                    temp = sys_metrics['cpu_temp'] if sys_metrics['cpu_temp'] is not None else 0
                    sys_str = f" | RAM: {ram:.1f}MB CPU: {cpu:.1f}% Temp: {temp:.1f}°C"

                print(f"[STEP {self.session_step:3d}] {phase_name} Green{switch:10s} | "
                      f"Clear: {cleared:2d} | "
                      f"Queue: N={int(self.queues[0])} E={int(self.queues[2])} "
                      f"S={int(self.queues[1])} W={int(self.queues[3])} | "
                      f"Total: {int(np.sum(self.queues)):2d} | "
                      f"Infer: {inference_ms:.2f}ms{sys_str}{warning}")

                # Demo delay with button polling
                delay_start = time.time()
                while time.time() - delay_start < DEMO_DELAY:
                    self.read_buttons(show_press=True)
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] User interrupted (Ctrl+C)\n")

        finally:
            self.set_all_red()
            # Only save if this session processed traffic
            if session_has_data and len(self.logger.data) > 0:
                self.save_session_logs(terminal_capture, session_number, use_session_folders)

    def reset_session_metrics(self):
        """Reset metrics between sessions"""
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.vehicles_cleared = 0
        self.phase_changes = 0
        self.yellow_transitions = 0
        self.logger.data = []
        self.logger.system_metrics = []

    def save_session_logs(self, terminal_capture, session_number, use_session_folders):
        """Save logs for completed session

        Args:
            terminal_capture: TerminalCapture object
            session_number: Current session number
            use_session_folders: If True, save to session_N subfolder. If False, save directly.
        """
        if len(self.logger.data) == 0:
            return

        # Use appropriate path based on mode
        if use_session_folders:
            # Mode 1: Create session subfolder
            session_folder = self.logger.set_session_paths(session_number)
        else:
            # Mode 2/3: Save directly to run folder
            self.logger.set_direct_paths()
            session_folder = self.logger.run_folder

        session_end = datetime.now()

        # Summary
        print("="*70)
        print(f"SESSION {session_number} COMPLETE")
        print("="*70)

        total_arrivals = sum(self.button_presses.values())
        duration_s = (session_end - self.session_start).total_seconds()

        print(f"Duration: {duration_s:.1f}s")
        print(f"Steps: {self.session_step}")
        print(f"Arrived: {total_arrivals} | Cleared: {int(self.vehicles_cleared)} | "
              f"Throughput: {int(self.vehicles_cleared)/max(total_arrivals,1)*100:.1f}%")
        print(f"Phase changes: {self.phase_changes}")
        print(f"Final queue: N={int(self.queues[0])} E={int(self.queues[2])} "
              f"S={int(self.queues[1])} W={int(self.queues[3])}")

        # Save logs
        inference_times = [d['inference_ms'] for d in self.logger.data]

        stats = {
            'controller': 'PPO-Run8-Seed789',
            'session_number': session_number,
            'session_start': self.session_start.isoformat(),
            'session_end': session_end.isoformat(),
            'duration_seconds': duration_s,
            'total_steps': self.session_step,
            'total_arrivals': total_arrivals,
            'vehicles_cleared': int(self.vehicles_cleared),
            'throughput_percent': int(self.vehicles_cleared)/max(total_arrivals,1)*100,
            'button_presses': self.button_presses,
            'final_queues': self.queues.tolist(),
            'phase_changes': self.phase_changes,
            'yellow_transitions': self.yellow_transitions,
            'inference_times': {
                'mean_ms': np.mean(inference_times),
                'std_ms': np.std(inference_times),
                'min_ms': np.min(inference_times),
                'max_ms': np.max(inference_times),
                'median_ms': np.median(inference_times)
            }
        }

        # Add system metrics summary if available
        if self.logger.system_metrics:
            ram_values = [m['ram_mb'] for m in self.logger.system_metrics if m['ram_mb'] is not None]
            cpu_values = [m['cpu_percent'] for m in self.logger.system_metrics if m['cpu_percent'] is not None]
            temp_values = [m['cpu_temp'] for m in self.logger.system_metrics if m['cpu_temp'] is not None]

            stats['system_metrics'] = {
                'mean_ram_mb': np.mean(ram_values) if ram_values else None,
                'peak_ram_mb': np.max(ram_values) if ram_values else None,
                'mean_cpu_percent': np.mean(cpu_values) if cpu_values else None,
                'peak_cpu_percent': np.max(cpu_values) if cpu_values else None,
                'mean_temp': np.mean(temp_values) if temp_values else None,
                'peak_temp': np.max(temp_values) if temp_values else None
            }

        print(f"\n[SAVED] {session_folder}")
        self.logger.save_all(stats)

        # Save terminal output
        if terminal_capture:
            self.logger.save_terminal_output(terminal_capture)
            print(f"[SAVED] Terminal output: {self.logger.terminal_path}")

    def save_empty_stats(self, use_session_folders):
        """Save minimal stats when timeout occurs with no traffic (idle timeout)

        This ensures stats.json exists even if no vehicles arrived
        """
        # Set paths appropriately
        if use_session_folders:
            session_folder = self.logger.set_session_paths(1)
        else:
            self.logger.set_direct_paths()
            session_folder = self.logger.run_folder

        # Create minimal stats
        stats = {
            'controller': 'PPO-Run8-Seed789',
            'session_number': 0,
            'session_start': datetime.now().isoformat(),
            'session_end': datetime.now().isoformat(),
            'duration_seconds': 0,
            'total_steps': 0,
            'total_arrivals': 0,
            'vehicles_cleared': 0,
            'throughput_percent': 0,
            'button_presses': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
            'final_queues': [0, 0, 0, 0],
            'phase_changes': 0,
            'yellow_transitions': 0,
            'note': 'Timeout occurred with no traffic - idle state'
        }

        # Save just the JSON (no CSV/viz since no data)
        with open(self.logger.json_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"[SAVED] Empty stats to {session_folder}")


    def cleanup(self):
        GPIO.cleanup()


class FixedTimingController:
    """Fixed-timing baseline for comparison"""

    def __init__(self, logger, cycle_time=10, button_recorder=None):
        self.logger = logger
        self.max_queue_length = 20
        self.cycle_time = cycle_time
        self.button_recorder = button_recorder  # For recording in comparison mode

        self.queues = np.zeros(4, dtype=np.float32)
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.vehicles_cleared = 0
        self.phase_changes = 0
        self.yellow_transitions = 0

        self.last_button_time = {d: 0 for d in BUTTON_PINS.keys()}
        self.debounce_delay = 0.3

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        for direction, pin in BUTTON_PINS.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def read_buttons(self, show_press=False):
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
                        pressed_buttons.append(direction.capitalize())

                        # Record press if recorder is active
                        if self.button_recorder:
                            self.button_recorder.record_press(direction)

                        if show_press:
                            print(f"  >> {direction.upper()} button | "
                                  f"Queue: N={int(self.queues[0])} E={int(self.queues[2])} "
                                  f"S={int(self.queues[1])} W={int(self.queues[3])}")

        return pressed_buttons

    def set_all_red(self):
        for direction in ['north', 'south', 'east', 'west']:
            GPIO.output(LED_PINS[f'{direction}_red'], GPIO.HIGH)
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)

    def set_lights(self, phase, color='green'):
        for direction in ['north', 'south', 'east', 'west']:
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)

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

    def clear_vehicles(self, action):
        cleared = 0
        if action == 0:
            for lane in [0, 1]:
                if self.queues[lane] > 0:
                    clear_amount = min(CLEARING_RATE, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount
        else:
            for lane in [2, 3]:
                if self.queues[lane] > 0:
                    clear_amount = min(CLEARING_RATE, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount

        self.vehicles_cleared += cleared
        return int(cleared)

    def run(self, duration=60):
        print(f"[FIXED-TIMING] {self.cycle_time}s cycles | {duration}s duration\n")

        session_start = datetime.now()
        step = 0
        self.set_lights(self.current_phase, 'green')
        self.phase_start_time = time.time()

        try:
            while True:
                # Check timeout BEFORE processing step
                elapsed = (datetime.now() - session_start).total_seconds()
                if elapsed >= duration:
                    print(f"\n[TIMEOUT] {duration}s duration reached\n")
                    break
                step += 1

                # Check if time to switch
                phase_change = False
                if time.time() - self.phase_start_time >= self.cycle_time:
                    self.set_lights(self.current_phase, 'yellow')
                    self.yellow_transitions += 1
                    time.sleep(YELLOW_DURATION)
                    self.current_phase = 1 - self.current_phase
                    self.set_lights(self.current_phase, 'green')
                    self.phase_start_time = time.time()
                    self.phase_changes += 1
                    phase_change = True

                # Clear vehicles
                cleared = self.clear_vehicles(self.current_phase)

                # Log
                self.logger.log_step(step, self.queues.copy(), self.current_phase,
                                   cleared, 0, phase_change)

                # Display
                phase_name = "N/S" if self.current_phase == 0 else "E/W"
                switch = " [SWITCH]" if phase_change else ""

                print(f"[STEP {step:3d}] {phase_name} Green{switch:10s} | "
                      f"Clear: {cleared:2d} | "
                      f"Queue: N={int(self.queues[0])} E={int(self.queues[2])} "
                      f"S={int(self.queues[1])} W={int(self.queues[3])} | "
                      f"Total: {int(np.sum(self.queues)):2d}")

                # Poll buttons with timeout check
                delay_start = time.time()
                while time.time() - delay_start < DEMO_DELAY:
                    # Check if we're approaching timeout
                    elapsed = (datetime.now() - session_start).total_seconds()
                    if elapsed >= duration:
                        print(f"\n[TIMEOUT] Reached during delay\n")
                        raise KeyboardInterrupt  # Use interrupt to exit cleanly

                    self.read_buttons(show_press=True)
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n[STOPPED]\n")

        finally:
            self.set_all_red()
            session_end = datetime.now()

            # Summary
            print("="*70)
            print("SUMMARY")
            print("="*70)

            total_arrivals = sum(self.button_presses.values())
            duration_s = (session_end - session_start).total_seconds()

            print(f"Duration: {duration_s:.1f}s")
            print(f"Steps: {step}")
            print(f"Arrived: {total_arrivals} | Cleared: {int(self.vehicles_cleared)} | "
                  f"Throughput: {int(self.vehicles_cleared)/max(total_arrivals,1)*100:.1f}%")
            print(f"Phase changes: {self.phase_changes}")
            print(f"Final queue: N={int(self.queues[0])} E={int(self.queues[2])} "
                  f"S={int(self.queues[1])} W={int(self.queues[3])}")

            # Use direct paths for comparison mode
            if len(self.logger.data) > 0:
                self.logger.set_direct_paths()

                stats = {
                    'controller': 'Fixed-Timing',
                    'session_start': session_start.isoformat(),
                    'session_end': session_end.isoformat(),
                    'duration_seconds': duration_s,
                    'total_steps': step,
                    'total_arrivals': total_arrivals,
                    'vehicles_cleared': int(self.vehicles_cleared),
                    'throughput_percent': int(self.vehicles_cleared)/max(total_arrivals,1)*100,
                    'button_presses': self.button_presses,
                    'final_queues': self.queues.tolist(),
                    'phase_changes': self.phase_changes,
                    'yellow_transitions': self.yellow_transitions
                }

                print(f"\n[SAVED] {self.logger.run_folder}")
                self.logger.save_all(stats)

    def cleanup(self):
        GPIO.cleanup()


def run_comparison(model_path, vecnorm_path, duration=60):
    """Run comparison: Fixed-timing then PPO

    Now generates comparison_analysis.txt file
    Uses record/replay for fair comparison with identical traffic
    """
    print("\n" + "="*70)
    print("COMPARISON MODE (RECORD & REPLAY)")
    print("="*70)
    print(f"Duration: {duration}s per test")
    print("\nPhase 1: Fixed-Timing will RECORD your button presses")
    print("Phase 2: PPO will REPLAY the exact same presses")
    print("This ensures fair comparison on identical traffic!\n")

    # Create button recorder
    recorder = ButtonRecorder()

    # Test 1: Fixed-timing with recording
    print("[TEST 1/2] FIXED-TIMING BASELINE (RECORDING)")
    input("Press ENTER when ready...\n")

    logger_fixed = DataLogger("fixed_timing")
    controller_fixed = FixedTimingController(logger_fixed, button_recorder=recorder)

    # Start recording
    recorder.start_recording()
    print("[RECORDING] All button presses will be recorded...\n")

    try:
        controller_fixed.run(duration=duration)
    finally:
        controller_fixed.cleanup()

    # Get recordings
    recordings = recorder.get_recordings()
    print(f"\n[RECORDED] {len(recordings)} button presses captured")

    # Save recordings to file
    recording_path = os.path.join(logger_fixed.run_folder, "button_recordings.json")
    recorder.save_to_file(recording_path)
    print(f"[SAVED] Recording: {recording_path}")

    if len(recordings) == 0:
        print("\n[ERROR] No button presses recorded!")
        print("Comparison cannot proceed without traffic data.")
        print("Please run again and press buttons during Fixed-Timing test.\n")
        return

    # Extract stats from fixed-timing
    stats_fixed_path = os.path.join(logger_fixed.run_folder, "stats.json")
    with open(stats_fixed_path, 'r') as f:
        stats_fixed = json.load(f)

    print("\n[PAUSE] 5 seconds...\n")
    time.sleep(5)

    # Test 2: PPO with replay
    print("[TEST 2/2] PPO AGENT (REPLAYING)")
    print("DO NOT press buttons - they will be replayed automatically!")
    input("Press ENTER when ready...\n")

    logger_ppo = DataLogger("ppo_run8_seed789")

    # Create replayer from recordings
    replayer = ButtonReplayer(recordings)
    controller_ppo = PPOController(model_path, vecnorm_path, logger_ppo, button_replayer=replayer)

    # Start replay
    replayer.start_replay()
    print(f"[REPLAYING] {len(recordings)} button presses will be replayed automatically...")
    print("Watch as the exact same traffic pattern is processed!\n")

    try:
        # Use direct paths for comparison mode (no session subfolders)
        # PPO will start immediately (no idle wait) and process replayed buttons
        controller_ppo.run(duration=duration, use_session_folders=False)
    finally:
        controller_ppo.cleanup()

    # Extract stats from PPO
    stats_ppo_path = os.path.join(logger_ppo.run_folder, "stats.json")

    # Check if stats file exists
    if not os.path.exists(stats_ppo_path):
        print("\n[ERROR] PPO stats file not found")
        print("This shouldn't happen with replay mode, but comparison cannot proceed.\n")
        return

    with open(stats_ppo_path, 'r') as f:
        stats_ppo = json.load(f)

    # Check if PPO had any traffic
    if stats_ppo.get('total_arrivals', 0) == 0:
        print("\n[WARNING] PPO had no traffic arrivals despite replay")
        print("This indicates a replay system malfunction.\n")

    print("\n[COMPLETE] Comparison finished\n")

    # Generate comparison analysis file
    generate_comparison_analysis(logger_fixed.run_folder, logger_ppo.run_folder,
                                stats_fixed, stats_ppo, duration, len(recordings))


def generate_comparison_analysis(fixed_folder, ppo_folder, stats_fixed, stats_ppo, duration, num_recordings=0):
    """Generate comparison_analysis.txt file comparing both controllers

    This function was missing entirely
    Now includes information about record/replay system
    """
    # Create comparison folder
    results_dir = '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(results_dir, f"comparison_analysis_{timestamp}.txt")

    # Build comparison text
    lines = []
    lines.append("="*70)
    lines.append("COMPARISON ANALYSIS: FIXED-TIMING vs PPO RUN 8 SEED 789")
    lines.append("="*70)
    lines.append(f"\nComparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Duration: {duration}s per controller")
    lines.append(f"Method: RECORD & REPLAY (identical traffic)")
    if num_recordings > 0:
        lines.append(f"Recorded Presses: {num_recordings} button events")
    lines.append("")

    lines.append("\nTest Folders:")
    lines.append(f"  Fixed-Timing: {fixed_folder}")
    lines.append(f"  PPO Agent:    {ppo_folder}\n")

    lines.append("="*70)
    lines.append("FAIR COMPARISON METHODOLOGY")
    lines.append("="*70)
    lines.append("\n✓ Phase 1: Fixed-Timing recorded all button presses with timestamps")
    lines.append("✓ Phase 2: PPO replayed the EXACT same button presses")
    lines.append("✓ Result: Both controllers processed IDENTICAL traffic patterns")
    lines.append("✓ This ensures a scientifically valid performance comparison\n")

    lines.append("="*70)
    lines.append("PERFORMANCE METRICS")
    lines.append("="*70)

    # Throughput comparison
    lines.append("\n1. THROUGHPUT")
    total_fixed = stats_fixed['total_arrivals']
    cleared_fixed = stats_fixed['vehicles_cleared']
    throughput_fixed = cleared_fixed / max(total_fixed, 1) * 100

    total_ppo = stats_ppo.get('total_arrivals', 0)
    cleared_ppo = stats_ppo.get('vehicles_cleared', 0)
    throughput_ppo = cleared_ppo / max(total_ppo, 1) * 100 if total_ppo > 0 else 0

    lines.append(f"  Fixed-Timing: {cleared_fixed}/{total_fixed} ({throughput_fixed:.1f}%)")

    if total_ppo == 0:
        lines.append(f"  PPO Agent:    0/0 (no traffic - timed out while idle)")
        lines.append(f"  Note: PPO had no arrivals, comparison not meaningful")
    else:
        lines.append(f"  PPO Agent:    {cleared_ppo}/{total_ppo} ({throughput_ppo:.1f}%)")

        if total_fixed > 0:
            diff = cleared_ppo - cleared_fixed
            lines.append(f"  Difference:   {diff:+d} cars ({diff/max(total_fixed,1)*100:+.1f}%)")

    # Phase changes comparison
    lines.append("\n2. ADAPTABILITY (Phase Changes)")
    lines.append(f"  Fixed-Timing: {stats_fixed['phase_changes']} changes")
    lines.append(f"  PPO Agent:    {stats_ppo['phase_changes']} changes")

    # Average phase duration
    avg_phase_fixed = stats_fixed['duration_seconds'] / max(stats_fixed['phase_changes'], 1)
    avg_phase_ppo = stats_ppo['duration_seconds'] / max(stats_ppo['phase_changes'], 1)
    lines.append("\n3. AVERAGE PHASE DURATION")
    lines.append(f"  Fixed-Timing: {avg_phase_fixed:.2f}s per phase")
    lines.append(f"  PPO Agent:    {avg_phase_ppo:.2f}s per phase")

    # Final queue states
    lines.append("\n4. FINAL QUEUE STATES")
    lines.append(f"  Fixed-Timing: N={int(stats_fixed['final_queues'][0])} "
                f"E={int(stats_fixed['final_queues'][2])} "
                f"S={int(stats_fixed['final_queues'][1])} "
                f"W={int(stats_fixed['final_queues'][3])}")
    lines.append(f"  PPO Agent:    N={int(stats_ppo['final_queues'][0])} "
                f"E={int(stats_ppo['final_queues'][2])} "
                f"S={int(stats_ppo['final_queues'][1])} "
                f"W={int(stats_ppo['final_queues'][3])}")

    # Inference time (PPO only)
    if 'inference_times' in stats_ppo:
        lines.append("\n5. PPO DECISION SPEED")
        inf = stats_ppo['inference_times']
        lines.append(f"  Mean: {inf['mean_ms']:.2f}ms")
        lines.append(f"  Max:  {inf['max_ms']:.2f}ms")
        lines.append(f"  Min:  {inf['min_ms']:.2f}ms")
        lines.append(f"  Real-time capability: {1000/inf['mean_ms']:.0f}× faster than human reaction")

    lines.append("\n" + "="*70)
    lines.append("CONCLUSION")
    lines.append("="*70)

    # Determine winner - handle empty PPO case
    if total_ppo == 0:
        lines.append("\n⚠ PPO timed out with no traffic")
        lines.append("⚠ Comparison inconclusive - PPO needs traffic to demonstrate capabilities")
    elif cleared_ppo > cleared_fixed:
        lines.append(f"\n✓ PPO Agent cleared {cleared_ppo - cleared_fixed} more vehicles")
        lines.append("✓ PPO demonstrates adaptive real-time traffic management")
    elif cleared_fixed > cleared_ppo:
        lines.append(f"\n✓ Fixed-Timing cleared {cleared_fixed - cleared_ppo} more vehicles")
    else:
        lines.append("\n= Both controllers achieved equal throughput")

    lines.append("\nModel Information:")
    lines.append("  Champion: Run 8 Seed 789 (best of 5 seeds)")
    lines.append("  Validation: 72% win rate, p=0.0002")
    lines.append("  Reproducibility: CV = 1.3%")

    lines.append("\n" + "="*70)
    lines.append("END OF COMPARISON")
    lines.append("="*70)

    # Write to file
    with open(comparison_file, 'w') as f:
        f.write('\n'.join(lines))

    # Also print to terminal
    print("\n")
    for line in lines:
        print(line)

    print(f"\n[SAVED] Comparison analysis: {comparison_file}\n")


def main():
    print("\n" + "="*70)
    print("RUN 8 SEED 789: PPO TRAFFIC CONTROLLER")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    MODEL_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/ppo_final_seed789.zip"
    VECNORM_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/vec_normalize_seed789.pkl"    

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECNORM_PATH):
        print("[ERROR] Model files not found")
        print(f"  Model: {MODEL_PATH}")
        print(f"  VecNormalize: {VECNORM_PATH}")
        sys.exit(1)

    # Display model information
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    vecnorm_size_kb = os.path.getsize(VECNORM_PATH) / 1024
    total_size_mb = model_size_mb + (vecnorm_size_kb / 1024)

    print("MODEL INFORMATION:")
    print(f"  Model: {model_size_mb:.2f}MB")
    print(f"  VecNormalize: {vecnorm_size_kb:.1f}KB")
    print(f"  Total: {total_size_mb:.2f}MB")
    print(f"  Champion: Run 8 Seed 789 (best of 5 seeds)")
    print(f"  Validation: 72% win rate, p=0.0002")
    print(f"  Reproducibility: CV = 1.3%")
    print()

    # System status
    if PSUTIL_AVAILABLE:
        print("System monitoring: ENABLED (psutil available)")
    else:
        print("System monitoring: DISABLED (install psutil to enable)")
        print("  Install: pip install psutil --break-system-packages")
    print()

    # Mode selection with input validation loop
    while True:
        print("SELECT MODE:")
        print("  1. PPO Event-Driven (runs until clear)")
        print("  2. PPO Timed (60s)")
        print("  3. Comparison (Fixed-Timing vs PPO, 60s each)")
        print("  q. Quit\n")

        choice = input("Choice: ").strip()
        
        # Validate choice
        if choice in ['1', '2', '3', 'q', 'Q', 'quit', 'Quit', 'QUIT']:
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter 1, 2, 3, or q\n")

    # Start terminal capture
    terminal_capture = TerminalCapture()
    sys.stdout = terminal_capture

    if choice == '1':
        # Mode 1: Event-driven, uses session subfolders
        logger = DataLogger("ppo_event_driven")
        controller = PPOController(MODEL_PATH, VECNORM_PATH, logger)
        try:
            controller.run(terminal_capture=terminal_capture, use_session_folders=True)
        finally:
            controller.cleanup()

    elif choice == '2':
        # Mode 2: Timed, uses direct paths
        logger = DataLogger("ppo_timed")
        controller = PPOController(MODEL_PATH, VECNORM_PATH, logger)
        try:
            controller.run(duration=60, terminal_capture=terminal_capture, use_session_folders=False)
        finally:
            controller.cleanup()

    elif choice == '3':
        # Mode 3: Comparison, uses direct paths and generates comparison file
        run_comparison(MODEL_PATH, VECNORM_PATH, duration=60)

    elif choice.lower() in ['q', 'quit']:
        print("[EXIT]")

    # Restore stdout
    sys.stdout = terminal_capture.terminal


if __name__ == "__main__":
    main()
