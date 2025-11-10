"""
Event-Driven PPO Traffic Controller - Run 8 Seed 789 Champion
Hardware Deployment Script - Production Version

NEW CHANGES (v2.0):
1. Event-Driven Architecture (not time-based)
   - Runs continuously until all lanes clear
   - No arbitrary 60s/120s time limits
   - Hibernates when idle, wakes on button press

2. Static Display with Increments
   - Uses ANSI escape codes to update in place
   - Shows +/- deltas in green/red
   - No scrolling text confusion

3. Separate Inference Time vs Demo Delay
   - Inference time: Real 5.78ms metric (unchanged)
   - Demo delay: 15-20s pause for jury visibility
   - Clearly labeled to avoid confusion

4. System Resource Monitoring
   - RAM usage tracking (MB and %)
   - CPU usage monitoring
   - Temperature monitoring (Raspberry Pi specific)
   - Logged and visualized

5. Session Management
   - Starts: First button press
   - Ends: All lanes clear OR Ctrl+C force stop
   - Idle state between sessions

6. Model Information
   - Displays full model size at startup
   - Shows validation metrics
   - Multi-seed champion context

Key Components:
Hardware Control:
    - 12 GPIO pins for traffic LEDs (3 per direction: red, yellow, green)
    - 4 button inputs (debounced) to simulate vehicle arrivals
    - Safe yellow light transitions (2 seconds)
    - All-red idle mode for safety

RL Inference:
    - Loads trained Stable-Baselines3 PPO model with VecNormalize
    - Real-time phase decisions (North/South vs East/West green)
    - Processes normalized queue observations
    - Outputs discrete actions

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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import json
import sys
import os
from datetime import datetime

# import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] psutil not installed. System monitoring disabled.")
    print("Install with: pip install psutil --break-system-packages\n")

# Add path for environments
sys.path.append('/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/environments')

# GPIO Pin Configuration
LED_PINS = {
    'north_red': 16, 'north_yellow': 20, 'north_green': 21,
    'east_red': 5, 'east_yellow': 6, 'east_green': 13,
    'south_red': 23, 'south_yellow': 24, 'south_green': 25,
    'west_red': 14, 'west_yellow': 4, 'west_green': 18
}

BUTTON_PINS = {
    'north': 9,   # GPIO 9  - Pin 21
    'east': 10,   # GPIO 10 - Pin 19
    'south': 22,  # GPIO 22 - Pin 15
    'west': 17    # GPIO 17 - Pin 11
}

# CRITICAL: Visualization delay for demonstration (does NOT affect inference time!)
DEMO_DELAY = 15  # seconds - pause between steps for jury clarity
YELLOW_DURATION = 2  # seconds - safety transition time

# Utility Functions
def clear_lines(n):
    """
    Move cursor up n lines and clear using ANSI escape codes
    
    Args:
        n: Number of lines to clear
    """
    for _ in range(n):
        sys.stdout.write('\033[F')  # Move cursor up one line
        sys.stdout.write('\033[K')  # Clear line
    sys.stdout.flush()


def format_delta(val, delta):
    """
    Format value with colored delta indicator
    
    Args:
        val: Current value
        delta: Change from previous value
    
    Returns:
        Formatted string with ANSI color codes
    """
    val_str = f"{int(val):2d}"
    if delta > 0:
        return f"{val_str} \033[92m(+{int(delta)})\033[0m"  # Green for additions
    elif delta < 0:
        return f"{val_str} \033[91m({int(delta)})\033[0m"   # Red for removals
    else:
        return f"{val_str}        "  # Spaces for alignment


def get_system_metrics():
    """
    Get real-time system resource usage
    
    Returns:
        Dictionary with RAM, CPU, and temperature metrics
    """
    if not PSUTIL_AVAILABLE:
        return {
            'ram_mb': None,
            'ram_percent': None,
            'cpu_percent': None,
            'cpu_temp': None
        }
    
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


# Data Logging
class DataLogger:
    """
    Comprehensive data logging for hardware deployment
    
    Logs all metrics to timestamped folder with multiple output formats:
    - CSV: Step-by-step data
    - JSON: Summary statistics
    - PNG: Visualizations
    - TXT: Human-readable report
    """
    
    def __init__(self, log_dir='/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'):
        """
        Initialize logger with timestamped output folder
        
        Args:
            log_dir: Base directory for results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join(log_dir, f"run8_seed789_event_driven_{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)
        
        # File paths
        self.csv_path = os.path.join(self.run_folder, "deployment_log.csv")
        self.viz_path = os.path.join(self.run_folder, "deployment_viz.png")
        self.json_path = os.path.join(self.run_folder, "deployment_stats.json")
        self.txt_path = os.path.join(self.run_folder, "deployment_report.txt")
        
        self.data = []
        self.start_time = time.time()
        
        print(f" Run Log: {self.run_folder}\n")
    
    def log_step(self, step, queues, action, cleared, inference_ms, phase_change, 
                 ram_mb=None, cpu_percent=None, cpu_temp=None):
        """
        Log a single step with all metrics
        
        Args:
            step: Step number
            queues: Current queue state [N, S, E, W]
            action: PPO action (0=N/S, 1=E/W)
            cleared: Vehicles cleared this step
            inference_ms: Inference time in milliseconds
            phase_change: Boolean - did phase change?
            ram_mb: RAM usage in MB
            cpu_percent: CPU usage percentage
            cpu_temp: CPU temperature in Celsius
        """
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
            'phase_change': 1 if phase_change else 0,
            'ram_mb': ram_mb,
            'cpu_percent': cpu_percent,
            'cpu_temp': cpu_temp
        })
    
    def save_all(self, stats):
        """
        Save all outputs (CSV, visualization, JSON, text report)
        
        Args:
            stats: Summary statistics dictionary
        
        Returns:
            DataFrame of logged data
        """
        # Save CSV
        df = pd.DataFrame(self.data)
        df.to_csv(self.csv_path, index=False)
        
        # Create visualization
        self.create_visualization(df)
        
        # Save JSON
        with open(self.json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save text report
        self.save_text_report(stats)
        
        return df
    
    def create_visualization(self, df):
        """
        Create comprehensive 6-panel visualization
        
        Args:
            df: DataFrame of logged data
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Run 8 Seed 789: Event-Driven Deployment\n(Multi-Seed Champion)', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Queue dynamics over time
        axes[0, 0].plot(df['timestamp'], df['north_queue'], label='North', alpha=0.7)
        axes[0, 0].plot(df['timestamp'], df['south_queue'], label='South', alpha=0.7)
        axes[0, 0].plot(df['timestamp'], df['east_queue'], label='East', alpha=0.7)
        axes[0, 0].plot(df['timestamp'], df['west_queue'], label='West', alpha=0.7)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Queue Length')
        axes[0, 0].set_title('Queue Dynamics by Direction')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Total queue over time
        axes[0, 1].plot(df['timestamp'], df['total_queue'], color='red', linewidth=2)
        axes[0, 1].fill_between(df['timestamp'], 0, df['total_queue'], alpha=0.3, color='red')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Total Queue Length')
        axes[0, 1].set_title('Total Congestion Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Inference time distribution
        axes[1, 0].hist(df['inference_ms'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].axvline(df['inference_ms'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: {df['inference_ms'].mean():.2f}ms")
        axes[1, 0].axvline(100, color='orange', linestyle='--', linewidth=2, label='100ms threshold')
        axes[1, 0].set_xlabel('Inference Time (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('AI Decision Speed (Real Metric)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Vehicles cleared per step
        axes[1, 1].bar(df['step'], df['cleared'], alpha=0.6, color='green')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Vehicles Cleared')
        axes[1, 1].set_title('Clearing Performance')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 5: System resources (RAM)
        if df['ram_mb'].notna().any():
            axes[2, 0].plot(df['timestamp'], df['ram_mb'], color='blue', linewidth=2, label='RAM Usage')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_ylabel('RAM (MB)')
            axes[2, 0].set_title('Memory Usage')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'RAM monitoring unavailable\n(psutil not installed)', 
                          ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Memory Usage')
        
        # Plot 6: CPU temperature
        if df['cpu_temp'].notna().any():
            axes[2, 1].plot(df['timestamp'], df['cpu_temp'], color='red', linewidth=2, label='CPU Temp')
            axes[2, 1].axhline(70, color='orange', linestyle='--', label='Warning (70°C)')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Temperature (°C)')
            axes[2, 1].set_title('CPU Temperature')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'Temperature monitoring unavailable', 
                          ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('CPU Temperature')
        
        plt.tight_layout()
        plt.savefig(self.viz_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_text_report(self, stats):
        """
        Save human-readable text report
        
        Args:
            stats: Summary statistics dictionary
        """
        with open(self.txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" EVENT-DRIVEN DEPLOYMENT - RUN 8 SEED 789\n")
            f.write(" Multi-Seed Champion Model\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Session Start: {stats['session_start']}\n")
            f.write(f"Session End: {stats['session_end']}\n")
            f.write(f"Duration: {stats['duration_seconds']:.1f} seconds\n")
            f.write(f"Total Steps: {stats['total_steps']}\n\n")
            
            f.write("="*70 + "\n")
            f.write(" TRAFFIC METRICS\n")
            f.write("="*70 + "\n")
            f.write(f"Total Cars Arrived: {stats['total_arrivals']}\n")
            f.write(f"Total Cars Cleared: {stats['vehicles_cleared']}\n")
            f.write(f"Throughput: {stats['throughput_percent']:.1f}%\n")
            f.write(f"Final Queue: {int(sum(stats['final_queues']))} cars\n")
            f.write(f"Button Presses: N={stats['button_presses']['north']}, "
                   f"S={stats['button_presses']['south']}, "
                   f"E={stats['button_presses']['east']}, "
                   f"W={stats['button_presses']['west']}\n\n")
            
            f.write("="*70 + "\n")
            f.write(" INFERENCE PERFORMANCE (Real AI Speed)\n")
            f.write("="*70 + "\n")
            inf = stats['inference_times']
            f.write(f"Mean: {inf['mean_ms']:.3f}ms\n")
            f.write(f"Std Dev: {inf['std_ms']:.3f}ms\n")
            f.write(f"Min: {inf['min_ms']:.3f}ms\n")
            f.write(f"Max: {inf['max_ms']:.3f}ms\n")
            f.write(f"Median: {inf['median_ms']:.3f}ms\n")
            f.write(f"Real-time Capable: {'YES' if inf['max_ms'] < 100 else 'NO'} (<100ms requirement)\n")
            if inf['mean_ms'] > 0:
                f.write(f"Speed Factor: {1000/inf['mean_ms']:.0f}× faster than human reaction\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write(" VISUALIZATION DELAY (Demo Only)\n")
            f.write("="*70 + "\n")
            f.write(f"Delay per step: {DEMO_DELAY}s (for jury clarity)\n")
            f.write(f"This does NOT affect inference time metric!\n")
            f.write(f"Real deployment would have NO delay.\n")
            f.write(f"The {inf['mean_ms']:.2f}ms inference time is the real metric.\n\n")
            
            f.write("="*70 + "\n")
            f.write(" CONTROL METRICS\n")
            f.write("="*70 + "\n")
            f.write(f"Phase Changes: {stats['phase_changes']}\n")
            f.write(f"Yellow Transitions: {stats['yellow_transitions']}\n")
            avg_phase = stats['duration_seconds'] / max(stats['phase_changes'], 1)
            f.write(f"Average Phase Duration: {avg_phase:.2f} seconds\n\n")
            
            if stats.get('system_metrics') and stats['system_metrics'].get('peak_ram_mb'):
                f.write("="*70 + "\n")
                f.write(" SYSTEM RESOURCES\n")
                f.write("="*70 + "\n")
                sys_m = stats['system_metrics']
                f.write(f"Peak RAM: {sys_m['peak_ram_mb']:.1f}MB\n")
                f.write(f"Mean RAM: {sys_m['mean_ram_mb']:.1f}MB\n")
                f.write(f"Mean CPU: {sys_m['mean_cpu_percent']:.1f}%\n")
                if sys_m.get('peak_temp'):
                    f.write(f"Peak Temp: {sys_m['peak_temp']:.1f}°C\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write(" MODEL VALIDATION\n")
            f.write("="*70 + "\n")
            f.write("This is the champion model from Run 8 multi-seed validation.\n")
            f.write("Validation:\n")
            f.write("  - Best of 5 independent seeds (42, 123, 456, 789, 1000)\n")
            f.write("  - Reproducibility: CV = 1.3% (exceptional)\n")
            f.write("  - Baseline win rate: 72% (18/25 scenarios)\n")
            f.write("  - Statistical significance: p=0.0002\n")
            f.write("  - Real-time inference: 5.78ms mean\n\n")
            
            f.write("="*70 + "\n")


# Event-Driven Controller
class EventDrivenController:
    """
    Event-driven PPO traffic controller
    
    Runs continuously until all cars are cleared (or user stops with Ctrl+C).
    Features:
    - IDLE state: Hibernates when no cars present
    - ACTIVE state: Controls traffic with PPO
    - Automatic state transitions
    - Static display with increments
    - System resource monitoring
    """
    
    def __init__(self, model_path, vecnorm_path, logger):
        """
        Initialize event-driven controller
        
        Args:
            model_path: Path to PPO model file (.zip)
            vecnorm_path: Path to VecNormalize file (.pkl)
            logger: DataLogger instance
        """
        print("="*70)
        print(" INITIALIZING EVENT-DRIVEN SYSTEM")
        print("="*70)
        
        self.logger = logger
        self.max_queue_length = 20
        
        # Load PPO model
        print("\n[LOADING RUN 8 SEED 789]")
        print(f"  Model: {model_path}")
        print(f"  VecNormalize: {vecnorm_path}")
        
        from run7_env import Run7TrafficEnv
        
        self.model = PPO.load(model_path)
        dummy_env = DummyVecEnv([lambda: Run7TrafficEnv()])
        self.vec_env = VecNormalize.load(vecnorm_path, dummy_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        
        print("  Champion model loaded")
        print("  Multi-seed validated (best of 5)")
        print("  Statistical significance: p=0.0002")
        
        # Initialize state
        self.queues = np.zeros(4, dtype=np.float32)
        self.last_queues = np.zeros(4, dtype=np.float32)
        
        # GPIO Setup
        print("\n[GPIO SETUP]")
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup LEDs
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Setup buttons with pull-up resistors
        for direction, pin in BUTTON_PINS.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        print("  12 LEDs configured")
        print("  4 buttons configured (debounced)")
        
        # State tracking
        self.current_phase = 0  # 0 = N/S, 1 = E/W
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.vehicles_cleared = 0
        self.phase_changes = 0
        self.yellow_transitions = 0
        
        # Debouncing for buttons
        self.last_button_time = {d: 0 for d in BUTTON_PINS.keys()}
        self.debounce_delay = 0.3  # 300ms
        
        # System state
        self.state = 'IDLE'  # IDLE or ACTIVE
        
        print("\n[READY] System initialized")
        print("="*70 + "\n")
    
    def read_buttons(self):
        """
        Read button states with debouncing
        
        Returns:
            List of pressed button names (e.g., ['North', 'East'])
        """
        current_time = time.time()
        pressed_buttons = []
        
        for direction, pin in BUTTON_PINS.items():
            # Buttons are active LOW (pull-up resistor)
            if GPIO.input(pin) == GPIO.LOW:
                # Check debounce
                if current_time - self.last_button_time[direction] > self.debounce_delay:
                    self.last_button_time[direction] = current_time
                    
                    # Map button to queue lane
                    lane_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
                    lane = lane_map[direction]
                    
                    # Add to queue if not full
                    if self.queues[lane] < self.max_queue_length:
                        self.queues[lane] += 1
                        self.button_presses[direction] += 1
                        pressed_buttons.append(direction.capitalize())
        
        return pressed_buttons
    
    def set_all_red(self):
        """
        Set all lights to red (safety/idle mode)
        """
        for direction in ['north', 'south', 'east', 'west']:
            GPIO.output(LED_PINS[f'{direction}_red'], GPIO.HIGH)
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)
    
    def set_lights(self, phase, color='green'):
        """
        Control traffic lights
        
        Args:
            phase: 0 for N/S, 1 for E/W
            color: 'green' or 'yellow'
        """
        # Turn off all yellows first
        for direction in ['north', 'south', 'east', 'west']:
            GPIO.output(LED_PINS[f'{direction}_yellow'], GPIO.LOW)
        
        if color == 'green':
            if phase == 0:  # North/South green
                GPIO.output(LED_PINS['north_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['north_red'], GPIO.LOW)
                GPIO.output(LED_PINS['south_red'], GPIO.LOW)
                GPIO.output(LED_PINS['east_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['east_green'], GPIO.LOW)
                GPIO.output(LED_PINS['west_green'], GPIO.LOW)
            else:  # East/West green
                GPIO.output(LED_PINS['east_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_green'], GPIO.HIGH)
                GPIO.output(LED_PINS['east_red'], GPIO.LOW)
                GPIO.output(LED_PINS['west_red'], GPIO.LOW)
                GPIO.output(LED_PINS['north_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_red'], GPIO.HIGH)
                GPIO.output(LED_PINS['north_green'], GPIO.LOW)
                GPIO.output(LED_PINS['south_green'], GPIO.LOW)
        
        elif color == 'yellow':
            if phase == 0:  # North/South yellow
                GPIO.output(LED_PINS['north_green'], GPIO.LOW)
                GPIO.output(LED_PINS['south_green'], GPIO.LOW)
                GPIO.output(LED_PINS['north_yellow'], GPIO.HIGH)
                GPIO.output(LED_PINS['south_yellow'], GPIO.HIGH)
            else:  # East/West yellow
                GPIO.output(LED_PINS['east_green'], GPIO.LOW)
                GPIO.output(LED_PINS['west_green'], GPIO.LOW)
                GPIO.output(LED_PINS['east_yellow'], GPIO.HIGH)
                GPIO.output(LED_PINS['west_yellow'], GPIO.HIGH)
    
    def clear_vehicles(self, action):
        """
        Clear vehicles from active lanes
        
        Args:
            action: 0 for N/S, 1 for E/W
        
        Returns:
            Number of vehicles cleared
        """
        cleared = 0
        
        if action == 0:  # North/South green
            for lane in [0, 1]:  # North, South
                if self.queues[lane] > 0:
                    clear_amount = min(1, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount
        else:  # East/West green
            for lane in [2, 3]:  # East, West
                if self.queues[lane] > 0:
                    clear_amount = min(1, self.queues[lane])
                    self.queues[lane] -= clear_amount
                    cleared += clear_amount
        
        self.vehicles_cleared += cleared
        return int(cleared)
    
    def get_action(self):
        """
        Get action from PPO model
        
        IMPORTANT: This measures the REAL inference time (not demo delay)
        
        Returns:
            Tuple of (action, inference_time_ms)
        """
        # Normalize observation
        obs = self.queues.copy() / self.max_queue_length
        obs_norm = self.vec_env.normalize_obs(obs)
        
        # Time the inference (REAL metric)
        start_time = time.perf_counter()
        action, _ = self.model.predict(obs_norm, deterministic=True)
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        return int(action), inference_time
    
    def display_state_static(self, step, action, inference_ms, cleared, sys_metrics):
        """
        Display state with static update (no scrolling)
        
        Uses ANSI escape codes to update display in place.
        Shows colored deltas: green (+), red (-), or blank (no change)
        
        Args:
            step: Current step number
            action: PPO action (0=N/S, 1=E/W)
            inference_ms: Inference time in milliseconds
            cleared: Vehicles cleared this step
            sys_metrics: System resource metrics dictionary
        """
        
        # Clear previous display (12 lines)
        if step > 1:
            clear_lines(12)
        
        # Calculate deltas from last step
        deltas = self.queues - self.last_queues
        
        # Action name
        action_name = "North/South GREEN" if action == 0 else "East/West  GREEN"
        
        # Display box (Unicode box drawing characters)
        print("╔═══════════════════════════════════════════════════════╗")
        print(f"║ STEP {step:4d} │ {action_name:30s} ║")
        print("╠═══════════════════════════════════════════════════════╣")
        print(f"║ North: {format_delta(self.queues[0], deltas[0])}  │  Cleared: {cleared:2d} cars       ║")
        print(f"║ South: {format_delta(self.queues[1], deltas[1])}  │  Total:   {int(self.vehicles_cleared):3d} cars      ║")
        print(f"║ East:  {format_delta(self.queues[2], deltas[2])}  ├─────────────────────────║")
        print(f"║ West:  {format_delta(self.queues[3], deltas[3])}  │  Inference: {inference_ms:6.2f}ms  ║")
        print(f"║                                                       ║")
        
        # System metrics (if available)
        if sys_metrics['ram_mb'] is not None:
            print(f"║ RAM: {sys_metrics['ram_mb']:5.1f}MB  CPU: {sys_metrics['cpu_percent']:4.1f}%  Temp: {sys_metrics['cpu_temp'] if sys_metrics['cpu_temp'] else 0:4.1f}°C ║")
        else:
            print(f"║ System monitoring unavailable (install psutil)       ║")
        
        print("╚═══════════════════════════════════════════════════════╝")
        print(f"Demo delay: {DEMO_DELAY}s (for clarity - NOT inference time)")
        print()
        
        sys.stdout.flush()
        
        # Update last queues for next delta calculation
        self.last_queues = self.queues.copy()
    
    def run_continuous(self):
        """
        Event-driven continuous deployment
        
        Flow:
        1. Wait for first button press (IDLE)
        2. Control traffic until all lanes clear (ACTIVE)
        3. Return to IDLE and wait for next button press
        4. Repeat until Ctrl+C
        """
        print("╔═══════════════════════════════════════════════════════╗")
        print("║          EVENT-DRIVEN TRAFFIC CONTROL SYSTEM          ║")
        print("╠═══════════════════════════════════════════════════════╣")
        print("║  Mode: Continuous until all lanes clear              ║")
        print("║  Press buttons to add cars (max 20 per lane)         ║")
        print("║  Press Ctrl+C to force stop                          ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print()
        
        session_start = datetime.now()
        step = 0
        
        try:
            while True:
                # ============================================================
                # IDLE MODE: Wait for first car
                # ============================================================
                if np.sum(self.queues) == 0:
                    self.state = 'IDLE'
                    self.set_all_red()  # Safety: all red when idle
                    
                    if step == 0:
                        print(" System IDLE: Waiting for vehicles...")
                        print("   (Press any button to start)\n")
                    else:
                        print("\n All lanes cleared!")
                        print(" Returning to IDLE mode...")
                        print("   (Add more cars or press Ctrl+C to exit)\n")
                    
                    # Hibernate - wait for button press
                    while np.sum(self.queues) == 0:
                        pressed = self.read_buttons()
                        if pressed:
                            print(f" Vehicle detected from {', '.join(pressed)}!")
                            print(" Activating traffic control...\n")
                            time.sleep(1)  # Brief pause before starting
                            break
                        time.sleep(0.1)  # Poll every 100ms
                    
                    # If we just completed a session, continue waiting
                    if step > 0:
                        continue
                
                # ============================================================
                # ACTIVE MODE: Control traffic
                # ============================================================
                self.state = 'ACTIVE'
                step += 1
                
                # Check for new arrivals during operation
                pressed = self.read_buttons()
                
                # Get PPO decision (REAL inference time - not demo delay!)
                action, inference_ms = self.get_action()
                
                # Get system metrics
                sys_metrics = get_system_metrics()
                
                # Check for phase change
                phase_change = False
                if action != self.current_phase:
                    # Yellow transition
                    self.set_lights(self.current_phase, 'yellow')
                    self.yellow_transitions += 1
                    time.sleep(YELLOW_DURATION)
                    
                    # Switch phase
                    self.current_phase = action
                    self.set_lights(self.current_phase, 'green')
                    self.phase_changes += 1
                    phase_change = True
                
                # Clear vehicles from active lanes
                cleared = self.clear_vehicles(self.current_phase)
                
                # Log step
                self.logger.log_step(
                    step, self.queues.copy(), action, cleared, 
                    inference_ms, phase_change,
                    sys_metrics['ram_mb'], sys_metrics['cpu_percent'], 
                    sys_metrics['cpu_temp']
                )
                
                # Display state (static update - no scrolling)
                self.display_state_static(step, action, inference_ms, cleared, sys_metrics)
                
                # DEMO DELAY (for jury clarity - does NOT affect inference metric!)
                time.sleep(DEMO_DELAY)
        
        except KeyboardInterrupt:
            print("\n\n User stopped system (Ctrl+C)")
        
        finally:
            # Safety: all red when stopping
            self.set_all_red()
            session_end = datetime.now()
            
            # ============================================================
            # Final summary
            # ============================================================
            print("\n" + "="*70)
            print(" SESSION SUMMARY")
            print("="*70)
            
            total_arrivals = sum(self.button_presses.values())
            duration = (session_end - session_start).total_seconds()
            
            print(f"Duration: {duration:.1f}s")
            print(f"Steps: {step}")
            print(f"Cars arrived: {total_arrivals}")
            print(f"Cars cleared: {int(self.vehicles_cleared)}")
            
            if total_arrivals > 0:
                throughput = int(self.vehicles_cleared)/total_arrivals*100
                print(f"Throughput: {throughput:.1f}%")
            
            print(f"Phase changes: {self.phase_changes}")
            print(f"Final queues: N={int(self.queues[0])}, S={int(self.queues[1])}, "
                  f"E={int(self.queues[2])}, W={int(self.queues[3])}")
            print()
            
            # Compute statistics
            if len(self.logger.data) > 0:
                inference_times = [d['inference_ms'] for d in self.logger.data]
                ram_usage = [d['ram_mb'] for d in self.logger.data if d['ram_mb']]
                cpu_usage = [d['cpu_percent'] for d in self.logger.data if d['cpu_percent']]
                temps = [d['cpu_temp'] for d in self.logger.data if d['cpu_temp']]
                
                stats = {
                    'session_start': session_start.isoformat(),
                    'session_end': session_end.isoformat(),
                    'duration_seconds': duration,
                    'total_steps': step,
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
                    },
                    'system_metrics': {
                        'peak_ram_mb': max(ram_usage) if ram_usage else None,
                        'mean_ram_mb': np.mean(ram_usage) if ram_usage else None,
                        'mean_cpu_percent': np.mean(cpu_usage) if cpu_usage else None,
                        'peak_temp': max(temps) if temps else None
                    },
                    'demo_delay_seconds': DEMO_DELAY,
                    'note': 'Demo delay is for visualization only - does not affect inference time'
                }
                
                print("\n[SAVING] Generating reports...")
                self.logger.save_all(stats)
                print(f"  CSV: {self.logger.csv_path}")
                print(f"  Visualization: {self.logger.viz_path}")
                print(f"  Statistics: {self.logger.json_path}")
                print(f"  Report: {self.logger.txt_path}")
                print()

    def cleanup(self):
        """Clean up GPIO resources"""
        print("[CLEANUP] Shutting down GPIO...")
        GPIO.cleanup()
        print("  GPIO cleaned up")


# Main Execution
def main():
    """
    Main execution function
    
    Initializes system, runs event-driven deployment, and saves results.
    """
    print("\n" + "="*70)
    print(" RUN 8 SEED 789: EVENT-DRIVEN DEPLOYMENT")
    print(" Multi-Seed Champion Model")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check psutil availability
    if not PSUTIL_AVAILABLE:
        print("[INFO] System monitoring will be limited without psutil")
        print("Install with: pip install psutil --break-system-packages\n")
    
    # Model paths
    MODEL_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/ppo_final_seed789.zip"
    VECNORM_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_8/seed_789/vec_normalize_seed789.pkl"
    
    # Verify files exist
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(VECNORM_PATH):
        print(f"[ERROR] VecNormalize not found: {VECNORM_PATH}")
        sys.exit(1)
    
    print(" Model files found")
    print(" Champion from 5-seed validation")
    print(" Statistical validation: p=0.0002\n")
    
    # Display model size
    model_size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    vecnorm_size_kb = os.path.getsize(VECNORM_PATH) / 1024
    print(f" Model size: {model_size_mb:.2f}MB")
    print(f" VecNormalize size: {vecnorm_size_kb:.1f}KB")
    print(f" Total: {model_size_mb:.2f}MB (lightweight!)\n")
    
    print(f"  Demo delay: {DEMO_DELAY}s between steps")
    print(f"   (For jury clarity - NOT inference time!)")
    print(f"   Real inference: ~5.78ms\n")
    
    # Initialize system
    logger = None
    controller = None
    
    try:
        # Create logger
        logger = DataLogger()
        
        # Create controller
        controller = EventDrivenController(MODEL_PATH, VECNORM_PATH, logger)
        
        # Run continuous event-driven system
        controller.run_continuous()
        
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
