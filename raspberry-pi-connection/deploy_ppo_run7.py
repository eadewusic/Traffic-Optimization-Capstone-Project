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

BUTTON_PINS = {'north': 26, 'east': 25, 'south': 27, 'west': 8}

class DataLogger:
    """Comprehensive data logging for hardware deployment"""
    
    def __init__(self, log_dir='/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"deployment_log_{timestamp}.csv")
        self.viz_path = os.path.join(log_dir, f"deployment_viz_{timestamp}.png")
        self.json_path = os.path.join(log_dir, f"deployment_stats_{timestamp}.json")
        
        self.data = []
        self.start_time = time.time()
    
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
        print(f" CSV saved: {self.csv_path}")
        return df
    
    def create_visualization(self, df):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Hardware Deployment Performance - Run 7 PPO', 
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
        print(f" Visualization saved: {self.viz_path}")
        plt.close()
    
    def save_statistics(self, stats):
        """Save summary statistics"""
        with open(self.json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f" Statistics saved: {self.json_path}")


class HardwareController:
    """Full-featured hardware controller with logging"""
    
    def __init__(self, model_path, vecnorm_path, logger):
        print("\n Initializing Hardware Controller...")
        
        self.logger = logger
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup LEDs
        print("   Setting up LED outputs...")
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Setup buttons
        print("   Setting up button inputs...")
        for pin in BUTTON_PINS.values():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Test hardware
        print("   Testing hardware...")
        self.test_hardware_quick()
        
        # Load model
        print("   Loading PPO model...")
        self.model = PPO.load(model_path)
        
        print("   Loading VecNormalize...")
        from environments.run7_env import Run7TrafficEnv
        dummy_env = DummyVecEnv([lambda: Run7TrafficEnv()])
        self.vec_env = VecNormalize.load(vecnorm_path, dummy_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        
        # State
        self.queues = np.zeros(4, dtype=float)
        self.max_queue = 20
        self.current_phase = 0
        
        # Metrics
        self.total_cleared = 0
        self.total_steps = 0
        self.phase_changes = 0
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.inference_times = []
        
        # Button debouncing
        self.last_button_time = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.debounce_delay = 0.3  # 300ms debounce
        
        print(" Hardware Controller Ready\n")
    
    def test_hardware_quick(self):
        """Quick hardware validation"""
        # Test each direction briefly
        for direction in ['north', 'east', 'south', 'west']:
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(LED_PINS[f'{direction}_green'], GPIO.LOW)
    
    def set_lights(self, phase):
        """Set traffic lights"""
        # Turn off all
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)
        
        if phase == 0:  # N/S green
            GPIO.output(LED_PINS['north_green'], GPIO.HIGH)
            GPIO.output(LED_PINS['south_green'], GPIO.HIGH)
            GPIO.output(LED_PINS['east_red'], GPIO.HIGH)
            GPIO.output(LED_PINS['west_red'], GPIO.HIGH)
        else:  # E/W green
            GPIO.output(LED_PINS['north_red'], GPIO.HIGH)
            GPIO.output(LED_PINS['south_red'], GPIO.HIGH)
            GPIO.output(LED_PINS['east_green'], GPIO.HIGH)
            GPIO.output(LED_PINS['west_green'], GPIO.HIGH)
    
    def read_queues_debounced(self):
        """Read button presses with debouncing"""
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
        """Demo mode - 60 second demonstration"""
        print(f" DEMO MODE - {duration} SECOND DEMONSTRATION")
        print("\n Press buttons to simulate vehicle arrivals")
        print("\n  Press Ctrl+C to stop\n")
        
        self._reset_metrics()
        start_time = time.time()
        step = 0
        
        try:
            while (time.time() - start_time) < duration:
                step += 1
                
                # Read inputs
                self.read_queues_debounced()
                
                # Get PPO decision
                obs = self.queues.copy()
                obs_norm = self.vec_env.normalize_obs(obs)
                
                start_inf = time.time()
                action, _ = self.model.predict(obs_norm, deterministic=True)
                inference_ms = (time.time() - start_inf) * 1000
                
                action = int(action)
                self.inference_times.append(inference_ms)
                
                # Track phase changes
                phase_change = (action != self.current_phase)
                if phase_change:
                    self.phase_changes += 1
                    self.current_phase = action
                
                # Apply
                self.set_lights(action)
                cleared = self.clear_vehicles(action)
                
                # Log
                self.logger.log_step(step, self.queues, action, cleared, 
                                    inference_ms, phase_change)
                
                # Display
                self._print_status(step, action, cleared, inference_ms)
                
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
        self.inference_times = []
        self.button_presses = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    
    def _print_status(self, step, action, cleared, inference_ms):
        """Print current status"""
        q = self.queues.astype(int)
        phase = 'N/S' if action == 0 else 'E/W'
        print(f"Step {step:3d} | Q:[N={q[0]:2d} S={q[1]:2d} E={q[2]:2d} W={q[3]:2d}] | "
              f"Phase:{phase} | Clear:{cleared} | Inf:{inference_ms:.2f}ms", end='\r')
    
    def _print_final_stats(self, elapsed, steps):
        """Print final statistics"""
        print("\n DEPLOYMENT RESULTS")
        
        print(f"\n  Duration: {elapsed:.1f}s ({steps} steps)")
        
        print(f"\n Traffic Metrics:")
        print(f"   Total cleared: {self.total_cleared}")
        print(f"   Button presses: N={self.button_presses['north']}, "
              f"S={self.button_presses['south']}, "
              f"E={self.button_presses['east']}, "
              f"W={self.button_presses['west']}")
        print(f"   Final queues: N={int(self.queues[0])}, S={int(self.queues[1])}, "
              f"E={int(self.queues[2])}, W={int(self.queues[3])}")
        
        print(f"\n Control Metrics:")
        print(f"   Phase changes: {self.phase_changes}")
        print(f"   Avg phase duration: {elapsed/max(self.phase_changes, 1):.2f}s")
        
        print(f"\n Performance Metrics:")
        if self.inference_times:
            print(f"   Mean inference: {np.mean(self.inference_times):.2f}ms")
            print(f"   Max inference: {np.max(self.inference_times):.2f}ms")
            print(f"   Min inference: {np.min(self.inference_times):.2f}ms")
            print(f"   Std inference: {np.std(self.inference_times):.2f}ms")
            print(f"   Real-time: {'YES' if np.max(self.inference_times) < 100 else 'NO'}")
    
    def _get_statistics(self, elapsed, steps):
        """Get statistics dictionary"""
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'controller': 'PPO_Run7',
            'duration_seconds': float(elapsed),
            'total_steps': int(steps),
            'vehicles_cleared': int(self.total_cleared),
            'phase_changes': int(self.phase_changes),
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
        print(" GPIO cleaned up")


def main():
    """Main execution"""
    print("\n FULL-FEATURED HARDWARE DEPLOYMENT - RUN 7 PPO")
    print(f"\n {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" PPO + Raspberry Pi Traffic Light Controller with Auto-Logging\n")
    
    # Paths
    MODEL_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_7/final_model.zip"
    VECNORM_PATH = "/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/models/hardware_ppo/run_7/vecnormalize.pkl"
    
    # Check files
    if not os.path.exists(MODEL_PATH):
        print(f" Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(VECNORM_PATH):
        print(f" VecNormalize not found: {VECNORM_PATH}")
        sys.exit(1)
    
    print(" Model files found")
    
    # Mode selection
    print("\n SELECT MODE:")
    print("   1. Demo Mode (60s with full logging)")
    print("   2. Extended Demo (120s)")
    print("   3. Quick Test (30s)")
    
    try:
        mode = input("\nEnter mode (1-3): ").strip()
        durations = {'1': 60, '2': 120, '3': 30}
        duration = durations.get(mode, 60)
    except:
        duration = 60
    
    print(f"\n Starting {duration}s deployment with full logging...\n")
    
    logger = None
    controller = None
    
    try:
        # Initialize
        logger = DataLogger()
        controller = HardwareController(MODEL_PATH, VECNORM_PATH, logger)
        
        # Run
        stats = controller.run_demo_mode(duration=duration)
        
        # Save everything
        print("\n Saving results...")
        df = logger.save_csv()
        logger.create_visualization(df)
        logger.save_statistics(stats)
        
        print("\n DEPLOYMENT COMPLETE - ALL DATA SAVED")
        print(f"\n Results saved to:")
        print(f"   CSV: {logger.csv_path}")
        print(f"   Plot: {logger.viz_path}")
        print(f"   Stats: {logger.json_path}")
        
    except KeyboardInterrupt:
        print("\n\n  Deployment stopped by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            controller.cleanup()


if __name__ == "__main__":
    main()
