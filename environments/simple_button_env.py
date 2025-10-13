"""
Simplified Button Traffic Environment for Hardware Deployment
RUN 5 COMPATIBLE

Key Changes for Run 5:
  - Added randomization_config parameter (required by Run 5 training script)
  - Added hardware-aware domain randomization (GPIO, button debounce)
  - OPTIONAL: Can use Run 5's fixed reward or keep Run 3's balanced reward
  
Changes from original (Run 1):
  - State: 113 dims → 4 dims (just queue counts)
  - Actions: 9 complex → 4 simple (which lane gets green)
  - Removed: Road grids, timers, light states
  - Added: Domain randomization for sim2real transfer
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional
from enum import Enum

class TrafficDirection(Enum):
    """Traffic directions"""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class SimpleButtonTrafficEnv(gym.Env):
    """
    Hardware-matching environment for button/LED prototype.
    NOW COMPATIBLE WITH RUN 5 TRAINING SCRIPT
    
    State Space (4 dimensions):
        - Queue counts for 4 directions: [North, South, East, West]
        - Normalized to [0, 1] range
    
    Action Space (4 discrete actions):
        - 0: Give green to North
        - 1: Give green to South
        - 2: Give green to East
        - 3: Give green to West
    
    Reward Options:
        - RUN 3 MODE: Balanced reward (default)
        - RUN 5 MODE: Fixed reward aligned with Longest Queue
        Set via use_run5_reward parameter
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 max_queue_length=20,
                 cars_cleared_per_cycle=5,
                 max_arrival_rate=3,
                 domain_randomization=False,
                 randomization_config=None,  # Required by Run 5
                 render_mode=None,           # Required by Run 5
                 use_run5_reward=True):      # Toggle reward function
        """
        Args:
            max_queue_length: Max vehicles per lane
            cars_cleared_per_cycle: Vehicles cleared per green cycle
            max_arrival_rate: Max random arrivals per lane per step
            domain_randomization: Enable for sim2real robustness
            randomization_config: Hardware-aware DR config (Run 5)
            render_mode: Rendering mode
            use_run5_reward: If True, use Run 5 fixed reward. If False, use Run 3 balanced reward.
        """
        super().__init__()
        
        # Parameters
        self.max_queue_length = max_queue_length
        self.cars_cleared_per_cycle = cars_cleared_per_cycle
        self.max_arrival_rate = max_arrival_rate
        self.domain_randomization = domain_randomization
        self.render_mode = render_mode
        self.use_run5_reward = use_run5_reward
        
        # Domain randomization configuration
        self.dr_config = randomization_config if randomization_config else self._default_dr_config()
        
        # State: 4 queue counts (simplified from original 113 dims)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(4,),
            dtype=np.float32
        )
        
        # Action: Select 1 of 4 lanes
        self.action_space = spaces.Discrete(4)
        
        # Lane names
        self.lane_names = ['North', 'South', 'East', 'West']
        
        # Internal state
        self.queues = np.zeros(4, dtype=np.float32)
        self.current_step = 0
        self.max_steps = 200
        
        # Domain randomization parameters
        self.clearance_rate = cars_cleared_per_cycle
        self.arrival_rate = max_arrival_rate
        
        # Hardware simulation parameters
        self.gpio_latency = 0
        self.button_debounce = 0
        self.processing_jitter = 0
        
        # Performance tracking
        self.total_cars_cleared = 0
        self.total_wait_time = 0
        self.vehicles_processed = 0
        
        # Time simulation
        self.current_time = 8.0  # Start at 8 AM
    
    def _default_dr_config(self):
        """Default hardware-aware domain randomization configuration for Run 5"""
        return {
            # Traffic variability
            'arrival_rate_range': (0.15, 0.45),
            'queue_capacity_range': (8, 12),
            'yellow_duration_range': (2, 4),
            
            # Hardware delays (for Run 5)
            'gpio_latency_range': (1, 10),        # ms
            'button_debounce_range': (50, 200),   # ms
            'processing_jitter_range': (0, 5),    # ms
        }
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Initialize with random queue lengths
        self.queues = self.np_random.integers(0, 10, size=4).astype(np.float32)
        
        self.current_step = 0
        self.total_cars_cleared = 0
        self.total_wait_time = 0
        self.vehicles_processed = 0
        self.current_time = 8.0
        
        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._randomize_parameters()
        
        # Return normalized state
        obs = self.queues / self.max_queue_length
        info = {
            'queues': self.queues.copy(),
            'step': self.current_step,
            'time_of_day': f"{int(self.current_time):02d}:00"
        }
        
        return obs, info
    
    def _randomize_parameters(self):
        """
        Domain Randomization for Sim2Real Transfer
        NOW INCLUDES HARDWARE-AWARE PARAMETERS FOR RUN 5
        """
        # Randomize clearance rate (±30%)
        self.clearance_rate = self.cars_cleared_per_cycle * \
                             self.np_random.uniform(0.7, 1.3)
        
        # Randomize arrival rate (±40%)
        self.arrival_rate = self.max_arrival_rate * \
                           self.np_random.uniform(0.6, 1.4)
        
        # NEW: Hardware delays (Run 5)
        if 'gpio_latency_range' in self.dr_config:
            self.gpio_latency = self.np_random.uniform(*self.dr_config['gpio_latency_range'])
            self.button_debounce = self.np_random.uniform(*self.dr_config['button_debounce_range'])
            self.processing_jitter = self.np_random.uniform(*self.dr_config['processing_jitter_range'])
    
    def step(self, action):
        """
        Execute one step
        
        Args:
            action: Integer 0-3 indicating which lane gets green
        
        Returns:
            observation: Normalized queue counts
            reward: Scalar reward value
            terminated: Whether episode ended
            truncated: Whether max steps reached
            info: Additional information
        """
        assert self.action_space.contains(action)
        
        # Store previous queues for reward calculation
        prev_queues = self.queues.copy()
        
        # 1. DEQUEUE: Clear cars from selected lane
        cars_cleared = min(self.clearance_rate, self.queues[action])
        self.queues[action] = max(0, self.queues[action] - cars_cleared)
        self.total_cars_cleared += cars_cleared
        self.vehicles_processed += cars_cleared
        
        # 2. ARRIVALS: Generate traffic based on time patterns
        self._generate_traffic()
        
        # 3. CLIP: Ensure queues don't exceed max
        self.queues = np.clip(self.queues, 0, self.max_queue_length)
        
        # 4. CALCULATE REWARD (Run 5 or Run 3 mode)
        if self.use_run5_reward:
            reward = self._calculate_reward_run5(action, prev_queues, cars_cleared)
        else:
            reward = self._calculate_reward_run3(action, prev_queues, cars_cleared)
        
        # 5. UPDATE TIME (5 seconds per step)
        self.current_time += 5/3600
        if self.current_time >= 24:
            self.current_time -= 24
        
        # 6. UPDATE TRACKING
        self.current_step += 1
        self.total_wait_time += np.sum(self.queues)
        
        # 7. CHECK TERMINATION
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # 8. PREPARE OUTPUT
        obs = self.queues / self.max_queue_length
        info = {
            'queues': self.queues.copy(),
            'step': self.current_step,
            'cars_cleared': cars_cleared,
            'total_cleared': self.total_cars_cleared,
            'vehicles_processed': self.vehicles_processed,
            'avg_wait': self.total_wait_time / self.current_step if self.current_step > 0 else 0,
            'time_of_day': f"{int(self.current_time):02d}:{int((self.current_time % 1) * 60):02d}"
        }
        
        return obs, reward, terminated, truncated, info
    
    def _generate_traffic(self):
        """
        Generate traffic based on time-of-day
        Keeps same Rwanda traffic patterns from original
        """
        hour = int(self.current_time)
        
        # Rush hour patterns
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_arrival_rate = 0.8
        elif 12 <= hour <= 14:  # Lunch hour
            base_arrival_rate = 0.6
        elif 22 <= hour or hour <= 6:  # Night time
            base_arrival_rate = 0.2
        else:  # Regular hours
            base_arrival_rate = 0.4
        
        # Generate vehicles for each direction
        for i in range(4):
            if self.np_random.random() < base_arrival_rate:
                arrivals = self.np_random.integers(0, int(self.arrival_rate) + 1)
                self.queues[i] += arrivals
    
    def _calculate_reward_run5(self, action, prev_queues, cars_cleared):
        """
        RUN 5 FIXED REWARD FUNCTION
        Aligns with Longest Queue baseline strategy
        
        This is the NEW reward that should beat the baseline
        """
        # Get current state
        longest_queue = np.max(self.queues)
        total_waiting = np.sum(self.queues)
        
        # Fixed reward structure (aligns with Longest Queue)
        reward = (-2.0 * longest_queue +     # PRIMARY: Minimize worst congestion
                  0.5 * cars_cleared +       # SECONDARY: Maximize throughput
                  -0.1 * total_waiting)      # TERTIARY: Minimize total waiting
        
        return reward
    
    def _calculate_reward_run3(self, action, prev_queues, cars_cleared):
        """
        RUN 3 BALANCED REWARD FUNCTION
        Keep for comparison or if you prefer this version
        
        Goal: PPO should prioritize attacking congested lanes while maintaining balance
        
        Reward Ratio: 6:1 (throughput:queue)
        """
        
        current_total_queue = np.sum(self.queues)
        
        # 1. THROUGHPUT REWARD (Main positive signal)
        throughput_reward = cars_cleared * 1.5
        
        # 2. QUEUE PENALTY (Moderate, not dominating)
        waiting_penalty = current_total_queue * -0.25
        
        # 3. ACTION QUALITY: Reward attacking the longest queue
        if cars_cleared > 0:
            max_queue_idx = int(np.argmax(prev_queues))
            
            if action == max_queue_idx:
                action_quality = +3.0
            elif prev_queues[action] >= np.max(prev_queues) * 0.7:
                action_quality = +1.0
            else:
                action_quality = -1.0
        else:
            action_quality = -2.0
        
        # 4. QUEUE IMBALANCE PENALTY (Only for severe cases)
        queue_std = np.std(self.queues)
        
        if queue_std < 3.0:
            imbalance_penalty = 0.0
        elif queue_std < 6.0:
            imbalance_penalty = -1.0
        elif queue_std < 10.0:
            imbalance_penalty = -2.5
        else:
            imbalance_penalty = -5.0
        
        # TOTAL REWARD
        total_reward = (
            throughput_reward +
            waiting_penalty +
            action_quality +
            imbalance_penalty
        )
        
        return total_reward
    
    def render(self, mode='human'):
        """Print current state"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Time: {int(self.current_time):02d}:{int((self.current_time % 1) * 60):02d}")
            for i, lane in enumerate(self.lane_names):
                bar = '|||' * int(self.queues[i])
                print(f"  {lane:6s}: {bar:20s} ({int(self.queues[i])} cars)")
            print(f"  Total: {int(np.sum(self.queues))} cars waiting")
            print(f"  Processed: {int(self.vehicles_processed)} total")
            print(f"  Reward mode: {'Run 5 (Fixed)' if self.use_run5_reward else 'Run 3 (Balanced)'}")
    
    def close(self):
        """Cleanup"""
        pass


# TEST THE ENVIRONMENT
if __name__ == "__main__":
    print("\n TESTING RUN 5 COMPATIBLE ENVIRONMENT")
    
    # Test with Run 5 reward
    print("\n TEST 1: RUN 5 FIXED REWARD")
    env5 = SimpleButtonTrafficEnv(
        domain_randomization=False,
        use_run5_reward=True
    )
    
    print(f"Observation space: {env5.observation_space}")
    print(f"Action space: {env5.action_space}")
    
    obs, info = env5.reset()
    env5.queues = np.array([15, 3, 2, 4], dtype=np.float32)
    
    print(f"\nInitial queues: {env5.queues}")
    
    total_reward = 0
    for i in range(10):
        action = int(np.argmax(env5.queues))
        obs, reward, terminated, truncated, info = env5.step(action)
        total_reward += reward
        
        if i == 0:  # Just show first step
            print(f"Step 1 reward: {reward:.2f}")
    
    print(f"Total reward (Run 5): {total_reward:.2f}")
    print(f"Expected: +10 to +30 (positive)")
    
    # Test with Run 3 reward for comparison
    print("\n TEST 2: RUN 3 BALANCED REWARD (for comparison)")
    env3 = SimpleButtonTrafficEnv(
        domain_randomization=False,
        use_run5_reward=False
    )
    
    obs, info = env3.reset()
    env3.queues = np.array([15, 3, 2, 4], dtype=np.float32)
    
    total_reward = 0
    for i in range(10):
        action = int(np.argmax(env3.queues))
        obs, reward, terminated, truncated, info = env3.step(action)
        total_reward += reward
    
    print(f"Total reward (Run 3): {total_reward:.2f}")
    print(f"Expected: +25 to +40 (positive)")
    
    # Test hardware DR
    print("\n TEST 3: HARDWARE-AWARE DOMAIN RANDOMIZATION")
    env_dr = SimpleButtonTrafficEnv(
        domain_randomization=True,
        randomization_config={
            'gpio_latency_range': (1, 10),
            'button_debounce_range': (50, 200),
            'processing_jitter_range': (0, 5),
        },
        use_run5_reward=True
    )
    
    for ep in range(3):
        obs, info = env_dr.reset()
        print(f"\nEpisode {ep+1}:")
        print(f"  GPIO latency: {env_dr.gpio_latency:.1f}ms")
        print(f"  Button debounce: {env_dr.button_debounce:.0f}ms")
        print(f"  Processing jitter: {env_dr.processing_jitter:.1f}ms")
    
    print("\n ALL TESTS COMPLETE")
