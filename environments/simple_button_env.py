"""
Simplified Button Traffic Environment - RUN 4 VERSION
Adjusted throughput weight for better performance

Changes from Run 3:
- Throughput weight: 1.5 → 1.75
- Ratio: 6:1 → 7:1
- Everything else same
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
    RUN 4 VERSION with optimized reward balance.
    
    State Space (4 dimensions):
        - Queue counts for 4 directions: [North, South, East, West]
        - Normalized to [0, 1] range
    
    Action Space (4 discrete actions):
        - 0: Give green to North
        - 1: Give green to South
        - 2: Give green to East
        - 3: Give green to West
    
    Reward (RUN 4 - Optimized):
        - Throughput: +1.75 per car (was +1.5 in Run 3)
        - Queue penalty: -0.25 per car (same as Run 3)
        - Ratio: 7:1 (was 6:1 in Run 3)
        - Action bonus: +3.0 for attacking longest queue
        - Imbalance penalty: -5.0 for severe cases
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 max_queue_length=20,
                 cars_cleared_per_cycle=5,
                 max_arrival_rate=3,
                 domain_randomization=False):  # Default FALSE for Run 4
        """
        Args:
            max_queue_length: Max vehicles per lane
            cars_cleared_per_cycle: Vehicles cleared per green cycle
            max_arrival_rate: Max random arrivals per lane per step
            domain_randomization: Enable for sim2real (DEFAULT FALSE for Run 4)
        """
        super().__init__()
        
        # Parameters
        self.max_queue_length = max_queue_length
        self.cars_cleared_per_cycle = cars_cleared_per_cycle
        self.max_arrival_rate = max_arrival_rate
        self.domain_randomization = domain_randomization
        
        # State: 4 queue counts
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
        
        # Performance tracking
        self.total_cars_cleared = 0
        self.total_wait_time = 0
        self.vehicles_processed = 0
        
        # Time simulation
        self.current_time = 8.0
        
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
        """
        self.clearance_rate = self.cars_cleared_per_cycle * \
                             self.np_random.uniform(0.7, 1.3)
        
        self.arrival_rate = self.max_arrival_rate * \
                           self.np_random.uniform(0.6, 1.4)
    
    def step(self, action):
        """Execute one step"""
        assert self.action_space.contains(action)
        
        # Store previous queues for reward calculation
        prev_queues = self.queues.copy()
        
        # 1. DEQUEUE: Clear cars from selected lane
        cars_cleared = min(self.clearance_rate, self.queues[action])
        self.queues[action] = max(0, self.queues[action] - cars_cleared)
        self.total_cars_cleared += cars_cleared
        self.vehicles_processed += cars_cleared
        
        # 2. ARRIVALS: Generate traffic
        self._generate_traffic()
        
        # 3. CLIP: Ensure queues don't exceed max
        self.queues = np.clip(self.queues, 0, self.max_queue_length)
        
        # 4. CALCULATE REWARD (RUN 4 VERSION)
        reward = self._calculate_reward(action, prev_queues, cars_cleared)
        
        # 5. UPDATE TIME
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
        """Generate traffic based on time-of-day"""
        hour = int(self.current_time)
        
        # Rush hour patterns
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_arrival_rate = 0.8
        elif 12 <= hour <= 14:
            base_arrival_rate = 0.6
        elif 22 <= hour or hour <= 6:
            base_arrival_rate = 0.2
        else:
            base_arrival_rate = 0.4
        
        # Generate vehicles
        for i in range(4):
            if self.np_random.random() < base_arrival_rate:
                arrivals = self.np_random.integers(0, int(self.arrival_rate) + 1)
                self.queues[i] += arrivals
    
    def _calculate_reward(self, action, prev_queues, cars_cleared):
        """
        RUN 4 REWARD FUNCTION - Optimized Balance
        
        Key change from Run 3: Throughput weight 1.5 → 1.75
        This makes ratio 7:1 instead of 6:1
        
        Goal: Encourage more throughput while maintaining queue balance
        """
        
        current_total_queue = np.sum(self.queues)
        
        # 1. THROUGHPUT REWARD (INCREASED from 1.5 to 1.75)
        throughput_reward = cars_cleared * 1.75  # Changed!
        
        # 2. QUEUE PENALTY (Same as Run 3)
        waiting_penalty = current_total_queue * -0.25
        
        # 3. ACTION QUALITY: Reward attacking longest queue
        if cars_cleared > 0:
            max_queue_idx = int(np.argmax(prev_queues))
            
            if action == max_queue_idx:
                action_quality = +3.0  # Perfect choice
            elif prev_queues[action] >= np.max(prev_queues) * 0.7:
                action_quality = +1.0  # Good choice
            else:
                action_quality = -1.0  # Suboptimal
        else:
            action_quality = -2.0  # Wasted cycle
        
        # 4. QUEUE IMBALANCE PENALTY (Only for severe cases)
        queue_std = np.std(self.queues)
        
        if queue_std < 3.0:
            imbalance_penalty = 0.0  # Good balance
        elif queue_std < 6.0:
            imbalance_penalty = -1.0  # Moderate imbalance
        elif queue_std < 10.0:
            imbalance_penalty = -2.5  # Bad imbalance
        else:
            imbalance_penalty = -5.0  # Severe imbalance
        
        # TOTAL REWARD
        total_reward = (
            throughput_reward +      # +8.75 typical (5 * 1.75)
            waiting_penalty +        # -5.0 typical (20 * -0.25)
            action_quality +         # +3.0 or -1.0
            imbalance_penalty        # 0 to -5.0
        )
        
        # Expected range: +3 to +7 in normal operation (vs Run 3's +2.5 to +5.5)
        
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
    
    def close(self):
        """Cleanup"""
        pass


# TEST THE ENVIRONMENT
if __name__ == "__main__":
    print("\n TESTING RUN 4 REWARD FUNCTION")
    print("Change: Throughput weight 1.5 → 1.75")
    print("Ratio: 6:1 → 7:1\n")
    
    env = SimpleButtonTrafficEnv(domain_randomization=False)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test North Heavy scenario
    print("\n TESTING NORTH HEAVY SCENARIO")
    
    obs, info = env.reset()
    env.queues = np.array([15, 3, 2, 4], dtype=np.float32)
    
    print(f"Initial queues: {env.queues}")
    
    total_reward = 0
    
    for i in range(10):
        action = int(np.argmax(env.queues))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        lane_name = env.lane_names[action]
        print(f"\nStep {i+1}: Attack {lane_name}")
        print(f"  Queues: {info['queues']}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Cleared: {info['cars_cleared']:.1f} cars")
        
        if terminated or truncated:
            break
    
    print("\n TEST COMPLETE")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average per step: {total_reward/10:.2f}")
    print("\nExpected: Slightly higher than Run 3 due to increased throughput weight")