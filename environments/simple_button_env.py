"""
Simplified Button Traffic Environment for Hardware Deployment
FIXED VERSION 3 - Properly balanced reward function

Key Changes from V2 (which failed):
  - Reward rebalanced: penalties no longer dominate
  - Simplified components: removed conflicting objectives
  - Focus on core trade-off: throughput vs queue management
  
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
    
    State Space (4 dimensions):
        - Queue counts for 4 directions: [North, South, East, West]
        - Normalized to [0, 1] range
    
    Action Space (4 discrete actions):
        - 0: Give green to North
        - 1: Give green to South
        - 2: Give green to East
        - 3: Give green to West
    
    Reward (BALANCED VERSION):
        - Core: Throughput (+1.5 per car) vs Queue penalty (-0.25 per car)
        - Ratio: 6:1 (was 13:1 in Run 1, 2.5:1 in Run 2)
        - Bonus: Attack longest queue (+3.0)
        - Penalty: Severe imbalance (-2.0 to -5.0)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 max_queue_length=20,
                 cars_cleared_per_cycle=5,
                 max_arrival_rate=3,
                 domain_randomization=True):
        """
        Args:
            max_queue_length: Max vehicles per lane
            cars_cleared_per_cycle: Vehicles cleared per green cycle
            max_arrival_rate: Max random arrivals per lane per step
            domain_randomization: Enable for sim2real robustness
        """
        super().__init__()
        
        # Parameters
        self.max_queue_length = max_queue_length
        self.cars_cleared_per_cycle = cars_cleared_per_cycle
        self.max_arrival_rate = max_arrival_rate
        self.domain_randomization = domain_randomization
        
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
        
        # Performance tracking
        self.total_cars_cleared = 0
        self.total_wait_time = 0
        self.vehicles_processed = 0
        
        # Time simulation
        self.current_time = 8.0  # Start at 8 AM
        
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
        Makes model robust to real-world variations
        """
        # Randomize clearance rate (±30%)
        self.clearance_rate = self.cars_cleared_per_cycle * \
                             self.np_random.uniform(0.7, 1.3)
        
        # Randomize arrival rate (±40%)
        self.arrival_rate = self.max_arrival_rate * \
                           self.np_random.uniform(0.6, 1.4)
    
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
        
        # 4. CALCULATE REWARD (FIXED VERSION 3)
        reward = self._calculate_reward(action, prev_queues, cars_cleared)
        
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
    
    def _calculate_reward(self, action, prev_queues, cars_cleared):
        """
        BALANCED REWARD FUNCTION (Version 3)
        
        Goal: PPO should prioritize attacking congested lanes while maintaining balance
        
        Design Philosophy:
        - Simple is better: Only 4 components (was 6)
        - Rewards should dominate penalties in normal operation
        - Penalties kick in for bad behavior (severe imbalance, starvation)
        
        Reward Ratio: 6:1 (throughput:queue)
        - Run 1: 13:1 → PPO ignored queues
        - Run 2: 2.5:1 → PPO over-prioritized queues
        - Run 3: 6:1 → Balanced middle ground
        """
        
        current_total_queue = np.sum(self.queues)
        
        # 1. THROUGHPUT REWARD (Main positive signal)
        # Increased from 1.0 to 1.5 for stronger positive signal
        throughput_reward = cars_cleared * 1.5
        
        # 2. QUEUE PENALTY (Moderate, not dominating)
        # Reduced from -0.4 to -0.25 so it doesn't overwhelm throughput
        waiting_penalty = current_total_queue * -0.25
        
        # 3. ACTION QUALITY: Reward attacking the longest queue
        if cars_cleared > 0:
            max_queue_idx = int(np.argmax(prev_queues))
            
            if action == max_queue_idx:
                # Perfect! Attacked the most congested lane
                action_quality = +3.0
            elif prev_queues[action] >= np.max(prev_queues) * 0.7:
                # Good enough, attacked a busy lane
                action_quality = +1.0
            else:
                # Suboptimal: wasted capacity on a short queue
                action_quality = -1.0
        else:
            # Tried to clear from empty lane
            action_quality = -2.0
        
        # 4. QUEUE IMBALANCE PENALTY (Only for severe cases)
        # Only penalize when imbalance is really bad
        queue_std = np.std(self.queues)
        
        if queue_std < 3.0:
            # Good balance across lanes
            imbalance_penalty = 0.0
        elif queue_std < 6.0:
            # Moderate imbalance, small penalty
            imbalance_penalty = -1.0
        elif queue_std < 10.0:
            # Bad imbalance
            imbalance_penalty = -2.5
        else:
            # Severe imbalance (one lane starved while others overflow)
            imbalance_penalty = -5.0
        
        # TOTAL REWARD
        total_reward = (
            throughput_reward +      # +7.5 typical (5 cars * 1.5)
            waiting_penalty +        # -5.0 typical (20 queue * -0.25)
            action_quality +         # +3.0 or -1.0
            imbalance_penalty        # 0 to -5.0
        )
        
        # Expected range in normal operation: +2.5 to +5.5
        # Expected range with problems: -3.0 to +1.0
        
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
    print("\n TESTING BALANCED REWARD FUNCTION (V3)")
    
    env = SimpleButtonTrafficEnv(domain_randomization=False)
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"  Actions: 0=North, 1=South, 2=East, 3=West")
    
    # Test the North Heavy scenario
    print("\n TESTING NORTH HEAVY SCENARIO")
    print("Run 1: PPO scored 87 (ignored queues)")
    print("Run 2: PPO scored -408 (over-penalized queues)")
    print("Run 3: Should score 200-400 (balanced)")
    
    obs, info = env.reset()
    env.queues = np.array([15, 3, 2, 4], dtype=np.float32)
    
    print(f"\nInitial queues: {env.queues}")
    
    total_reward = 0
    
    for i in range(10):
        # Greedy: attack longest queue
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
    
    print("\n EXPECTED BEHAVIOR:")
    print("  Rewards should be mostly POSITIVE (+2 to +5 per step)")
    print("  Total over 10 steps: +25 to +40")
    print("  This balances throughput and queue management")