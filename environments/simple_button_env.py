"""
Simplified Button Traffic Environment for Hardware Deployment
Adapted from TrafficJunctionEnv for 4-button prototype

Key Changes from Original:
  - State: 113 dims to 4 dims (just queue counts)
  - Actions: 9 complex to 4 simple (which lane gets green)
  - Removed: Road grids, timers, light states (hardware doesn't have these)
  - Added: Domain randomization for sim2real transfer
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional
from enum import Enum

class TrafficDirection(Enum):
    """Keep same directions as original environment"""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class SimpleButtonTrafficEnv(gym.Env):
    """
    Hardware-matching environment for button/LED prototype.
    Simplified from original TrafficJunctionEnv.
    
    State Space (4 dimensions):
        - Queue counts for 4 directions: [North, South, East, West]
        - Normalized to [0, 1] range
    
    Action Space (4 discrete actions):
        - 0: Give green to North
        - 1: Give green to South
        - 2: Give green to East
        - 3: Give green to West
    
    Reward:
        - Simplified from original complex reward
        - Focus: Minimize total waiting vehicles
        - Bonus: Clear congested lanes
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 max_queue_length=20,
                 cars_cleared_per_cycle=5,
                 max_arrival_rate=3,
                 domain_randomization=True):
        """
        Args:
            max_queue_length: Max vehicles per lane (same as original)
            cars_cleared_per_cycle: Vehicles cleared per green cycle
            max_arrival_rate: Max random arrivals per lane per step
            domain_randomization: Enable for sim2real robustness
        """
        super().__init__()
        
        # Parameters (adapted from original environment)
        self.max_queue_length = max_queue_length
        self.cars_cleared_per_cycle = cars_cleared_per_cycle
        self.max_arrival_rate = max_arrival_rate
        self.domain_randomization = domain_randomization
        
        # State: Just 4 queue counts (MUCH simpler than original 113 dims)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(4,),  # Was 113 in original!
            dtype=np.float32
        )
        
        # Action: Select 1 of 4 lanes (simplified from original 9 actions)
        self.action_space = spaces.Discrete(4)
        
        # Lane names (same as original)
        self.lane_names = ['North', 'South', 'East', 'West']
        
        # Internal state
        self.queues = np.zeros(4, dtype=np.float32)
        self.current_step = 0
        self.max_steps = 200  # Episode length
        
        # Domain randomization parameters
        self.clearance_rate = cars_cleared_per_cycle
        self.arrival_rate = max_arrival_rate
        
        # Performance tracking (similar to original)
        self.total_cars_cleared = 0
        self.total_wait_time = 0
        self.vehicles_processed = 0  # Same as original tracking
        
        # Time simulation (simplified from original)
        self.current_time = 8.0  # Start at 8 AM like original
        
    def reset(self, seed=None, options=None):
        """Reset environment (similar to original but simplified)"""
        super().reset(seed=seed)
        
        # Initialize with random queue lengths (like original)
        self.queues = self.np_random.integers(0, 10, size=4).astype(np.float32)
        
        self.current_step = 0
        self.total_cars_cleared = 0
        self.total_wait_time = 0
        self.vehicles_processed = 0
        self.current_time = 8.0
        
        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._randomize_parameters()
        
        # Return normalized state (like original but simpler)
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
        Execute one step (simplified from original complex step)
        
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
        
        # Store previous queue for reward calculation
        prev_total_queue = np.sum(self.queues)
        
        # 1. DEQUEUE: Clear cars from selected lane
        cars_cleared = min(self.clearance_rate, self.queues[action])
        self.queues[action] = max(0, self.queues[action] - cars_cleared)
        self.total_cars_cleared += cars_cleared
        self.vehicles_processed += cars_cleared  # Same as original tracking
        
        # 2. ARRIVALS: Generate traffic based on time patterns (from original)
        self._generate_traffic()
        
        # 3. CLIP: Ensure queues don't exceed max
        self.queues = np.clip(self.queues, 0, self.max_queue_length)
        
        # 4. CALCULATE REWARD (simplified from original complex reward)
        reward = self._calculate_reward(action, prev_total_queue, cars_cleared)
        
        # 5. UPDATE TIME (like original: 5 seconds per step)
        self.current_time += 5/3600  # 5 seconds in hours
        if self.current_time >= 24:
            self.current_time -= 24
        
        # 6. UPDATE TRACKING
        self.current_step += 1
        self.total_wait_time += np.sum(self.queues)
        
        # 7. CHECK TERMINATION (simplified from original)
        terminated = False  # Continuous traffic simulation
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
        Generate traffic based on time-of-day (FROM ORIGINAL ENVIRONMENT)
        Keeps same Rwanda traffic patterns
        """
        hour = int(self.current_time)
        
        # Same rush hour patterns as original
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
    
    def _calculate_reward(self, action, prev_total_queue, cars_cleared):
        """        
        Goal: Positive rewards for good performance, negative for bad
                
        New design:
        - Base reward: +1 per car cleared (direct throughput measure)
        - Queue penalty: -0.1 per car waiting (small ongoing cost)
        - Efficiency bonus: +5 for keeping all queues short
        - Congestion penalty: Only for extreme cases
        
        Expected outcomes:
        - Good agent: +50 to +100 per episode (positive!)
        - Bad agent: -200 to -500 per episode (negative)
        - Random agent: Around 0 (neutral)
        """
        
        current_total_queue = np.sum(self.queues)
        
        # 1. PRIMARY REWARD: Throughput (Always Positive!)
        # Reward every car that successfully clears the intersection
        # This is the MAIN measure of success
        throughput_reward = cars_cleared * 2.0  # +2 per car cleared
        
        # 2. QUEUE MAINTENANCE COST (Small Penalty)
        # Small penalty for cars waiting (cost of congestion)
        # Much smaller than the throughput reward
        waiting_penalty = current_total_queue * -0.15
        
        # 3. EFFICIENCY BONUS (Positive for Good Flow)
        # Bonus for maintaining good flow across all lanes
        if current_total_queue <= 5:
            efficiency_bonus = 10.0  # Excellent flow!
        elif current_total_queue <= 10:
            efficiency_bonus = 5.0   # Good flow
        elif current_total_queue <= 15:
            efficiency_bonus = 0.0   # Acceptable
        else:
            efficiency_bonus = -5.0  # Only penalize extreme congestion
        
        # 4. LANE BALANCE BONUS (Encourage Fairness)
        # Bonus for keeping all lanes relatively balanced
        # Prevents ignoring any single lane
        queue_std = np.std(self.queues)
        if queue_std < 2.0:
            balance_bonus = 3.0  # Very balanced
        elif queue_std < 4.0:
            balance_bonus = 1.0  # Reasonably balanced
        else:
            balance_bonus = 0.0  # Unbalanced (no penalty, just no bonus)
        
        # 5. ACTION QUALITY BONUS
        # Bonus for clearing the busiest lane
        if cars_cleared > 0:
            # Was this lane relatively congested?
            lane_congestion = self.queues[action] / (np.sum(self.queues) + 1e-6)
            if lane_congestion > 0.3:  # Cleared a busy lane
                action_quality_bonus = 2.0
            else:
                action_quality_bonus = 0.5
        else:
            action_quality_bonus = 0.0
        
        # TOTAL REWARD CALCULATION
        total_reward = (
            throughput_reward +      # Main positive component
            waiting_penalty +        # Small negative component
            efficiency_bonus +       # Conditional bonus/penalty
            balance_bonus +          # Fairness bonus
            action_quality_bonus     # Smart decision bonus
        )
        
        # EXPECTED RANGES (for validation)
        # Good episode (237 cars cleared, avg 10 waiting):
        #   Throughput:  237 * 2.0 = +474
        #   Waiting:     10 * -0.15 * 50 steps = -75
        #   Bonuses:     ~5 * 50 = +250
        #   Total:       ~+649 (POSITIVE!)
        #
        # Bad episode (150 cars cleared, avg 20 waiting):
        #   Throughput:  150 * 2.0 = +300
        #   Waiting:     20 * -0.15 * 50 steps = -150
        #   Penalties:   -5 * 50 = -250
        #   Total:       ~-100 (NEGATIVE)
        
        return total_reward    
    
    def render(self, mode='human'):
        """Print current state (simplified from original)"""
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
    print("\n TESTING SIMPLIFIED BUTTON ENVIRONMENT")
    
    # Create environment with domain randomization
    env = SimpleButtonTrafficEnv(domain_randomization=True)
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"  Actions: 0=North, 1=South, 2=East, 3=West")
    
    # Reset and show initial state
    obs, info = env.reset()
    print(f"\nInitial observation: {obs}")
    print(f"Initial queues: {info['queues']}")
    
    # Run a few test steps
    print("\n RUNNING 10 TEST STEPS")
    
    total_reward = 0
    
    for i in range(10):
        # Select action: prioritize longest queue
        action = int(np.argmax(obs))
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display
        lane_name = env.lane_names[action]
        print(f"\nStep {i+1}:")
        print(f"  Action: Give green to {lane_name}")
        print(f"  Queues: {info['queues']}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Cleared: {info['cars_cleared']:.1f} cars")
        
        if terminated or truncated:
            print("\nEpisode ended")
            break
    
    print("\n TEST COMPLETE")
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Total vehicles processed: {info['vehicles_processed']}")
    print("\n Environment is ready for training!")
