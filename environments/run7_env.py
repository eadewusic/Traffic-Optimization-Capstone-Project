#!/usr/bin/env python3
"""
Incorporates all lessons learned:
- Comparative reward (proven to work - 13k improvement)
- Corrected scaling (8.0 win, 5.0 loss)
- Semi-deterministic arrivals
- 2 actions only
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Run7TrafficEnv(gym.Env):
    """
    FINAL Run 7 Environment
    
    Key features:
    1. Comparative reward: Explicit bonus for beating baseline
    2. Balanced scaling: Win ×8, Loss ×5 (not too harsh)
    3. Reduced stochasticity: Semi-deterministic arrivals
    4. Simple actions: Only N/S and E/W
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, max_queue_length=20, cars_cleared_per_cycle=5):
        super().__init__()
        
        self.max_queue_length = max_queue_length
        self.cars_cleared_per_cycle = cars_cleared_per_cycle
        
        # Reduced stochasticity
        self.arrival_rate = 0.15
        
        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=max_queue_length, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # Only N/S, E/W
        
        # State
        self.queues = None
        self.current_step = 0
        self.max_steps = 200
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'initial_queues' in options:
            self.queues = np.array(options['initial_queues'], dtype=np.float32)
        else:
            self.queues = np.random.uniform(2, 8, size=4).astype(np.float32)
        
        self.current_step = 0
        return self.queues.copy(), {}
    
    def step(self, action):
        self.current_step += 1
        prev_queues = self.queues.copy()
        
        # What would baseline do?
        baseline_action = self._baseline_action(prev_queues)
        
        # Execute PPO's action
        ppo_cleared = self._execute_action(action)
        
        # Simulate baseline
        baseline_cleared = self._simulate_baseline_clearing(baseline_action, prev_queues)
        
        # Add arrivals
        self._add_arrivals()
        
        # Clip
        self.queues = np.clip(self.queues, 0, self.max_queue_length)
        
        # OPTIMIZED COMPARATIVE REWARD
        reward = self._calculate_comparative_reward(
            ppo_cleared, 
            baseline_cleared, 
            prev_queues,
            action
        )
        
        terminated = self.current_step >= self.max_steps
        
        info = {
            'cars_cleared': int(ppo_cleared),
            'baseline_would_clear': int(baseline_cleared),
            'beat_baseline': ppo_cleared > baseline_cleared,
            'total_waiting': float(np.sum(self.queues)),
            'longest_queue': float(np.max(self.queues))
        }
        
        return self.queues.copy(), reward, terminated, False, info
    
    def _baseline_action(self, queues):
        """Longest-queue-first baseline"""
        ns_total = queues[0] + queues[1]
        ew_total = queues[2] + queues[3]
        return 0 if ns_total > ew_total else 1
    
    def _simulate_baseline_clearing(self, action, queues):
        """How many cars would baseline clear?"""
        if action == 0:
            return min(queues[0], self.cars_cleared_per_cycle) + min(queues[1], self.cars_cleared_per_cycle)
        else:
            return min(queues[2], self.cars_cleared_per_cycle) + min(queues[3], self.cars_cleared_per_cycle)
    
    def _execute_action(self, action):
        """Execute action"""
        cars_cleared = 0
        
        if action == 0:  # N/S
            cars_from_north = min(self.queues[0], self.cars_cleared_per_cycle)
            cars_from_south = min(self.queues[1], self.cars_cleared_per_cycle)
            self.queues[0] -= cars_from_north
            self.queues[1] -= cars_from_south
            cars_cleared = cars_from_north + cars_from_south
            
        elif action == 1:  # E/W
            cars_from_east = min(self.queues[2], self.cars_cleared_per_cycle)
            cars_from_west = min(self.queues[3], self.cars_cleared_per_cycle)
            self.queues[2] -= cars_from_east
            self.queues[3] -= cars_from_west
            cars_cleared = cars_from_east + cars_from_west
        
        return cars_cleared
    
    def _add_arrivals(self):
        """
        Semi-deterministic arrivals
        - 1 guaranteed arrival per step (rotating lane)
        - 10% chance of additional arrival to other lanes
        """
        # Deterministic component
        primary_lane = self.current_step % 4
        self.queues[primary_lane] = min(
            self.queues[primary_lane] + 1,
            self.max_queue_length
        )
        
        # Stochastic component (minimal)
        for i in range(4):
            if i != primary_lane:
                if np.random.random() < 0.10:
                    self.queues[i] = min(
                        self.queues[i] + 1,
                        self.max_queue_length
                    )
    
    def _calculate_comparative_reward(self, ppo_cleared, baseline_cleared, prev_queues, ppo_action):
        """
        OPTIMIZED COMPARATIVE REWARD
        
        Based on friend's analysis and our testing:
        - Win bonus: ×8.0 (was ×20, too aggressive)
        - Loss penalty: ×5.0 (was ×15, too harsh)
        - This creates learnable gradient from -5k to +600
        """
        
        # 1. Base throughput
        throughput_reward = ppo_cleared * 3.0
        
        # 2. OPTIMIZED COMPARATIVE COMPONENT
        differential = ppo_cleared - baseline_cleared
        
        if differential > 0:
            # Beat baseline - moderate bonus
            comparative_bonus = differential * 8.0  # Reduced from 20
        elif differential < 0:
            # Worse than baseline - gentler penalty
            comparative_penalty = differential * 5.0  # Reduced from 15
            comparative_bonus = 0
        else:
            # Tied
            comparative_bonus = 2.0
            comparative_penalty = 0
        
        # 3. Strategic alignment
        longest_queue_idx = int(np.argmax(prev_queues))
        if ppo_cleared > 0:
            if ppo_action == 0 and longest_queue_idx in [0, 1]:
                strategic_bonus = 5.0
            elif ppo_action == 1 and longest_queue_idx in [2, 3]:
                strategic_bonus = 5.0
            else:
                strategic_bonus = 0
        else:
            strategic_bonus = 0
        
        # 4. Congestion penalties (secondary)
        longest_queue = np.max(self.queues)
        total_waiting = np.sum(self.queues)
        
        congestion_penalty = (
            -0.5 * longest_queue +
            -0.1 * total_waiting
        )
        
        # TOTAL
        if differential >= 0:
            total_reward = (
                throughput_reward +
                comparative_bonus +
                strategic_bonus +
                congestion_penalty
            )
        else:
            total_reward = (
                throughput_reward +
                comparative_penalty +
                congestion_penalty
            )
        
        return total_reward
    
    def render(self):
        if self.render_mode == 'human':
            print(f"Step {self.current_step}: Queues={self.queues}")