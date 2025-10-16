#!/usr/bin/env python3
"""
Simplified Button Traffic Environment - RUN 6
PROPERLY BALANCED REWARD FUNCTION - Positive for Good Performance

Key Changes from Run 5:
1. FIXED REWARD SCALE: Good performance = POSITIVE rewards
2. KEPT STRATEGIC ALIGNMENT: Still prioritizes longest queue (what worked in Run 5)
3. REBALANCED WEIGHTS: Throughput rewards dominate congestion penalties
4. ACADEMIC STANDARD: Positive = success, Negative = failure

The Strategy from Run 5 (that beat baseline):
- Prioritize reducing longest queue
- Maintain good throughput
- Keep overall fairness

The Fix for Run 6:
- Same strategy, but reward scale shows success as positive
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleButtonTrafficEnv(gym.Env):
    """
    Hardware-matching environment with PROPERLY BALANCED rewards.
    
    Observation: [queue_north, queue_south, queue_east, queue_west]
    Action: 0=N/S green, 1=E/W green, 2=all red (transition), 3=emergency all red
    
    NEW REWARD FUNCTION (Run 6 - Properly Balanced):
    - Throughput rewards DOMINATE (primary positive signal)
    - Congestion penalties are SMALLER (keep agent honest)
    - Strategic bonuses for attacking longest queue
    - Result: POSITIVE rewards for good performance
    
    Expected reward ranges:
    - Excellent performance: +8 to +15 per step
    - Good performance: +3 to +8 per step
    - Acceptable: 0 to +3 per step
    - Poor: -5 to 0 per step
    - Bad: -10 to -5 per step
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        max_queue_length=20,
        cars_cleared_per_cycle=5,
        max_arrival_rate=3,
        domain_randomization=False,
        randomization_config=None,
        render_mode=None
    ):
        super().__init__()
        
        # Environment parameters
        self.max_queue_length = max_queue_length
        self.cars_cleared_per_cycle = cars_cleared_per_cycle
        self.max_arrival_rate = max_arrival_rate
        self.render_mode = render_mode
        
        # Domain randomization setup
        self.domain_randomization = domain_randomization
        self.dr_config = randomization_config if randomization_config else self._default_dr_config()
        
        # Properly balanced reward weights for Run 6
        self.reward_weights = {
            'throughput': 3.0,           # PRIMARY: Strong positive signal
            'longest_queue': -0.4,       # Secondary: Light penalty (was -2.0 in Run 5!)
            'total_waiting': -0.05,      # Tertiary: Very light penalty (was -0.1)
            'strategic_bonus': 5.0,      # Bonus: Reward attacking longest queue
            'empty_action_penalty': -3.0 # Penalty: Wasting cycles on empty lanes
        }
        
        # Spaces
        self.observation_space = spaces.Box(
            low=0,
            high=max_queue_length,
            shape=(4,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(4)
        
        # State variables
        self.queues = None
        self.current_step = 0
        self.max_steps = 200
        
        # Domain randomization parameters (set per episode)
        self.arrival_rate = 0.3
        self.yellow_duration = 3
        self.queue_capacity = 10
        self.gpio_latency = 0
        self.button_debounce = 0
        self.processing_jitter = 0
        
    def _default_dr_config(self):
        """Default hardware-aware domain randomization configuration"""
        return {
            # Traffic variability
            'arrival_rate_range': (0.15, 0.45),
            'queue_capacity_range': (8, 12),
            'yellow_duration_range': (2, 4),
            
            # Hardware delays
            'gpio_latency_range': (1, 10),
            'button_debounce_range': (50, 200),
            'processing_jitter_range': (0, 5),
        }
    
    def _randomize_parameters(self):
        """Apply domain randomization if enabled"""
        if not self.domain_randomization:
            self.arrival_rate = 0.3
            self.yellow_duration = 3
            self.queue_capacity = 10
            self.gpio_latency = 0
            self.button_debounce = 0
            self.processing_jitter = 0
            return
        
        # Randomize traffic parameters
        self.arrival_rate = np.random.uniform(*self.dr_config['arrival_rate_range'])
        self.queue_capacity = np.random.uniform(*self.dr_config['queue_capacity_range'])
        self.yellow_duration = np.random.uniform(*self.dr_config['yellow_duration_range'])
        
        # Randomize hardware delays
        self.gpio_latency = np.random.uniform(*self.dr_config['gpio_latency_range'])
        self.button_debounce = np.random.uniform(*self.dr_config['button_debounce_range'])
        self.processing_jitter = np.random.uniform(*self.dr_config['processing_jitter_range'])
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Apply domain randomization
        self._randomize_parameters()
        
        # Initialize queues with some initial traffic
        if options and 'initial_queues' in options:
            self.queues = np.array(options['initial_queues'], dtype=np.float32)
        else:
            self.queues = np.random.uniform(2, 8, size=4).astype(np.float32)
        
        self.current_step = 0
        
        return self.queues.copy(), {}
    
    def step(self, action):
        """Execute one time step"""
        self.current_step += 1
        
        # Store previous state for reward calculation
        prev_queues = self.queues.copy()
        
        # Simulate hardware delays (if DR enabled)
        effective_action = self._apply_hardware_delays(action)
        
        # Execute action and clear cars
        cars_cleared = self._execute_action(effective_action)
        
        # Add new arrivals to queues
        self._add_arrivals()
        
        # Clip queues to max capacity
        self.queues = np.clip(self.queues, 0, self.max_queue_length)
        
        # NEW: Calculate reward using properly balanced function
        reward = self._calculate_balanced_reward(effective_action, prev_queues, cars_cleared)
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Info dictionary
        info = {
            'cars_cleared': cars_cleared,
            'total_cleared': int(cars_cleared),
            'longest_queue': float(np.max(self.queues)),
            'total_waiting': float(np.sum(self.queues)),
            'step': self.current_step
        }
        
        return self.queues.copy(), reward, terminated, truncated, info
    
    def _apply_hardware_delays(self, action):
        """Simulate hardware delays"""
        if not self.domain_randomization:
            return action
        
        total_delay = self.gpio_latency + self.button_debounce + self.processing_jitter
        
        # If total delay is high, there's a chance the action is "stuck"
        if total_delay > 150 and np.random.random() < 0.1:
            return 2  # Transition state (all red)
        
        return action
    
    def _execute_action(self, action):
        """Execute traffic light action and clear cars"""
        cars_cleared = 0
        
        if action == 0:  # N/S green
            cars_from_north = min(self.queues[0], self.cars_cleared_per_cycle)
            cars_from_south = min(self.queues[1], self.cars_cleared_per_cycle)
            self.queues[0] -= cars_from_north
            self.queues[1] -= cars_from_south
            cars_cleared = cars_from_north + cars_from_south
            
        elif action == 1:  # E/W green
            cars_from_east = min(self.queues[2], self.cars_cleared_per_cycle)
            cars_from_west = min(self.queues[3], self.cars_cleared_per_cycle)
            self.queues[2] -= cars_from_east
            self.queues[3] -= cars_from_west
            cars_cleared = cars_from_east + cars_from_west
            
        elif action == 2:  # Yellow / All red (transition)
            cars_cleared = 0
            
        elif action == 3:  # Emergency all red
            cars_cleared = 0
        
        return cars_cleared
    
    def _add_arrivals(self):
        """Add new cars to queues based on arrival rate"""
        for i in range(4):
            if np.random.random() < self.arrival_rate:
                arrivals = np.random.randint(1, self.max_arrival_rate + 1)
                self.queues[i] = min(self.queues[i] + arrivals, self.max_queue_length)
    
    def _calculate_balanced_reward(self, action, prev_queues, cars_cleared):
        """
        RUN 6 PROPERLY BALANCED REWARD FUNCTION
        
        Design Philosophy:
        - Throughput rewards DOMINATE (make rewards naturally positive)
        - Congestion penalties are SMALLER (keep agent honest but don't overwhelm)
        - Strategic bonuses reward smart behavior (attack longest queue)
        - Result: POSITIVE for good performance, NEGATIVE for bad
        
        Expected Reward Examples:
        
        EXCELLENT (Attack longest queue, clear 5 cars, queues ~15):
          +3.0*5 (throughput) + 5.0 (bonus) - 0.4*15 (longest) - 0.05*25 (total)
          = +15 + 5 - 6 - 1.25 = +12.75 POSITIVE
        
        GOOD (Attack busy queue, clear 5 cars, queues ~20):
          +3.0*5 (throughput) + 0 (no bonus) - 0.4*20 (longest) - 0.05*30 (total)
          = +15 + 0 - 8 - 1.5 = +5.5 POSITIVE
        
        ACCEPTABLE (Clear from short queue, queues ~15):
          +3.0*3 (throughput) + 0 (no bonus) - 0.4*15 (longest) - 0.05*25 (total)
          = +9 + 0 - 6 - 1.25 = +1.75 POSITIVE
        
        POOR (Waste cycle on empty lane, queues growing ~25):
          +3.0*0 (throughput) - 3.0 (penalty) - 0.4*25 (longest) - 0.05*35 (total)
          = 0 - 3 - 10 - 1.75 = -14.75 NEGATIVE
        
        Mathematical Justification:
        - Throughput coefficient (3.0) is 7.5x the longest queue penalty (0.4)
        - Clearing 5 cars gives +15, which easily overcomes typical congestion (-6 to -10)
        - Only truly bad behavior (clearing 0, ignoring congestion) gets negative
        - Strategic bonus (+5) rewards alignment with Longest Queue baseline
        """
        
        # Current state
        longest_queue = np.max(self.queues)
        total_waiting = np.sum(self.queues)
        
        # 1. THROUGHPUT REWARD (Primary positive signal - DOMINANT)
        throughput_reward = self.reward_weights['throughput'] * cars_cleared
        
        # 2. LONGEST QUEUE PENALTY (Secondary - keep strategy aligned)
        # This keeps the agent focused on attacking congestion (what worked in Run 5)
        # But penalty is MUCH smaller now (0.4 vs 2.0 in Run 5)
        longest_queue_penalty = self.reward_weights['longest_queue'] * longest_queue
        
        # 3. TOTAL WAITING PENALTY (Tertiary - very light fairness check)
        total_waiting_penalty = self.reward_weights['total_waiting'] * total_waiting
        
        # 4. STRATEGIC BONUS (Reward smart behavior)
        # Give bonus for attacking the longest queue (aligns with baseline strategy)
        if cars_cleared > 0:
            max_queue_idx = int(np.argmax(prev_queues))
            
            # Map action to lanes: 0=N/S (lanes 0,1), 1=E/W (lanes 2,3)
            if action == 0:
                action_affects_lanes = [0, 1]
            elif action == 1:
                action_affects_lanes = [2, 3]
            else:
                action_affects_lanes = []
            
            # Check if we attacked the longest queue
            if max_queue_idx in action_affects_lanes:
                strategic_bonus = self.reward_weights['strategic_bonus']
            else:
                strategic_bonus = 0.0
        else:
            # Cleared nothing - penalty for wasting a cycle
            strategic_bonus = self.reward_weights['empty_action_penalty']
        
        # TOTAL REWARD
        total_reward = (
            throughput_reward +         # +15 typical (5 cars * 3.0)
            longest_queue_penalty +     # -6 typical (15 queue * -0.4)
            total_waiting_penalty +     # -1.25 typical (25 total * -0.05)
            strategic_bonus             # +5 or 0 or -3
        )
        
        # Expected range:
        # Good performance: +5 to +15 per step (POSITIVE!)
        # Poor performance: -5 to -15 per step (NEGATIVE!)
        
        return total_reward
    
    def render(self):
        """Simple text rendering of current state"""
        if self.render_mode == 'human':
            print(f"\nStep {self.current_step}/{self.max_steps}")
            print(f"Queues: N={self.queues[0]:.1f}, S={self.queues[1]:.1f}, "
                  f"E={self.queues[2]:.1f}, W={self.queues[3]:.1f}")
            print(f"Longest queue: {np.max(self.queues):.1f}")
            print(f"Total waiting: {np.sum(self.queues):.1f}")


# TEST THE ENVIRONMENT
if __name__ == "__main__":
    print("TESTING RUN 6 ENVIRONMENT - PROPERLY BALANCED REWARDS")
    
    # Test 1: Verify positive rewards for good performance
    print("\nTest 1: Good Performance Should Give POSITIVE Rewards")
    
    env = SimpleButtonTrafficEnv(domain_randomization=False)
    
    # Scenario: Attack longest queue effectively
    env.queues = np.array([15.0, 3.0, 2.0, 2.0])
    print(f"Queues: {env.queues}")
    print(f"Longest queue: {np.max(env.queues)}")
    
    # Action 0: Clear from N/S (includes longest queue)
    prev_queues = env.queues.copy()
    cars_cleared = 5.0
    reward = env._calculate_balanced_reward(0, prev_queues, cars_cleared)
    
    print(f"\nAction: Clear from N/S (attack longest queue)")
    print(f"Cars cleared: {cars_cleared}")
    print(f"Reward: {reward:.2f}")
    print(f"Expected: +10 to +15 (POSITIVE)")
    print(f" PASS" if reward > 8 else "✗ FAIL")
    
    # Test 2: Verify negative rewards for bad performance
    print("\n Test 2: Bad Performance Should Give NEGATIVE Rewards")
    
    env.queues = np.array([20.0, 15.0, 18.0, 16.0])
    print(f"Queues (high congestion): {env.queues}")
    
    # Bad action: Clear nothing
    prev_queues = env.queues.copy()
    cars_cleared = 0.0
    reward = env._calculate_balanced_reward(2, prev_queues, cars_cleared)
    
    print(f"\nAction: All red (clear nothing)")
    print(f"Cars cleared: {cars_cleared}")
    print(f"Reward: {reward:.2f}")
    print(f"Expected: -10 to -15 (NEGATIVE)")
    print(f" PASS" if reward < -8 else "✗ FAIL")
    
    # Test 3: Full episode test
    print("\n Test 3: Full Episode with Longest Queue Strategy")
    
    env = SimpleButtonTrafficEnv(domain_randomization=False)
    obs, _ = env.reset()
    
    total_reward = 0
    total_cleared = 0
    positive_steps = 0
    negative_steps = 0
    
    print("\nRunning 20 steps with longest-queue-first policy...")
    
    for step in range(20):
        # Simple policy: always serve longest queue
        # Map to action: if N or S is longest, action 0. If E or W is longest, action 1.
        if obs[0] > obs[2] or obs[1] > obs[3]:
            action = 0  # N/S
        else:
            action = 1  # E/W
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cleared += info['cars_cleared']
        
        if reward > 0:
            positive_steps += 1
        else:
            negative_steps += 1
        
        if step < 5:  # Show first 5 steps
            print(f"Step {step+1:2d}: Queues={obs}, Reward={reward:6.2f}, Cleared={info['cars_cleared']:.0f}")
        
        if terminated or truncated:
            break
    
    print(f"\nResults over 20 steps:")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Average reward: {total_reward/20:.1f} per step")
    print(f"  Total cleared: {total_cleared:.0f} cars")
    print(f"  Positive rewards: {positive_steps}/20 steps")
    print(f"  Negative rewards: {negative_steps}/20 steps")
    print(f"\n PASS" if total_reward > 50 else "✗ FAIL (should be >50 for good policy)")
    
    # Test 4: Compare Run 5 vs Run 6 reward scales
    print("\n Test 4: Run 5 vs Run 6 Reward Comparison")
    
    # Same scenario for both
    test_queues = np.array([12.0, 5.0, 4.0, 6.0])
    cars_cleared = 5.0
    
    # Run 5 reward (old)
    run5_reward = (-2.0 * np.max(test_queues) + 
                   0.5 * cars_cleared + 
                   -0.1 * np.sum(test_queues))
    
    # Run 6 reward (new)
    env.queues = test_queues.copy()
    prev_queues = test_queues.copy()
    run6_reward = env._calculate_balanced_reward(0, prev_queues, cars_cleared)
    
    print(f"Same scenario: Queues={test_queues}, Cleared={cars_cleared}")
    print(f"\nRun 5 reward: {run5_reward:.2f} (NEGATIVE scale)")
    print(f"Run 6 reward: {run6_reward:.2f} (POSITIVE scale)")
    print(f"\nInterpretation:")
    print(f"  Run 5: Less negative = better (confusing)")
    print(f"  Run 6: More positive = better (clear!)")
    
    # Test 5: Hardware-aware DR
    print("\n Test 5: Hardware-Aware Domain Randomization")
    
    env_dr = SimpleButtonTrafficEnv(domain_randomization=True)
    
    print("Running 3 episodes with DR enabled...")
    for episode in range(3):
        obs, _ = env_dr.reset()
        print(f"\nEpisode {episode + 1}:")
        print(f"  Arrival rate: {env_dr.arrival_rate:.3f}")
        print(f"  GPIO latency: {env_dr.gpio_latency:.1f}ms")
        print(f"  Button debounce: {env_dr.button_debounce:.0f}ms")
    
    print("\n DR parameters vary each episode (hardware robustness)")
    
    print("ALL TESTS COMPLETE")