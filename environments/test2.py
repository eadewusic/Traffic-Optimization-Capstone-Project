#!/usr/bin/env python3
"""
Quick sanity check - does final model make sensible decisions?
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np

# 1. IMPORT YOUR ENVIRONMENT
# You MUST replace 'your_env_module' with the actual file/module where SimpleButtonTrafficEnv is defined.
# If it's in a file named 'traffic_env.py' in the parent directory, it might look like:
from simple_button_env import SimpleButtonTrafficEnv # <-- Correct this import path/name

# 2. DEFINE THE ENVIRONMENT CREATOR FUNCTION
def make_env():
    # Instantiate the environment exactly as it was configured during training.
    return SimpleButtonTrafficEnv(domain_randomization=False)

# Create the vector environment required by VecNormalize.load
env = DummyVecEnv([make_env]) 

# Load final model
model = PPO.load("../models/hardware_ppo/run_6/final_model")
# 3. PASS THE CORRECT VECTOR ENVIRONMENT TO VECNORMALIZE.LOAD
vec_norm = VecNormalize.load("../models/hardware_ppo/run_6/vecnormalize.pkl", venv=env)

# Test scenarios
test_cases = [
    {
        'name': 'Heavy North traffic',
        'obs': np.array([[18.0, 2.0, 3.0, 2.0]]),
        'expected': 0,  # Should choose N/S green
        'reason': 'North has 18 cars, should prioritize N/S'
    },
    {
        'name': 'Heavy East traffic', 
        'obs': np.array([[2.0, 3.0, 16.0, 2.0]]),
        'expected': 1,  # Should choose E/W green
        'reason': 'East has 16 cars, should prioritize E/W'
    },
    {
        'name': 'Balanced traffic',
        'obs': np.array([[8.0, 7.0, 8.0, 7.0]]),
        'expected': None,  # Could be either 0 or 1
        'reason': 'Equal traffic, either action is reasonable'
    }
]

print("SANITY CHECK: Final Model Decision-Making\n")

for i, test in enumerate(test_cases, 1):
    # Normalize observation
    obs_norm = vec_norm.normalize_obs(test['obs'])
    
    # Get action (action is a NumPy array here)
    action_array, _ = model.predict(obs_norm, deterministic=True)
    
    # Extract the scalar integer value from the array
    action = action_array.item() # Use .item() to safely convert a single-element array to an int
    
    # Check result
    if test['expected'] is None:
        status = "✓" if action in [0, 1] else "✗"
    else:
        status = "✓" if action == test['expected'] else "⚠"
    
    # This line now works because 'action' is an integer
    action_name = ["N/S Green", "E/W Green", "All Red", "Emergency"][action] 
    
    print(f"{status} Test {i}: {test['name']}")
    print(f"   Queues: {test['obs'][0]}")
    print(f"   Action: {action} ({action_name})")
    print(f"   Reason: {test['reason']}\n")

print("If all tests pass (✓), your final model is working correctly!")

#**Expected output:**
#✓ Test 1: Heavy North traffic
#✓ Test 2: Heavy East traffic  
#✓ Test 3: Balanced traffic