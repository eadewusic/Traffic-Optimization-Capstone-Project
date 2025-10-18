from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Load both models
final_model = PPO.load("models/hardware_ppo/run_6/final_model.zip")
best_model = PPO.load("models/hardware_ppo/run_6/best_model.zip")

# Check their training steps
print(f"Final model trained: {final_model.num_timesteps} steps")
print(f"Best model trained: {best_model.num_timesteps} steps")

# If they're close (within 10%), use final model
# If best_model is much earlier, it might underperform