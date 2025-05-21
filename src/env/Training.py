import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from env.MixingMachineEnv import InterfaceEnv


from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardRewardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        # Actual reward
        self.current_rewards += self.locals["rewards"][0]

        # Save reward if episode finished
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            
            # Log into Tensorboard
            mean_reward = np.mean(self.episode_rewards[-10:])  # 10 episode smoothing
            
            self.logger.record("rollout/average_reward_per_episode", mean_reward)

            # Reset for next episode
            self.current_rewards = 0
            self.current_length = 0

        return True






df = pd.read_csv('df_train.csv')

env = InterfaceEnv(df)
# env = gym.make("InterfaceEnv-v0", df=df)

# Check environment
check_env(env, warn=True)

# Vectorize the environment
env = DummyVecEnv([lambda: env])

name = 'PPO-1.zip'
reward_callback = TensorboardRewardCallback()

model = PPO("MultiInputPolicy", env, n_steps=1024, verbose=1,tensorboard_log="./offline_ppo_logs/", )
model.learn(total_timesteps=200000, callback=reward_callback, reset_num_timesteps=True)

# model = PPO.load(name, env=env)
# model.learn(total_timesteps=100000, callback=reward_callback, reset_num_timesteps=False)

# --------------------
# Guardar el modelo
# --------------------
model.save(name)
