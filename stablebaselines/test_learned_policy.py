import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.models import RewardModel
from gym.envs.registration import register


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

model = PPO.load("trex_policy_ckpts/rl_model_200000_steps", env=env)

for j in range(100):
    obs = env.reset()
    reward = 0
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        reward += rewards
        env.render()
        if dones:
            print(f"Episode {j} finished after {i} timesteps with reward {reward}")
            break