import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse

from utils.models import RewardModel
from gym.envs.registration import register


def get_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--env", type=str, default='CartPole-v0')
    return args.parse_args()

if __name__ == '__main__':
    args = get_args()

    env = gym.make(args.env)

    model = PPO.load("trex_policy_ckpts/"+ args.env + "/rl_model_200000_steps", env=env)

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