import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.models import RewardModel
from gym.envs.registration import register
import argparse

# class to wrap the gym environment and return a custom reward
class RewardWrapper(gym.Wrapper):
    def __init__(self, env, ckpt_path):
        super(RewardWrapper, self).__init__(env)
        self.reward_model = RewardModel(env.observation_space.shape[0])

        # load the model from the checkpoint
        self.reward_model.load_state_dict(torch.load(ckpt_path))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # convert the observation to a tensor
        obs = torch.from_numpy(obs).float()
        # get the reward
        reward = self.reward_model(obs)
        # convert the reward to a numpy array
        reward = reward.cpu().detach().numpy()[0]

        # # round the reward to 3 decimal places
        # reward = np.round(reward, 3)

        return obs, reward, done, info

def get_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--env", type=str, help="Environment to use", default="LunarLander-v2")
    args.add_argument("--num_eps", type=int, default=20, help="Number of episodes to run each checkpointed model for")
    args.add_argument("--save_freq", type=int, default=1000, help="Save trajectories after this many steps")
    args.add_argument("--total_timesteps", type=int, default=2000, help="Terminate training after this many steps")
    
    return args.parse_args()

if __name__ == '__main__':
    args = get_args()


    env = gym.make(args.env)
    env = RewardWrapper(env, './reward_ckpts/model.ckpt')
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./trex_policy_ckpts/')

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./trex_policy_ckpts/tensorboard_logs/")
    model.learn(total_timesteps=200000, callback=checkpoint_callback)