from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import gym 
import os
import torch
import numpy as np
import argparse


class CheckpointTrajCallback(BaseCallback):
    """
    Callback for saving a Trajectory every ``save_freq`` calls
    to ``env.step()``.

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, num_eps: int = 12, max_ep_length: int = 5000, verbose: int = 0):
        super(CheckpointTrajCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.num_eps = num_eps
        # self.env = gym.make('Pendulum-v0')

    def _init_callback(self) -> None:
        self.save_path = os.path.join(self.save_path,self.model.env.envs[0].spec.id)
        # Create folder if needed
        os.makedirs(self.save_path, exist_ok=True)
            

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            for ep in range(self.num_eps):
                states = []
                ep_reward = 0
                state = self.model.env.reset()
                for i in range(100):
                    action,_ = self.model.predict(state)
                    next_state, reward, done, info = self.model.env.step(action)
                    ep_reward += reward[0]
                    states.append(next_state[0])
                    state = next_state

                    if done[0]:
                        break
                states = np.array(states)
                path = os.path.join(self.save_path, '%s_Step_%s_Ep_%02d_Reward_%.3f' % (self.model.env.envs[0].spec.id,self.num_timesteps, ep, ep_reward))
                if self.verbose > 1:
                    print(f"Saving trajs to {path}")
                np.savez(path, states=states)
            

            
        return True

def get_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--env", type=str, help="Environment to use", default="LunarLander-v2")
    args.add_argument("--num_eps", type=int, default=20, help="Number of episodes to run each checkpointed model for")
    args.add_argument("--max_ep_length", type=int, default=500, help="Terminate episode after this many steps")
    args.add_argument("--save_freq", type=int, default=1000, help="Save trajectories after this many steps")
    args.add_argument("--total_timesteps", type=int, default=2000, help="Terminate training after this many steps")
    
    return args.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Save a trajectory every 1000 steps
    checkpoint_callback = CheckpointTrajCallback(
        save_freq=args.save_freq, 
        save_path='./trajs/', 
        num_eps=args.num_eps
        )


    model = PPO('MlpPolicy', args.env, verbose=1, tensorboard_log="./trajs/"+args.env+"/tensorboard_logs/")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

# python -m stablebaselines.get_data --env Pendulum-v0 --num_eps 20 --max_ep_length 500 --save_freq 2000 --total_timesteps 20000
# python -m stablebaselines.get_data --env CartPole-v0 --num_eps 20 --max_ep_length 500 --save_freq 2000 --total_timesteps 20000
# python -m stablebaselines.get_data --env LunarLander-v2 --num_eps 20 --max_ep_length 500 --save_freq 20000 --total_timesteps 200000
# python -m stablebaselines.get_data --env CartPole-v0 --num_eps 20 --max_ep_length 500 --save_freq 1000 --total_timesteps 10000