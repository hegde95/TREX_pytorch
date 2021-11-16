import os
import sys
import numpy as np
import argparse
import gym

from utils.models import RewardModel
from utils.datagenerator import DataGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorboardX

def get_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--env", type=str, default='CartPole-v0')
    args.add_argument("--ckpt_dir", type=str, help="Location to save T-REX checkpoints", default="reward_ckpts")
    args.add_argument("--restore_from_step", type=int, default=None, help="Checkpointed step to load and resume training from (if None, train from scratch)")
    args.add_argument("--n_train_steps", type=int, default=300000)
    args.add_argument("--learn_rate", type=float, default=5e-5)
    args.add_argument("--n_workers", type=int, default=8, help="For data loading and preprocessing")
    args.add_argument("--traj_length", type=int, default=50, help="We sample a random snippet of length traj_length from each demonstration sample to train on")
    return args.parse_args()

def train(args):
    # Set up environment
    env = gym.make(args.env)
    # get the state
    state_dim = env.observation_space.shape[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Set up model
    model = RewardModel(state_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    # tensorboard
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(os.path.join(args.ckpt_dir,args.env), "trex_tensorboard"))

    # Set up data generator
    data_generator = DataGenerator(args.env, device = device)

    # Set up checkpointing
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(os.path.join(args.ckpt_dir,args.env))
    ckpt_path = os.path.join(os.path.join(args.ckpt_dir,args.env), "model.ckpt")

    # # Load checkpoint if specified
    # if args.restore_from_step is not None:
    #     print("Loading checkpoint from step {}".format(args.restore_from_step))
    #     model.load_state_dict(torch.load(ckpt_path + ".{}".format(args.restore_from_step)))
    #     optimizer.load_state_dict(torch.load(ckpt_path + ".opt.{}".format(args.restore_from_step)))

    # Train
    for step in range(args.n_train_steps):
        # Sample data
        traj_pairs, preference = data_generator.get_batch(32)

        # Train on data
        model.train()
        optimizer.zero_grad()
        # loss = model(states, actions, rewards, next_states)
        # loss.backward()

        # get sum of predicted rewards
        pred_reward1 = torch.sum(model(traj_pairs[:, 0]), dim=1)
        pred_reward2 = torch.sum(model(traj_pairs[:, 1]), dim=1)

        reward_sum = torch.cat([pred_reward1, pred_reward2], dim=1)
        loss = nn.CrossEntropyLoss()(reward_sum, preference)
        loss.backward()


        optimizer.step()

        # Save checkpoint
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            print("Saving checkpoint to {}".format(ckpt_path))
            torch.save(model.state_dict(), ckpt_path)
            torch.save(optimizer.state_dict(), ckpt_path + ".opt")

            # tensorboard
            writer.add_scalar("loss", loss.item(), step)

    return    

if __name__ == '__main__':
    args = get_args()
    
    # Train
    train(args)
    
    
    
    