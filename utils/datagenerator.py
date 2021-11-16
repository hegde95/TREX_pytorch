import numpy as np
import os
import random
import torch

# read all trajectories from a directory
def get_trajectories(env_name, data_dir):
    path = os.path.join(data_dir, env_name)

    # get all the npz files in the directory
    files = [f for f in os.listdir(path) if f.endswith('.npz')]

    traj_dict = []
    min_traj_len = 999999999
    for f in files:
        reward = f.split('_')[-1][:-4]
        traj = np.load(os.path.join(path, f))['states']
        traj_dict.append((traj, reward))
        min_traj_len = min(min_traj_len, traj.shape[0])
    return traj_dict, min_traj_len


class DataGenerator(object):
    def __init__(self, env_name, device, dir_name = 'trajs', traj_len = 50):
        self.env_name = env_name
        self.dir_name = dir_name
        self.trajectories, min_traj_len = get_trajectories(env_name, dir_name)
        self.traj_len = min(traj_len, min_traj_len)
        self.device = device

    def get_batch(self, batch_size):
        X = []
        Y = []
        for i in range(batch_size):
            # randomly choose a trajectory
            traj1, reward1 = random.choice(self.trajectories)
            traj2, reward2 = random.choice(self.trajectories)

            if len(traj1) < self.traj_len or len(traj2) < self.traj_len:
                min_len = min(len(traj1), len(traj2))
                traj1 = traj1[:min_len]
                traj2 = traj2[:min_len]

            else:
                # randomly choose a starting point
                start1 = random.randint(0, len(traj1) - self.traj_len)
                start2 = random.randint(0, len(traj2) - self.traj_len)

                # select a random part of the trajectory
                traj1 = traj1[start1:start1+self.traj_len]
                traj2 = traj2[start2:start2+self.traj_len]

            # append to the batch
            X.append((traj1, traj2))

            # set y to be 1 if reward1 > reward2, else 0
            if reward1 > reward2:
                Y.append(1)
            else:
                Y.append(0)
        
        # convert to numpy arrays
        X = np.array(X)
        Y = np.array(Y)

        # convert the array to tensors
        X = torch.from_numpy(X).float().to(self.device)
        Y = torch.from_numpy(Y).long().to(self.device)

        # return the batch
        return X, Y

    # def __len__(self):
    #     return len(self.pairs)

    # def __getitem__(self, idx):
    #     return self.pairs[idx]




# main method to test
if __name__ == '__main__':
    dg = DataGenerator('LunarLander-v2')
    print(dg.get_batch(10))