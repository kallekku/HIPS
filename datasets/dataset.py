"""
Implements custom datasets from training the neural networks
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from envs import get_traj


class SokobanTrajectoryDataset(Dataset):
    """
    Consists of Sokoban Trajectories
    """
    def __init__(self, obs_filepath, act_filepath):
        obs_file = open(obs_filepath, 'rb')
        act_file = open(act_filepath, 'rb')
        self.obs = pickle.load(obs_file)
        self.acts = pickle.load(act_file)

    def __len__(self):
        if isinstance(self.obs, list):
            return len(self.obs)
        else:
            return self.obs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        obs = self.obs[idx]
        acts = self.acts[idx]
        obs = torch.Tensor(obs).float()
        acts = torch.Tensor(acts).long().squeeze()
        acts = (acts - 1) % 4
        return obs, acts


class TrajectoryDataset(Dataset):
    """
    Consists of General Trajectories
    """
    def __init__(self, obs_filepath, act_filepath):
        obs_file = open(obs_filepath, 'rb')
        act_file = open(act_filepath, 'rb')
        self.obs = pickle.load(obs_file)
        self.acts = pickle.load(act_file)

    def __len__(self):
        if isinstance(self.obs, list):
            return len(self.obs)
        else:
            return self.obs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        obs = self.obs[idx]
        acts = self.acts[idx]
        obs = torch.Tensor(obs).float()
        acts = torch.Tensor(acts).long().squeeze()
        return obs, acts


class TSPTrajectoryDataset(Dataset):
    """
    Consists of TSP Trajectories
    """
    def __init__(self, dataset_len, max_len, standard_dataset=False, dist_dataset=False,
                 prior_dataset=False, complete_dataset=False, abl_vqvae_dataset=False):
        # As we're generating trajectories on the fly, the dataset_len only matters
        # for the purposes of tracking epochs in dataloaders
        self.dataset_len = dataset_len
        self.max_len = max_len
        self.dist_dataset = dist_dataset
        self.prior_dataset = prior_dataset
        self.complete_dataset = complete_dataset
        self.standard_dataset = standard_dataset
        self.abl_vqvae_dataset = abl_vqvae_dataset

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        obs, acts = get_traj(size=25, targets=25, actions=True)
        while obs.shape[0] > self.max_len:
            obs, acts = get_traj(size=25, targets=25, actions=True)
        obs = torch.FloatTensor(obs).permute(0, 3, 1, 2)
        if self.prior_dataset:
            acts = torch.LongTensor(acts)
            return obs, acts
        elif self.dist_dataset:
            done = torch.LongTensor(np.arange(obs.shape[0]-1, -1, -1))
            return obs, done
        elif self.complete_dataset:
            obs = obs[:-1]
            acts = torch.LongTensor(acts)
            done = torch.LongTensor(np.arange(obs.shape[0], 0, -1))
            return obs, acts, done
        elif self.standard_dataset:
            acts = torch.LongTensor(acts)
            return obs, acts
        elif self.abl_vqvae_dataset:
            inp_indices = np.arange(4, obs.shape[0]+4, 1)
            inp_indices = np.minimum(inp_indices, obs.shape[0]-1)
            cond_indices = np.arange(obs.shape[0])
            return obs[inp_indices], obs[cond_indices]
        else:
            raise NotImplementedError


def tsp_vqvae_collate(batch):
    """
    Collate-function for TSP VQVAE training
    """
    inps, conds = zip(*batch)
    inps = torch.cat(inps, dim=0)
    conds = torch.cat(conds, dim=0)
    inds = np.random.choice(inps.shape[0], size=128, replace=False)
    return inps[inds], conds[inds]


def tsp_dist_collate(batch):
    """
    Collate-function for TSP heuristic training
    """
    obs, dists = zip(*batch)
    obs = torch.cat(obs, dim=0)
    dists = torch.cat(dists, dim=0)
    inds = np.random.choice(obs.shape[0], size=128, replace=False)
    return obs[inds], dists[inds]
