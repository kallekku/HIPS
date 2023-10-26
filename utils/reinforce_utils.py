"""
Utility functions for reinforce
"""

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_acts(policy, encoded, obs):
    """
    # Given a policy, the one-hot encoded subgoals and observations, predict the actions
    """
    obs_reshape = encoded.view(-1, *np.ones_like(obs.shape[1:])) * obs
    idxs = torch.where(encoded > 1 - 1e-6)[0]

    # Get the index of a subgoal for each state
    goal_idxs = []
    j = 0
    if idxs.shape[0] > 0:
        obs_ptr = idxs[0].item()
    else:
        obs_ptr = obs.shape[0] - 1
    for i in range(obs.shape[0] - 1):
        if i == obs_ptr and j < idxs.shape[0] - 1:
            j += 1
            obs_ptr = idxs[j].item()
        elif i == obs_ptr:
            obs_ptr = obs.shape[0] - 1
        goal_idxs.append(obs_ptr)

    # Given the subgoals, create the input tensor to the policy network
    goals = obs_reshape[goal_idxs]
    inp = torch.cat((obs[:-1], goals), dim=1)

    # Predict actions and return
    acts = policy(inp)
    return acts


def get_tensor(arr, longt=False):
    """
    Convert numpy array to a tensor and copies it on device
    """
    if longt:
        return torch.LongTensor(arr).to(device)
    return torch.FloatTensor(arr).to(device)
