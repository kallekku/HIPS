"""
Ute the trained detector to create a dataset of subgoal pairs for training the discrete VQVAE
"""

import argparse
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from datasets import SokobanTrajectoryDataset, TrajectoryDataset, TSPTrajectoryDataset
from envs import revert_obs
from utils import tiny_to_full

device = "cuda" if torch.cuda.is_available() else "cpu"


def create_dataset(env, detector_path, ds_name, visualize, validation, H, fixed_len,
                   histogram, visualize_subpart):
    """
    Creates a dataset of subgoals pairs
    """
    if not fixed_len:
        detector = torch.load(detector_path)

    if validation and env.lower() == 'sokoban':
        ds = SokobanTrajectoryDataset('./datasets/sokoban_obs_val.pkl',
                                      './datasets/sokoban_acts_val.pkl')
    elif not validation and env.lower() == 'sokoban':
        ds = SokobanTrajectoryDataset('./datasets/sokoban_obs.pkl',
                                      './datasets/sokoban_acts.pkl')
    elif validation and env.lower() == 'stp':
        ds = TrajectoryDataset('./datasets/stp_obs_val.pkl', './datasets/stp_acts_val.pkl')
    elif not validation and env.lower() == 'stp':
        ds = TrajectoryDataset('./datasets/stp_obs.pkl', './datasets/stp_acts.pkl')
    elif validation and env.lower() == 'bw':
        ds = TrajectoryDataset('./datasets/bw_obs_val.pkl', './datasets/bw_acts_val.pkl')
    elif not validation and env.lower() == 'bw':
        ds = TrajectoryDataset('./datasets/bw_obs.pkl', './datasets/bw_acts.pkl')
    elif validation and env.lower() == 'tsp':
        max_len = 450
        ds = TSPTrajectoryDataset(dataset_len=500, max_len=max_len, standard_dataset=True)
    elif not validation and env.lower() == 'tsp':
        max_len = 450
        ds = TSPTrajectoryDataset(dataset_len=10000, max_len=max_len, standard_dataset=True)

    dl = DataLoader(ds, batch_size=1, shuffle=True)

    X, y = None, None
    lens = []
    print("Dataset loaded")
    for it, (obs, _) in enumerate(dl):
        obs = obs.squeeze(dim=0).to(device)

        if not fixed_len:
            # Segment trajectory with detector
            i = 0
            kfs = [0]
            while i < obs.shape[0]-1:
                inp = obs[i+1:i+H+1]
                # i is the current state, inp contains the possible subgoals
                inp = torch.cat((obs[i].repeat(inp.shape[0], 1, 1, 1), inp), dim=1)
                # Compute probabilities with q, sapmle the next subgoal
                with torch.no_grad():
                    logits = detector(inp, compute_val=False)
                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    sample = dist.sample()
                kfs.append(i + sample.item() + 1)
                i += sample.item() + 1
        else:
            kfs = list(range(0, obs.shape[0]-1, H))
            kfs.append(obs.shape[0]-1)

        if histogram:
            distr = np.array(kfs[1:]) - np.array(kfs[:-1])
            lens += list(distr)
            if len(lens) > 5000:
                print("Saving histogram")
                plt.hist(lens, bins=H, density=True)
                plt.savefig(f'./visualizations/detector_{env}_histogram')
                sys.exit(0)

        if visualize_subpart:
            images = []
            for i, o in enumerate(obs):
                if env.lower() == 'sokoban':
                    img = tiny_to_full(revert_obs(o.long().cpu().permute(1, 2, 0).numpy(),
                                                  dim=4), show=False)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.savefig(f'./visualizations/{env.lower()}_frame_{i}',
                                bbox_inches='tight', pad_inches=0)
            sys.exit(0)

        if visualize:
            images = []
            for k in kfs:
                if env.lower() == 'sokoban':
                    images.append(tiny_to_full(obs[k].long().cpu().permute(1, 2, 0).numpy(),
                                  show=False))
                elif env.lower() == 'bw':
                    images.append(obs[k].long().cpu().permute(1, 2, 0).numpy())
            rows = (len(kfs) - 1) // 4 + 1

            fig, axis = plt.subplots(rows, 4)
            for i in range(rows):
                for j in range(4):
                    if rows > 1:
                        ax = axis[i, j]
                    else:
                        ax = axis[j]
                    if i*4 + j < len(images):
                        ax.imshow(images[i*4 + j])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
            fig.tight_layout()
            plt.savefig(f'./visualizations/keyframes_{it+1}')
            if it >= 10:
                sys.exit(0)
            continue

        # Save subgoal pairs of the trajectories
        cond_idxs = kfs[:-1]
        to_reconstruct_idxs = kfs[1:]

        inp = obs[to_reconstruct_idxs]
        cond = obs[cond_idxs]

        if X is None:
            X = inp
            y = cond
        else:
            X = torch.cat((X, inp), dim=0)
            y = torch.cat((y, cond))

    if visualize or histogram:
        return

    print("Saving")
    if validation:
        pickle.dump(X, open(f"./datasets/vqvae_{env}_{ds_name}_inputs_val.pkl", "wb"))
        pickle.dump(y, open(f"./datasets/vqvae_{env}_{ds_name}_conds_val.pkl", "wb"))
    else:
        pickle.dump(X, open(f"./datasets/vqvae_{env}_{ds_name}_inputs.pkl", "wb"))
        pickle.dump(y, open(f"./datasets/vqvae_{env}_{ds_name}_conds.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSP Dataset Creator')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--H', type=int, required=True)
    parser.add_argument('--detector', type=str)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--visualize_subpart', default=False, action='store_true')
    parser.add_argument('--validation', default=False, action='store_true')
    parser.add_argument('--fixed_len', default=False, action='store_true')
    parser.add_argument('--histogram', default=False, action='store_true')
    parser.add_argument('--env', type=str, required=True)
    args = parser.parse_args()
    print(args)
    ds_name = args.name
    detector_path = args.detector
    create_dataset(args.env, args.detector, args.name, args.visualize, args.validation, args.H,
                   args.fixed_len, args.histogram, args.visualize_subpart)
