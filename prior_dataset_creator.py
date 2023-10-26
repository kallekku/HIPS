"""
Create a dataset for training the prior
"""

import argparse
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from datasets import SokobanTrajectoryDataset, TrajectoryDataset, TSPTrajectoryDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def create_dataset(env, detector_path, vqvae_path, ds_name, validation, H, fixed_len):
    """
    Implements the dataset creation
    """
    if not fixed_len:
        detector = torch.load(detector_path)
    else:
        if env.lower() == 'sokoban':
            H = 5
        elif env.lower() == 'stp':
            H = 7
        elif env.lower() == 'bw':
            H = 5
        elif env.lower() == 'tsp':
            H = 13

    vqvae = torch.load(vqvae_path)
    if validation and env.lower() == 'sokoban':
        dataset = SokobanTrajectoryDataset('./datasets/sokoban_obs_val.pkl',
                                           './datasets/sokoban_acts_val.pkl')
    elif not validation and env.lower() == 'sokoban':
        dataset = SokobanTrajectoryDataset('./datasets/sokoban_obs.pkl',
                                           './datasets/sokoban_acts.pkl')
    elif validation and env.lower() == 'stp':
        dataset = TrajectoryDataset('./datasets/stp_obs_val.pkl',
                                    './datasets/stp_acts_val.pkl')
    elif not validation and env.lower() == 'stp':
        dataset = TrajectoryDataset('./datasets/stp_obs.pkl', './datasets/stp_acts.pkl')
    elif validation and env.lower() == 'bw':
        dataset = TrajectoryDataset('./datasets/bw_obs_val.pkl', './datasets/bw_acts_val.pkl')
    elif not validation and env.lower() == 'bw':
        dataset = TrajectoryDataset('./datasets/bw_obs.pkl', './datasets/bw_acts.pkl')
    elif validation and env.lower() == 'tsp':
        max_len = 450
        dataset = TSPTrajectoryDataset(dataset_len=100, max_len=max_len, prior_dataset=True)
    elif not validation and env.lower() == 'tsp':
        max_len = 450
        dataset = TSPTrajectoryDataset(dataset_len=2500, max_len=max_len, prior_dataset=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    inp_tot = []
    tgt_tot = []

    for it, (obs, acts) in enumerate(dataloader):
        if not (it + 1) % 100:
            print(it + 1)
        obs = obs.squeeze(dim=0).to(device)
        acts = acts.squeeze(dim=0).long().to(device)

        if not fixed_len:
            # Segment trajectory with detector
            i = 0
            kfs = [0]
            while i < obs.shape[0]-1:
                inp = obs[i+1:i+H+1]
                # i is the current state, inp contains the possible subgoals
                inp = torch.cat((obs[i].repeat(inp.shape[0], 1, 1, 1), inp), dim=1)
                # Compute probabilities with q, sample the next subgoal
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

        if env.lower() == 'tsp':
            inputs = obs[kfs[1:]]
            conds = obs[kfs[:-1]]
            acts = acts[kfs[:-1]]
        else:
            input_idxs = []
            for i in range(1, len(kfs)):
                input_idxs += (kfs[i] - kfs[i-1]) * [kfs[i]]
            inputs = obs[np.array(input_idxs)]
            conds = obs[:obs.shape[0]-1]

        encoder_input = torch.cat((conds, inputs), dim=1)

        with torch.no_grad():
            code_idxs = vqvae.encode(encoder_input).squeeze(dim=2).squeeze(dim=1).long()

        if env.lower() == 'tsp':
            inp = obs[kfs[:-1]].cpu()
        else:
            inp = obs[:obs.shape[0]-1].cpu()
        tgt = torch.cat((code_idxs[:, None], acts[:, None]), dim=1).cpu()

        inp_tot.append(inp)
        tgt_tot.append(tgt)

    inp_tot = torch.cat(inp_tot, dim=0)
    tgt_tot = torch.cat(tgt_tot, dim=0)

    print("Saving")
    if validation:
        pickle.dump(inp_tot, open(f"./datasets/vqvae_prior_{env}_{ds_name}_inputs_val.pkl", "ab"))
        pickle.dump(tgt_tot, open(f"./datasets/vqvae_prior_{env}_{ds_name}_targets_val.pkl", "ab"))
    else:
        pickle.dump(inp_tot, open(f"./datasets/vqvae_prior_{env}_{ds_name}_inputs.pkl", "ab"))
        pickle.dump(tgt_tot, open(f"./datasets/vqvae_prior_{env}_{ds_name}_targets.pkl", "ab"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSP Prior Dataset Creator')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--detector', type=str)
    parser.add_argument('--vqvae', type=str, required=True)
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--validation', default=False, action='store_true')
    parser.add_argument('--fixed_len', default=False, action='store_true')
    parser.add_argument('--env', type=str, required=True)
    args = parser.parse_args()
    print(args)
    ds_name = args.name
    detector_path = args.detector
    vqvae_path = args.vqvae
    create_dataset(args.env, args.detector, args.vqvae, args.name,
                   args.validation, args.H, args.fixed_len)
