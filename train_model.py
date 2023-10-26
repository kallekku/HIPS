"""
Trains a dynamics model
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from models import SokobanModel, STPModel, BWModel, TSPModel
from datasets import SokobanTrajectoryDataset, TrajectoryDataset, TSPTrajectoryDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_dynamics(env, run_id):
    """
    Implements the dynamics model training
    """
    dim = 64
    print_freq = 1000
    lr = 2e-4
    epochs = 1000

    if env == 'sokoban':
        net = SokobanModel(dim, 4, 2).to(device)
    elif env == 'stp':
        net = STPModel(dim, 25, 2).to(device)
    elif env == 'bw':
        net = BWModel(dim, 3, 256, expand=True).to(device)
    elif env == 'tsp':
        net = TSPModel(dim).to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    if env == 'sokoban':
        ds = SokobanTrajectoryDataset('./datasets/sokoban_obs.pkl', './datasets/sokoban_acts.pkl')
        ds = ConcatDataset([ds,
                            SokobanTrajectoryDataset('./datasets/sokoban_random_obs.pkl',
                                                         './datasets/sokoban_random_acts.pkl')])
    elif env == 'stp':
        ds = TrajectoryDataset('./datasets/stp_obs.pkl', './datasets/stp_acts.pkl')
        ds = ConcatDataset([ds, TrajectoryDataset('./datasets/stp_random_obs.pkl',
                                                  './datasets/stp_random_acts.pkl')])
    elif env == 'bw':
        ds = TrajectoryDataset('./datasets/bw_obs.pkl', './datasets/bw_acts.pkl')
        ds = ConcatDataset([ds, TrajectoryDataset('./datasets/bw_random_obs.pkl',
                                                  './datasets/bw_random_acts.pkl')])
    elif env == 'tsp':
        max_len = 450
        dataset_len = 1000
        ds = TSPTrajectoryDataset(dataset_len=dataset_len, max_len=max_len, standard_dataset=True)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)

    if env == 'sokoban':
        valid_ds = SokobanTrajectoryDataset('./datasets/sokoban_obs_val.pkl',
                                            './datasets/sokoban_acts_val.pkl')
        valid_ds = ConcatDataset([valid_ds,
                                  SokobanTrajectoryDataset('./datasets/sokoban_random_obs_val.pkl',
                                                           './datasets/sokoban_random_acts_val.pkl')])
    elif env == 'stp':
        valid_ds = TrajectoryDataset('./datasets/stp_obs_val.pkl', './datasets/stp_acts_val.pkl')
        valid_ds = ConcatDataset([valid_ds,                  
                                  TrajectoryDataset('./datasets/stp_random_obs_val.pkl',
                                                    './datasets/stp_random_acts_val.pkl')])
    elif env == 'bw':
        valid_ds = TrajectoryDataset('./datasets/bw_obs_val.pkl', './datasets/bw_acts_val.pkl')
        valid_ds = ConcatDataset([valid_ds,
                                  TrajectoryDataset('./datasets/bw_random_obs_val.pkl',
                                                    './datasets/bw_random_acts_val.pkl')])
    elif env == 'tsp':
        max_len = 450
        dataset_len = 100
        valid_ds = TSPTrajectoryDataset(dataset_len=dataset_len, max_len=max_len,
                                        standard_dataset=True)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=True, num_workers=4)

    losses, valid_losses = [], []
    n_preds, n_corrs, valid_n_preds, valid_n_corrs = 0, 0, 0, 0
    best_loss = np.inf
    for ep in range(1, epochs+1):
        for it, (obs, actions) in enumerate(dl):
            obs = obs.squeeze(dim=0).to(device)
            actions = actions.squeeze(dim=0).to(device).float()
            N = actions.shape[0]
            tgt = obs[1:].long()

            recon = net(obs[:N], actions)

            if env != 'tsp':
                loss = criterion(recon.flatten(0, recon.ndim-2), tgt.flatten())
                n_corrs += (recon.argmax(dim=-1) == tgt).sum()
            else:
                loss = F.binary_cross_entropy(recon, tgt.float())
                n_corrs += ((recon > 0.5) == tgt).sum()
            n_preds += np.prod(list(tgt.shape))

            # Update dynamics parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

            if not (it+1) % print_freq:
                mean_loss = np.mean(losses)
                print("It: {}, losses: {:.4f}, corr: {:.6f}".
                      format(it + 1, mean_loss, n_corrs/n_preds), flush=True)

        for it, (obs, actions) in enumerate(valid_dl):
            obs = obs.squeeze(dim=0).to(device)
            actions = actions.squeeze(dim=0).to(device).float()
            N = actions.shape[0]
            tgt = obs[1:].long()

            with torch.no_grad():
                recon = net(obs[:N], actions)
                if env != 'tsp':
                    loss = criterion(recon.flatten(0, 3), tgt.flatten())
                    valid_n_corrs += (recon.argmax(dim=-1) == tgt).sum()
                else:
                    loss = F.binary_cross_entropy(recon, tgt.float())
                    valid_n_corrs += ((recon > 0.5) == tgt).sum()

            valid_n_preds += np.prod(list(tgt.shape))
            valid_losses.append(loss.item())

        mean_loss = np.mean(losses)
        valid_loss = np.mean(valid_losses)
        print("Epoch: {}, losses: {:.4f}, corr: {:.6f}, valid loss: {:.4f}, valid_corr: {:.6f}".
              format(ep, mean_loss, n_corrs/n_preds, valid_loss, valid_n_corrs/valid_n_preds),
              flush=True)

        losses, valid_losses = [], []
        n_preds, n_corrs, valid_n_preds, valid_n_corrs = 0, 0, 0, 0
        torch.save(net, f'./out/{env}_model_{run_id}.pth')
        if valid_loss < best_loss:
            print("Best loss, saving")
            torch.save(net, f'./out/{env}_model_best_{run_id}.pth')
            best_loss = valid_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Discrete Dynamics Model')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--run_id', type=str, default="0")
    args = parser.parse_args()
    train_dynamics(args.env.lower(), args.run_id)
