"""
Trains a distance (value/heuristic) function for the environments
"""

import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import STPDistNetwork, BWDistNetwork, SokobanDistNetwork, TSPDistNetwork
from datasets import TSPTrajectoryDataset, tsp_dist_collate

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(env, dim, run_id, dropout):
    """
    Implements distance function training
    """
    epochs = 100000
    instance_norm = True
    if env == 'stp':
        net = STPDistNetwork(dim=dim, max_len=300,
                             instance_norm=instance_norm, dropout=dropout).to(device)
    elif env == 'sokoban':
        net = SokobanDistNetwork(dim=dim, max_len=220,
                                 instance_norm=instance_norm, dropout=dropout).to(device)
    elif env == 'bw':
        net = BWDistNetwork(dim=dim, max_len=75,
                            instance_norm=instance_norm, dropout=dropout).to(device)
    elif env == 'tsp':
        max_len = 450
        net = TSPDistNetwork(dim=dim, max_len=max_len,
                             instance_norm=instance_norm, dropout=dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    print_freq = 1000

    if env == 'stp':
        with open('./datasets/stp_obs_dists.pkl', 'rb') as handle:
            x_train = pickle.load(handle)
        with open('./datasets/stp_dists.pkl', 'rb') as handle:
            y_train = pickle.load(handle)
        with open('./datasets/stp_obs_dists_val.pkl', 'rb') as handle:
            x_val = pickle.load(handle)
        with open('./datasets/stp_dists_val.pkl', 'rb') as handle:
            y_val = pickle.load(handle)
    elif env == 'sokoban':
        with open('./datasets/sokoban_obs_dists.pkl', 'rb') as handle:
            x_train = pickle.load(handle)
        with open('./datasets/sokoban_dists.pkl', 'rb') as handle:
            y_train = pickle.load(handle)
        with open('./datasets/sokoban_obs_dists_val.pkl', 'rb') as handle:
            x_val = pickle.load(handle)
        with open('./datasets/sokoban_dists_val.pkl', 'rb') as handle:
            y_val = pickle.load(handle)
    elif env == 'bw':
        with open('./datasets/bw_obs_dists.pkl', 'rb') as handle:
            x_train = pickle.load(handle)
        with open('./datasets/bw_dists.pkl', 'rb') as handle:
            y_train = pickle.load(handle)
        with open('./datasets/bw_obs_dists_val.pkl', 'rb') as handle:
            x_val = pickle.load(handle)
        with open('./datasets/bw_dists_val.pkl', 'rb') as handle:
            y_val = pickle.load(handle)
    elif env == 'tsp':
        dataset = TSPTrajectoryDataset(dataset_len=10000, max_len=max_len, dist_dataset=True)
        dataloader = DataLoader(dataset, batch_size=4, num_workers=1, collate_fn=tsp_dist_collate)
        val_dataset = TSPTrajectoryDataset(dataset_len=500, max_len=max_len, dist_dataset=True)
        x_train, y_train, x_val, y_val = None, None, None, None
        val_dataloader = DataLoader(val_dataset, batch_size=4,
                                    num_workers=4, collate_fn=tsp_dist_collate)

    if x_train is not None:
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    if x_val is not None:
        val_dataset = TensorDataset(x_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    best_loss = np.inf

    for epoch in range(1, epochs+1):
        train_losses = []
        train_tot, train_corr = 0, 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            otp = net(x)
            loss = criterion(otp, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())
            train_tot += x.shape[0]
            train_corr += (otp.argmax(dim=-1) == y).sum()

            if not (i + 1) % print_freq:
                print("Update {}, train_loss: {:.2f}, train_corr: {:.2f} %"
                      .format(i+1, np.mean(train_losses), train_corr/train_tot*100),
                      flush=True)
        val_losses = []
        val_tot, val_corr = 0, 0
        net.eval()
        for (x, y) in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                otp = net(x)
                loss = criterion(otp, y)

            val_losses.append(loss.item())
            val_tot += x.shape[0]
            val_corr += (otp.argmax(dim=-1) == y).sum()
        net.train()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print("Epoch {}, Train loss: {:.2f}, train corr: {:.2f} %, "
              .format(epoch, train_loss, train_corr/train_tot*100) +
              "val loss: {:.2f}, val corr: {:.2f} %".format(val_loss, val_corr/val_tot*100),
              flush=True)
        torch.save(net, f"./out/dist_net_{env}_{run_id}.pth")
        if val_loss < best_loss:
            torch.save(net, f"./out/dist_net_{env}_{run_id}_best.pth")
            best_loss = val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dist function')
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--dropout', action='store_true')
    args = parser.parse_args()
    print(args)
    train(args.env, args.dim, args.run_id, args.dropout)
