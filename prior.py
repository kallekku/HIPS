"""
Trains a prior for both high-level actions (subgoals) and
low-level actions (unconditional BC-policy)
"""

import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import SokobanPrior, STPPrior, BWPrior, TSPPrior

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_prior(env, name, dim, K, save_name=None, ds_name=None, validate=False):
    """
    Train a prior for HIPS
    """
    print_freq = 1000
    lr = 2e-4
    epochs = 200

    if save_name is None:
        save_name = name
    if ds_name is None:
        ds_name = name

    if env.lower() == 'sokoban':
        net = SokobanPrior(dim, K, 4, input_nc=4).to(device)
    elif env.lower() == 'stp':
        net = STPPrior(dim, K, 4, input_nc=25).to(device)
    elif env.lower() == 'bw':
        net = BWPrior(dim, K, 4, input_nc=3).to(device)
    elif env.lower() == 'tsp':
        net = TSPPrior(dim, K, 4, input_nc=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Load the dataset
    with open(f'./datasets/vqvae_prior_{env}_{ds_name}_inputs.pkl', 'rb') as f:
        inps = pickle.load(f)
    with open(f'./datasets/vqvae_prior_{env}_{ds_name}_targets.pkl', 'rb') as f:
        tgts = pickle.load(f)
    dataset = TensorDataset(inps, tgts)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    if validate:
        with open(f'./datasets/vqvae_prior_{env}_{ds_name}_inputs_val.pkl', 'rb') as f:
            inps = pickle.load(f)
        with open(f'./datasets/vqvae_prior_{env}_{ds_name}_targets_val.pkl', 'rb') as f:
            tgts = pickle.load(f)
        dataset = TensorDataset(inps, tgts)
        val_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    losses, corr_acts, corr_codes, tots = [], 0, 0, 0
    best_loss = np.inf
    total_updates = 0

    for epoch in range(1, epochs+1):
        # Sample a subgoal pair
        for it, pair in enumerate(dataloader):
            inps, tgts = pair
            inps = inps.to(device)
            tgts = tgts.to(device)[:, [1, 0]]

            code_preds, act_preds = net(inps)

            action_tgt = tgts[:, 0]
            subgoal_tgt = tgts[:, 1]
            loss = criterion(code_preds, subgoal_tgt) + criterion(act_preds, action_tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_updates += 1
            tots += inps.shape[0]

            corr_acts += (act_preds.argmax(dim=-1) == tgts[:, 0]).sum()
            corr_codes += (code_preds.argmax(dim=-1) == tgts[:, 1]).sum()
            losses.append(loss.item())

            if not (it + 1) % print_freq:
                print("Update: {}, losses: {:.4f}, corr acts: {:.4f}, corr codes: {:.4f}".
                      format(total_updates, np.mean(losses),
                             corr_acts/tots, corr_codes/tots), flush=True)
                if np.mean(losses) < best_loss and not validate:
                    best_loss = np.mean(losses)
                    torch.save(net, f'./out/vqvae_prior_{env}_{save_name}_dataset_best.pth')
                losses, corr_acts, corr_codes, tots = [], 0, 0, 0
                torch.save(net, f'./out/vqvae_prior_{env}_{save_name}_dataset.pth')

        val_losses, val_corr_acts, val_corr_codes, val_tots = [], 0, 0, 0
        if validate:
            for it, pair in enumerate(val_dataloader):
                inps, tgts = pair
                inps = inps.to(device)
                tgts = tgts.to(device)[:, [1, 0]]

                with torch.no_grad():
                    code_preds, act_preds = net(inps)
                    action_tgt = tgts[:, 0]
                    subgoal_tgt = tgts[:, 1]
                    loss = criterion(code_preds, subgoal_tgt) + criterion(act_preds, action_tgt)

                val_tots += inps.shape[0]
                val_corr_acts += (act_preds.argmax(dim=-1) == tgts[:, 0]).sum()
                val_corr_codes += (code_preds.argmax(dim=-1) == tgts[:, 1]).sum()
                val_losses.append(loss.item())

            print("Validation epoch: {}, losses: {:.4f}, corr acts: {:.4f}, corr codes: {:.4f}".
                  format(epoch, np.mean(val_losses), val_corr_acts/val_tots,
                         val_corr_codes/val_tots), flush=True)
            if np.mean(val_losses) < best_loss:
                best_loss = np.mean(val_losses)
                print("Best model, saving")
                torch.save(net, f'./out/vqvae_prior_{env}_{save_name}_dataset_best.pth')

        print(f"Epoch {epoch} complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQVAE Prior')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--ds_name', type=str)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--validate', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    train_prior(args.env, args.name, args.dim, args.K, args.save_name, args.ds_name, args.validate)
