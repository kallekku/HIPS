"""
Trains a VQVAE for subgoal generation
"""

import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models import SokobanVQVAE, STPVQVAE, BWVQVAE, TSPVQVAE
from datasets import SokobanTrajectoryDataset, TrajectoryDataset, TSPTrajectoryDataset,\
    tsp_vqvae_collate

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_vqvae(env, name, continuous, dim, K, H, save_name=None, ds_name=None,
                output_dim=2, validate=False):
    """
    Implements the VQVAE training
    """
    print_freq = 500
    beta = 0.1
    lr = 2e-4
    epochs = 10000

    if save_name is None:
        save_name = name
    if ds_name is None:
        ds_name = name

    # If continuous, create a new VQVAE, otherwise load a pretrained VQVAE
    if continuous:
        if env.lower() == 'sokoban':
            channels = 4
            net = SokobanVQVAE(dim, K, input_nc=channels, output_dim=output_dim).to(device)
        elif env.lower() == 'stp':
            channels = 25
            net = STPVQVAE(dim, K, input_nc=channels, output_dim=output_dim).to(device)
        elif env.lower() == 'bw':
            channels = 3
            net = BWVQVAE(dim, K, input_nc=channels, output_dim=output_dim).to(device)
        elif env.lower() == 'tsp':
            channels = 4
            net = TSPVQVAE(dim, K).to(device)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        try:
            net = torch.load(f'./out/vqvae_{env}_{name}_dataset_continuous_best.pth',
                             map_location=device)
        except FileNotFoundError:
            net = torch.load(f'./out/vqvae_{env}_{name}_dataset_continuous.pth',
                             map_location=device)
    criterion = nn.CrossEntropyLoss()

    losses = []
    recon_losses = []
    vq_losses = []
    commit_losses = []
    n_codes = []
    best_loss = np.inf

    # Discrete
    if not continuous and env.lower() != 'tsp':
        # Load the dataset
        with open(f'./datasets/vqvae_{env}_{ds_name}_inputs.pkl', 'rb') as f:
            inps = pickle.load(f).cpu()
        with open(f'./datasets/vqvae_{env}_{ds_name}_conds.pkl', 'rb') as f:
            conds = pickle.load(f).cpu()
        dataset = TensorDataset(inps, conds)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        if validate:
            # Load the validation dataset

            with open(f'./datasets/vqvae_{env}_{ds_name}_inputs_val.pkl', 'rb') as f:
                inps = pickle.load(f).cpu()
            with open(f'./datasets/vqvae_{env}_{ds_name}_conds_val.pkl', 'rb') as f:
                conds = pickle.load(f).cpu()
            val_dataset = TensorDataset(inps, conds)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    # Continuous Sokoban
    elif env.lower() == 'sokoban':
        ds = SokobanTrajectoryDataset('./datasets/sokoban_obs.pkl',
                                      './datasets/sokoban_acts.pkl')
        dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)
        if validate:
            val_ds = SokobanTrajectoryDataset('./datasets/sokoban_obs_val.pkl',
                                              './datasets/sokoban_acts_val.pkl')
            val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)
    # Continuous STP
    elif env.lower() == 'stp':
        ds = TrajectoryDataset('./datasets/stp_obs.pkl', './datasets/stp_acts.pkl')
        dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)
        if validate:
            val_ds = TrajectoryDataset('./datasets/stp_obs_val.pkl', './datasets/stp_acts_val.pkl')
            val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)
    # Continuous BW
    elif env.lower() == 'bw':
        ds = TrajectoryDataset('./datasets/bw_obs.pkl', './datasets/bw_acts.pkl')
        dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)
        if validate:
            val_ds = TrajectoryDataset('./datasets/bw_obs_val.pkl', './datasets/bw_acts_val.pkl')
            val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)
    elif env.lower() == 'tsp':
        ds = TSPTrajectoryDataset(dataset_len=10000, max_len=450, abl_vqvae_dataset=True)
        dataloader = DataLoader(ds, batch_size=4, shuffle=True,
                                num_workers=4, collate_fn=tsp_vqvae_collate)
        if validate:
            val_ds = TSPTrajectoryDataset(dataset_len=500, max_len=450, abl_vqvae_dataset=True)
            val_dataloader = DataLoader(val_ds, batch_size=4, shuffle=True,
                                        num_workers=4, collate_fn=tsp_vqvae_collate)

    total_updates = 0
    for epoch in range(1, epochs+1):
        losses, recon_losses, vq_losses, commit_losses, n_codes, codes = [], [], [], [], [], set()
        for it, pair in enumerate(dataloader):
            if continuous and env.lower() != 'tsp':
                obs, _ = pair
                obs = obs.squeeze(dim=0).to(device)

                high = np.asarray(np.arange(1, obs.shape[0]))
                low = high - H
                low = np.where(low >= 0, low, 0)
                cond_idxs = np.random.randint(low, high)
                conds = obs[cond_idxs]
                inps = obs[1:]
            else:
                inps, conds = pair

            inps = inps.to(device)
            conds = conds.to(device)

            total_updates += 1
            # The encoder takes both elements of the pair as input
            encoder_input = torch.cat((conds, inps), dim=1)
            # The decoder takes as input just the previous subgoal
            decoder_input = conds

            # Reconstruct the latter subgoals
            x_tilde, z_e_x, z_q_x, n_c, c = net(encoder_input,
                                                decoder_input,
                                                continuous=continuous)

            # Compute reconstruction loss
            if env.lower() == 'tsp':
                loss_recon = F.mse_loss(x_tilde, inps)
            else:
                loss_recon = criterion(x_tilde.flatten(0, 3), inps.flatten().long())

            # Compute VQVAE loss
            if continuous:
                loss = loss_recon
            else:
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
                loss = loss_recon + loss_vq + beta * loss_commit
                # If discrete and the first iterations, keep collecting
                # encoder outputs for initialization
                if epoch == 1 and it < 10:
                    continue
                if epoch == 1 and it == 10:
                    # Initialize discrete codes with KMeans++
                    net.reinit_codes(dim, K)
                    net = net.to(device)
                    opt = torch.optim.Adam(net.parameters(), lr=lr)
                    continue

            # Update VQVAE
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            recon_losses.append(loss_recon.item())
            if not continuous:
                vq_losses.append(loss_vq.item())
                commit_losses.append(loss_commit.item())
            else:
                vq_losses.append(0)
                commit_losses.append(0)
            n_codes.append(n_c)
            codes.update(set(c))

            if not (it + 1) % print_freq:
                print("Update: {}, losses: {:.4f}, recon_losses: {:.4f}, ".
                      format(total_updates, np.mean(losses),
                             np.mean(recon_losses)) +
                      "vq losses: {:.4f}, commit losses: {:.4f}, ".
                      format(np.mean(vq_losses), np.mean(commit_losses)) +
                      "codes used: {:.2f}".
                      format(len(codes)),
                      flush=True)
                losses, recon_losses, vq_losses, commit_losses, n_codes, codes = \
                    [], [], [], [], [], set()
                if continuous:
                    torch.save(net, f'./out/vqvae_{env}_{save_name}_dataset_continuous.pth')
                else:
                    torch.save(net, f'./out/vqvae_{env}_{save_name}_dataset_discrete.pth')

        if validate:
            losses, recon_losses, vq_losses, commit_losses = [], [], [], []
            for it, pair in enumerate(val_dataloader):
                if continuous and env.lower() != 'tsp':
                    obs, _ = pair
                    obs = obs.squeeze(dim=0).to(device)
                    high = np.asarray(np.arange(1, obs.shape[0]))
                    low = high - H
                    low = np.where(low >= 0, low, 0)
                    cond_idxs = np.random.randint(low, high)
                    conds = obs[cond_idxs]
                    inps = obs[1:]
                else:
                    inps, conds = pair
                inps = inps.to(device)
                conds = conds.to(device)
                encoder_input = torch.cat((conds, inps), dim=1)
                decoder_input = conds
                with torch.no_grad():
                    x_tilde, z_e_x, z_q_x, _, _ = net(encoder_input,
                                                      decoder_input,
                                                      continuous=continuous)
                if env.lower() == 'tsp':
                    loss_recon = F.mse_loss(x_tilde, inps)
                else:
                    loss_recon = criterion(x_tilde.flatten(0, 3), inps.flatten().long())
                if continuous:
                    loss = loss_recon
                else:
                    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
                    loss = loss_recon + loss_vq + beta * loss_commit
                losses.append(loss.item())
                recon_losses.append(loss_recon.item())
                if not continuous:
                    vq_losses.append(loss_vq.item())
                    commit_losses.append(loss_commit.item())
                else:
                    vq_losses.append(0)
                    commit_losses.append(0)
            print("Validation: losses: {:.4f}, recon_losses: {:.4f}, ".
                  format(np.mean(losses), np.mean(recon_losses)) +
                  "vq losses: {:.4f}, commit losses: {:.4f}, ".
                  format(np.mean(vq_losses), np.mean(commit_losses)),
                  flush=True)
            loss = np.mean(losses)
            if continuous and loss < best_loss:
                torch.save(net, f'./out/vqvae_{env}_{save_name}_dataset_continuous_best.pth')
                print("Saving best model", flush=True)
                best_loss = loss
            elif not continuous and loss < best_loss:
                torch.save(net, f'./out/vqvae_{env}_{save_name}_dataset_discrete_best.pth')
                print("Saving best model", flush=True)
                best_loss = loss
        print(f"Epoch {epoch} complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Discrete Sokoban VQVAE')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--ds_name', type=str)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--env', type=str, required=True)
    args = parser.parse_args()
    print(args)
    if args.env == 'tsp' and args.K != 32:
        print(f"WARNING! Recommended K for TSP is 32. The current value is {args.K}")
    if args.env == 'tsp' and args.dim != 64:
        print(f"WARNING! Recommended dim for TSP is 64. The current value is {args.dim}")
    if args.env == 'tsp' and args.H != 50:
        print(f"WARNING! Recommended H for TSP is 50. The current value is {args.H}")
    train_vqvae(args.env, args.name, args.continuous, args.dim, args.K, args.H,
                args.save_name, args.ds_name, args.output_dim, args.validate)
