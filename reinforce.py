"""
This module implements the training of the detector network and
conditional low-level policy
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from models import SokobanDetectorNetwork, SokobanPolicy, STPPolicy, STPDetectorNetwork, \
    BWDetectorNetwork, BWPolicy, TSPPolicy, TSPDetectorNetwork
from datasets import SokobanTrajectoryDataset, TrajectoryDataset, TSPTrajectoryDataset
from utils import get_acts

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_reinforce(args):
    """
    Train the detector network and conditional low-level policy
    """
    lr = 1e-3
    detector_lr = 1e-3
    epochs = 1000
    gamma = 0.99
    print_freq = 1000
    norm = nn.InstanceNorm2d
    run_id = args.run_id
    alpha = args.alpha
    H = args.H
    env = args.env.lower()

    if env == 'sokoban':
        resnet_blocks = [2, 2, 2]
        resnet_channels = 16
        _k = 10
    elif env == 'stp':
        resnet_blocks = [6]
        resnet_channels = 64
        _k = 25
    elif env == 'bw':
        resnet_blocks = [2, 2, 2]
        resnet_channels = 64
        _k = 10
        N = 3
        D = 3
    elif env == 'tsp':
        resnet_blocks = [2, 2, 2]
        resnet_channels = 16
        _k = 25
        print_freq = 100

    def get_policy():
        if env == 'sokoban':
            policy = SokobanPolicy(resnet_blocks, n_channels=resnet_channels,
                                   num_classes=4, norm=norm, input_nc=4).to(device)
        elif env == 'stp':
            policy = STPPolicy(resnet_blocks, n_channels=resnet_channels,
                               num_classes=4, norm=norm, input_nc=25).to(device)
        elif env == 'bw':
            policy = BWPolicy(N=N, D=D, norm=norm, n_actions=4).to(device)
        elif env == 'tsp':
            policy = TSPPolicy(resnet_blocks, n_channels=resnet_channels,
                               num_classes=4, norm=norm).to(device)
        policy.train()
        policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
        return policy, policy_opt

    if env == 'sokoban':
        detector = SokobanDetectorNetwork(norm, input_nc=4).to(device)
    elif env == 'stp':
        detector = STPDetectorNetwork(resnet_channels, norm, input_nc=25).to(device)
    elif env == 'bw':
        detector = BWDetectorNetwork(N=N, D=D, norm=norm, input_nc=3).to(device)
    elif env == 'tsp':
        detector = TSPDetectorNetwork(norm).to(device)
    detector.train()
    detector_opt = torch.optim.Adam(detector.parameters(),
                                    lr=detector_lr)
    detector_criterion = nn.CrossEntropyLoss(reduction='none')

    best_perf = np.inf
    total_loss, corr, tot, tot_kfs = 0, 0, 0, 0

    if env == 'sokoban':
        dataset = SokobanTrajectoryDataset('./datasets/sokoban_obs.pkl',
                                           './datasets/sokoban_acts.pkl')
    elif env == 'stp':
        dataset = TrajectoryDataset('./datasets/stp_obs.pkl', './datasets/stp_acts.pkl')
    elif env == 'bw':
        dataset = TrajectoryDataset('./datasets/bw_obs.pkl', './datasets/bw_acts.pkl')
    elif env == 'tsp':
        max_len = 450
        dataset_len = 1000
        dataset = TSPTrajectoryDataset(dataset_len=dataset_len, max_len=max_len,
                                       standard_dataset=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    def pretrain():
        first_corr = None
        total_loss, corr, tot = 0, 0, 0
        for it, (obs, actions) in enumerate(dataloader):
            obs = obs.squeeze(dim=0).to(device)
            actions = actions.squeeze(dim=0).to(device)

            encoded = np.zeros((obs.shape[0]))

            size = min(_k, obs.shape[0]-2)
            sels = np.random.choice(np.arange(1, obs.shape[0]-1, 1),
                                    size=size, replace=False)
            encoded[sels] = 1
            encoded = torch.FloatTensor(encoded).to(device)
            acts = get_acts(policy, encoded, obs)
            loss = detector_criterion(acts, actions)

            if env != 'bw':
                policy_opt.zero_grad()
                loss.sum().backward()
                policy_opt.step()
            else:
                loss.sum().backward()
                if not (it + 1) % 10:
                    policy_opt.step()
                    policy_opt.zero_grad()

            n_corr = (acts.argmax(dim=-1) == actions).sum().item()
            n_acts = actions.shape[0]

            total_loss += loss.sum().item()
            corr += n_corr
            tot += n_acts

            if not (it + 1) % print_freq:
                corr_pct = corr / tot * 100
                print("Pretrain iter {}, loss: {:.3f},".
                      format(it + 1, total_loss/print_freq) +
                      " correctness: {:.2f} %".format(corr_pct), flush=True)
                total_loss, corr, tot = 0, 0, 0
                if not first_corr:
                    first_corr = corr_pct
                last_corr = corr_pct
            if it > 10000:
                policy_opt.zero_grad()
                break
        return first_corr, last_corr

    # Re-initialize the low-level policy network until it starts learning
    # Normally a re-initialization isn't needed, but sometimes it might
    # be necessary
    first_corr, last_corr = 1, 1
    while first_corr * 1.2 > last_corr:
        policy, policy_opt = get_policy()
        first_corr, last_corr = pretrain()

    for epoch in range(1, epochs+1):
        for it, (obs, actions) in enumerate(dataloader):
            obs = obs.squeeze(dim=0).to(device)
            actions = actions.squeeze(dim=0).to(device)

            log_probs = []
            vals = []
            kfs = []
            i = 0
            while i < obs.shape[0]-1:
                inp = obs[i+1:i+H+1]
                inp = torch.cat((obs[i].repeat(inp.shape[0], 1, 1, 1), inp), dim=1)
                logits, v = detector(inp, compute_val=True)
                probs = F.softmax(logits, dim=-1)

                dist = Categorical(probs)
                sample = dist.sample()
                kfs.append(i + sample.item() + 1)
                log_probs.append(torch.log(probs[sample]).unsqueeze(dim=0))
                vals.append(v[sample])
                i += sample.item() + 1

            encoded = np.zeros((obs.shape[0]))
            encoded[kfs] = 1
            encoded = torch.FloatTensor(encoded).to(device)

            acts = get_acts(policy, encoded, obs)
            loss = detector_criterion(acts, actions)

            policy_opt.zero_grad()
            loss.sum().backward()
            policy_opt.step()

            prev = 0
            rewards = []
            for k in kfs:
                r = -loss[prev:k].sum() - alpha
                rewards.append(r)
                prev = k

            R = 0
            returns = []
            for i in range(len(rewards)-1, -1, -1):
                R = rewards[i] + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns).to(device)

            vals = torch.cat(vals, dim=0)
            log_probs = torch.cat(log_probs, dim=0)

            advantage = returns - vals
            policy_loss = (-log_probs * advantage.detach()).sum()
            value_loss = F.smooth_l1_loss(vals, returns.detach())

            detector_opt.zero_grad()
            detector_loss = policy_loss + value_loss
            detector_loss.backward()
            nn.utils.clip_grad_norm_(detector.parameters(), 0.5)
            detector_opt.step()

            n_corr = (acts.argmax(dim=-1) == actions).sum().item()
            n_acts = actions.shape[0]

            total_loss += loss.sum().item()
            corr += n_corr
            tot += n_acts
            tot_kfs += len(kfs)
            if not (it + 1) % print_freq:
                print("Iteration {}, loss: {:.3f}, correctness: {:.2f} %".
                      format(it + 1, total_loss / (it+1), corr / tot * 100) +
                      ", subgoals: {:.2f}".format(tot_kfs / (it+1)), flush=True)
            dataset_size = it

        print("Epoch {}, loss: {:.3f}, correctness: {:.2f} %".
              format(epoch, total_loss / dataset_size, corr / tot * 100) +
              ", subgoals: {:.2f}".format(tot_kfs / dataset_size), flush=True)
        torch.save(policy, f'./out/reinforce_{env}_{run_id}_policy_{alpha}.pth')
        torch.save(detector, f'./out/reinforce_{env}_{run_id}_detector_{alpha}.pth')
        if total_loss < best_perf:
            print("Best performance, saving")
            best_perf = total_loss
            torch.save(policy, f'./out/reinforce_{env}_{run_id}_policy_best_{alpha}.pth')
            torch.save(detector, f'./out/reinforce_{env}_{run_id}_detector_best_{alpha}.pth')
        total_loss, corr, tot, tot_kfs = 0, 0, 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Discrete Reinforce')
    parser.add_argument('--run_id', type=str, default="0")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--env', type=str, required=True)
    args = parser.parse_args()
    print(args)
    if args.env.lower() == 'tsp' and args.H != 50:
        print(f"WARNING! Recommended H for TSP is 50. The current value is {args.H}")
    if args.env.lower() == 'tsp' and args.alpha != 0.05:
        print(f"WARNING! Recommended alpha for TSP is 0.05. The current value is {args.alpha}")
    train_reinforce(args)
