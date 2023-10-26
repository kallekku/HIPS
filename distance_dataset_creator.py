"""
Given the trajectory datasets, convert them into a format for training
the value function
"""
import pickle
import numpy as np
import torch


def distance_dataset():
    """
    Implements the dataset modification
    """
    with open('./datasets/sokoban_obs.pkl', 'rb') as handle:
    # with open('./datasets/stp_obs.pkl', 'rb') as handle:
    # with open('./datasets/bw_obs.pkl', 'rb') as handle:
        x_train = pickle.load(handle)
    with open('./datasets/sokoban_obs_val.pkl', 'rb') as handle:
    # with open('./datasets/stp_obs_val.pkl', 'rb') as handle:
    # with open('./datasets/bw_obs_val.pkl', 'rb') as handle:
        x_val = pickle.load(handle)

    for j, ds in enumerate([x_train, x_val]):
        X, Y = [], []
        max_len = 0
        for i in range(ds.shape[0]):
            if not (i+1) % 100:
                print(i)
            x = ds[i]
            X.append(torch.FloatTensor(x))
            Y.append(torch.LongTensor(np.arange(x.shape[0]-1, -1, -1)))
            max_len = max(max_len, x.shape[0]-1)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        print("max_len", max_len)
        X_save_name = "sokoban_obs_dists.pkl" if not j else "sokoban_obs_dists_val.pkl"
        # X_save_name = "stp_obs_dists.pkl" if not j else "stp_obs_dists_val.pkl"
        # X_save_name = "bw_obs_dists.pkl" if not j else "bw_obs_dists_val.pkl"
        pickle.dump(X, open('./datasets/' + X_save_name, "wb"))
        Y_save_name = "sokoban_dists.pkl" if not j else "sokoban_dists_val.pkl"
        # Y_save_name = "stp_dists.pkl" if not j else "stp_dists_val.pkl"
        # Y_save_name = "bw_dists.pkl" if not j else "bw_dists_val.pkl"
        pickle.dump(Y, open('./datasets/' + Y_save_name, "wb"))


if __name__ == "__main__":
    distance_dataset()
