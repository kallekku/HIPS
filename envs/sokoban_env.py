"""
Implements SokobanEnv
"""

import numpy as np
import torch

from .sokoban_wrappers import modify_obs, revert_obs

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

wall = (0, 0, 0)
floor = (243, 248, 238)
box_target = (254, 126, 125)
box_on_target = (254, 95, 56)
box = (142, 121, 56)
player = (160, 212, 56)
player_on_target = (219, 212, 56)


class SokobanEnv:
    """
    Simulates a game of Sokoban
    """
    def __init__(self, img):
        # Image input c, h, w
        if isinstance(img, torch.Tensor):
            self.img = img.cpu().long().numpy()
        elif isinstance(img, np.ndarray):
            self.img = img.astype(int)
        else:
            raise RuntimeError

        self.dim = img.shape[0]
        if self.img.shape[0] != 3:
            i = np.transpose(self.img, (1, 2, 0))
            i = revert_obs(i, self.dim)
            self.img = np.transpose(i, (2, 0, 1))

        self.player = np.asarray(np.where(np.logical_or(self.img[0] == player[0],
                                                        self.img[1] == player_on_target[1])))\
            .T.squeeze(axis=0)

    def replace_player_square(self, i, j):
        """
        Delete the player from the image representation of the game
        """
        if tuple(self.img[:, i, j]) == player:
            self.img[:, i, j] = floor
        elif tuple(self.img[:, i, j]) == player_on_target:
            self.img[:, i, j] = box_target
        else:
            raise RuntimeError

    def step(self, act):
        i = self.player[0]
        j = self.player[1]

        if isinstance(act, torch.Tensor):
            act = act.item()
        assert act in [0, 1, 2, 3]

        player_change = CHANGE_COORDINATES[act]
        tgt_i = i + CHANGE_COORDINATES[act][0]
        tgt_i2 = i + 2*CHANGE_COORDINATES[act][0]
        tgt_j = j + CHANGE_COORDINATES[act][1]
        tgt_j2 = j + 2*CHANGE_COORDINATES[act][1]

        channels = tuple(self.img[:, tgt_i, tgt_j])
        if channels == wall:
            pass
        elif channels == floor or channels == box_target:
            self.player += player_change
            if channels == floor:
                self.img[:, tgt_i, tgt_j] = player
            else:
                self.img[:, tgt_i, tgt_j] = player_on_target
            self.replace_player_square(i, j)
        elif channels == box_on_target or channels == box:
            # Check if we can move or not
            channels_behind = tuple(self.img[:, tgt_i2, tgt_j2])
            if channels_behind in [floor, box_target]:
                # Can move
                self.player += player_change
                if channels == box_on_target:
                    self.img[:, tgt_i, tgt_j] = player_on_target
                else:
                    self.img[:, tgt_i, tgt_j] = player
                if channels_behind == floor:
                    self.img[:, tgt_i2, tgt_j2] = box
                else:
                    self.img[:, tgt_i2, tgt_j2] = box_on_target
                self.replace_player_square(i, j)
            else:
                pass
                # Can't move, do nothing
        else:
            raise RuntimeError

        assert (np.asarray(np.where(np.logical_or(self.img[0] == player[0],
                                                  self.img[1] == player_on_target[1])))
                .T.squeeze(axis=0) == self.player).all()

        # c, h ,w
        i = np.transpose(self.img.copy(), (1, 2, 0))
        # h, w, c
        i = modify_obs(i, self.dim)
        # h, w, c
        i = np.transpose(i, (2, 0, 1))
        # c, h, w
        return i

