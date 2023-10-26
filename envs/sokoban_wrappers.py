"""
Utility functions for the Sokoban environment
"""
import gym
from gym.spaces import Box
import numpy as np


class SokobanStepWrapper(gym.Wrapper):
    """
    Wrapper that requests the Sokoban represntation in the grid-like format
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        d = self.env.observation_space.shape[0] // 16
        self.observation_space = Box(low=0, high=255, shape=(d, d, 3), dtype=np.uint8)

    def reset(self):
        return self.env.reset(render_mode='tiny_rgb_array')

    def step(self, action):
        return self.env.step(action)


def revert_obs(obs, dim):
    """
    Converts the 4-channeled binary or 7-channeled one-hot representation of Sokoban
    back into the RGB image representation
    """
    if dim == 3:
        return obs

    wall = tuple([0, 0, 0])
    floor = tuple([243, 248, 238])
    box_target = tuple([254, 126, 125])
    box_on_target = tuple([254, 95, 56])
    box = tuple([142, 121, 56])
    player = tuple([160, 212, 56])
    player_on_target = tuple([219, 212, 56])

    arr = np.zeros((obs.shape[0], obs.shape[1], 3))

    fd_wall = tuple([1, 0, 0, 0])
    fd_floor = tuple([0, 0, 0, 0])
    fd_box_target = tuple([0, 1, 0, 0])
    fd_box_on_target = tuple([0, 1, 1, 0])
    fd_box = tuple([0, 0, 1, 0])
    fd_player = tuple([0, 0, 0, 1])
    fd_player_on_target = tuple([0, 1, 0, 1])

    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            # 4-dimensional: wall, target, box, agent
            if dim == 4:
                channels = tuple(obs[i, j])
                if channels == fd_wall:
                    arr[i, j] = wall
                elif channels == fd_box_target:
                    arr[i, j] = box_target
                elif channels == fd_box_on_target:
                    arr[i, j] = box_on_target
                elif channels == fd_box:
                    arr[i, j] = box
                elif channels == fd_player:
                    arr[i, j] = player
                elif channels == fd_player_on_target:
                    arr[i, j] = player_on_target
                elif channels == fd_floor:
                    arr[i, j] = floor
                else:
                    raise RuntimeError
            # 7-dimensional: wall, floor, box_target, box_on_target, box, player, player_on_target
            elif dim == 7:
                if obs[i, j, 0] == 1:
                    arr[i, j] = wall
                elif obs[i, j, 1] == 1:
                    arr[i, j] = floor
                elif obs[i, j, 2] == 1:
                    arr[i, j] = box_target
                elif obs[i, j, 3] == 1:
                    arr[i, j] = box_on_target
                elif obs[i, j, 4] == 1:
                    arr[i, j] = box
                elif obs[i, j, 5] == 1:
                    arr[i, j] = player
                elif obs[i, j, 6] == 1:
                    arr[i, j] = player_on_target
                else:
                    raise RuntimeError
            else:
                raise RuntimeError
    return arr


def modify_obs(obs, dim):
    """
    Converts the RGB-representation of Sokoban into a 4-channeled binary representation or
    7-channeled one-hot encoding
    """
    if dim == 3:
        return obs

    wall = tuple([0, 0, 0])
    floor = tuple([243, 248, 238])
    box_target = tuple([254, 126, 125])
    box_on_target = tuple([254, 95, 56])
    box = tuple([142, 121, 56])
    player = tuple([160, 212, 56])
    player_on_target = tuple([219, 212, 56])

    arr = np.zeros((obs.shape[0], obs.shape[1], dim))

    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            channels = tuple(obs[i, j])
            if channels == wall:
                arr[i, j, 0] = 1
            elif channels == floor:
                if dim == 7:
                    arr[i, j, 1] = 1
            elif channels == box_target:
                if dim == 7:
                    arr[i, j, 2] = 1
                elif dim == 4:
                    arr[i, j, 1] = 1
            elif channels == box_on_target:
                if dim == 7:
                    arr[i, j, 3] = 1
                elif dim == 4:
                    arr[i, j, 1] = 1
                    arr[i, j, 2] = 1
            elif channels == box:
                if dim == 7:
                    arr[i, j, 4] = 1
                elif dim == 4:
                    arr[i, j, 2] = 1
            elif channels == player:
                if dim == 7:
                    arr[i, j, 5] = 1
                elif dim == 4:
                    arr[i, j, 3] = 1
            elif channels == player_on_target:
                if dim == 7:
                    arr[i, j, 6] = 1
                elif dim == 4:
                    arr[i, j, 1] = 1
                    arr[i, j, 3] = 1
            else:
                print(channels)
                raise RuntimeError
    return arr


class SokobanDimWrapper(gym.Wrapper):
    """
    Wrapper that creates a self.dim-channeled representation for the learning agent
    3 channels: RGB
    4 channels: binary representation (wall, target, box, player)
    7 channels: one-hot representation
    """
    def __init__(self, env, dim=3):
        super().__init__(env)
        self.env = env
        self.dim = dim

    def reset(self):
        obs = self.env.reset()
        return modify_obs(obs, self.dim)

    def step(self, action):
        obs, r, d, i = self.env.step(action)
        return modify_obs(obs, self.dim), r, d, i


class SokobanRewWrapper(gym.Wrapper):
    """
    Wrapper that makes the Sokoban rewards sparse
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        o, r, d, i = self.env.step(action)
        if r < 10:
            r = 0
        else:
            r = 1
        return o, r, d, i
