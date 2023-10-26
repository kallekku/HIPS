"""
Create trajectories with random actions for training better dynamics models
Note: this is not necessary for reproducing the results, as random trajectories
are contained in all_datasets.zip and the google drive
"""
import pickle

import gym
import gym_sokoban
from gym.spaces import Discrete, Box
import numpy as np

from envs import BoxWorld, SlidingTilePuzzle, SokobanStepWrapper, SokobanDimWrapper


class STPWrapper:
    """
    Wrapper for Sliding Tile Puzzles
    """
    def __init__(self, path, start_cnt=None):
        file = open(path, 'r')
        instances = file.readlines()
        self.instances = [i.strip() for i in instances]
        self.n_instances = len(self.instances)
        self.ctr = 0 if not start_cnt else start_cnt
        self.dim = int(np.sqrt(len(self.instances[0].split(' '))))
        self.observation_space = Box(low=0, high=24, shape=(self.dim, self.dim, self.dim**2),
                                     dtype=np.uint8)
        self.action_space = Discrete(4)
        self.env = None

    def reset(self):
        instance = self.instances[self.ctr]
        self.ctr = (self.ctr + 1) % self.n_instances
        self.env = SlidingTilePuzzle(instance)
        return self.env.get_image_representation().transpose(2, 0, 1)

    def step(self, action):
        return self.env.step(action)


class TransposeWrapper:
    """
    Transposes an observation into channels x H x W
    """
    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset().transpose(2, 0, 1)

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        return obs.transpose(2, 0, 1), rew, done, info


def create(env_name, n_trajs, val=False):
    """
    Create random trajectories
    """
    if env_name == 'sokoban':
        env = gym.make('Sokoban-v1')
        env = SokobanStepWrapper(env)
        env = SokobanDimWrapper(env, 4)
        env = TransposeWrapper(env)
        act_range = np.arange(1, 5)
        upper = 120
    elif env_name == 'stp':
        if val:
            env = STPWrapper('./datasets/puzzles_50000', 20000)
        else:
            env = STPWrapper('./datasets/puzzles_50000', 10000)
        act_range = np.arange(0, 4)
        upper = 300
    elif env_name == 'bw':
        env = BoxWorld(n=12, goal_length=4, num_distractor=3, distractor_length=1)
        env = TransposeWrapper(env)
        act_range = np.arange(0, 4)
        upper = 75

    tot_a = []
    tot_o = []

    for i in range(n_trajs):
        done = False
        obs = env.reset()
        states = [obs]
        acts = []
        j = 0
        while not done:
            act = np.random.choice(act_range)
            obs, _, _, _ = env.step(act)
            acts.append(act)
            states.append(obs)
            j += 1
            if j >= upper:
                break
        obs = np.asarray(states)
        acts = np.asarray(acts)
        tot_a.append(acts)
        tot_o.append(obs)

    tot_a = np.array(tot_a)
    tot_o = np.array(tot_o)

    if val:
        pickle.dump(tot_a, open(f"./datasets/{env_name}_random_acts_val.pkl", "ab"))
        pickle.dump(tot_o, open(f"./datasets/{env_name}_random_obs_val.pkl", "ab"))
    else:
        pickle.dump(tot_a, open(f"./datasets/{env_name}_random_acts.pkl", "ab"))
        pickle.dump(tot_o, open(f"./datasets/{env_name}_random_obs.pkl", "ab"))


if __name__ == "__main__":
    # create('sokoban', 100)
    # create('stp', 100)
    # create('bw', 100)
    # create('sokoban', 20, val=True)
    # create('stp', 20, val=True)
    # create('bw', 20, val=True)
    pass
