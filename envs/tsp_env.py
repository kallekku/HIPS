"""
Implements the traveling salesman environment
"""
import gym
import numpy as np
import torch

from gym.spaces.discrete import Discrete
from gym.spaces import Box

# Left, right, down, up
N_ACTIONS = 4
# Cities, Visited_cities, Agent, Goal
N_CHANNELS = 4
# Rewards
R_FINAL = 1
R_DEFAULT = 0
# Channels
CITIES = 0
VISITED_CITIES = 1
AGENT = 2
GOAL = 3
# Actions
LEFT = 0
RIGHT = 1
DOWN = 2
UP = 3
# Action lookup
ACTION_LOOKUP = {
    LEFT: 'left',
    RIGHT: 'right',
    DOWN: 'down',
    UP: 'up'
}


class TSPEnv(gym.Env):
    """
    The traveling salesman environment
    """
    def __init__(self, size=10, targets=6):
        super().__init__()

        self.size = size
        self.targets = targets

        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.observation_space = Box(low=0, high=1,
                                     shape=(size, size, N_CHANNELS),
                                     dtype=np.bool_)

        self.done = True
        self.window = None

        self.cities = None
        self.target_ind = None
        self.goal = None
        self.agent = None
        self.visited_cities = None
        self.arr = None

        self.reset()

    def reset(self, image=True):
        cities = np.random.choice(self.size**2-1, self.targets, replace=False)
        x_coords = cities % self.size
        y_coords = cities // self.size

        self.cities = np.asarray(list(zip(x_coords, y_coords)))
        self.target_ind = np.random.randint(self.targets)
        self.goal = np.array(self.cities[self.target_ind])
        self.agent = np.array(self.goal)
        self.visited_cities = np.array([False for k in range(self.targets)])
        self.visited_cities[self.target_ind] = True

        self.arr = np.zeros((self.size, self.size, N_CHANNELS))
        self.done = False
        self.arr[x_coords, y_coords, CITIES] = 1
        self.arr[self.agent[0], self.agent[1], :] = 1

        if image:
            return np.array(self.arr)
        return np.array((self.cities.copy(), self.visited_cities.copy(),
                         self.agent.copy(), self.goal.copy()),
                        dtype=object)

    def coordinates_to_image(self, state):
        """
        If state is given as coordinates, turn it into a image
        """
        size = state[0].shape[0]
        arr = np.zeros((size, size, N_CHANNELS))
        x_coords = state[0][:, 0]
        y_coords = state[0][:, 1]
        arr[x_coords, y_coords, CITIES] = 1
        arr[x_coords, y_coords, VISITED_CITIES] = state[1]
        arr[state[2][0], state[2][1], AGENT] = 1
        arr[state[3][0], state[3][1], GOAL] = 1
        return arr

    def set_state(self, state):
        """
        Set a state in the environment
        """
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = state.copy()

        self.arr = state
        self.done = False

        agent_coords = np.where(self.arr[:, :, AGENT] == 1)
        self.agent = np.array([agent_coords[0],
                               agent_coords[1]]
                              ).squeeze(axis=1)

        city_coords = np.array(np.where(self.arr[:, :, CITIES] == 1)).T
        self.cities = city_coords

        goal_coords = np.array(np.where(self.arr[:, :, GOAL] == 1)
                               ).squeeze(axis=1)
        ind = int(np.where((self.cities == goal_coords).all(axis=1))[0])
        self.goal = goal_coords
        self.target_ind = ind

        visited_cities = np.array([False] * self.cities.shape[0])
        visited_coords = np.array(np.where(self.arr[:, :, VISITED_CITIES] == 1)
                                  ).T

        for i in range(visited_coords.shape[0]):
            ind = int(np.where((self.cities == visited_coords[i])
                               .all(axis=1))[0])
            visited_cities[ind] = True
        self.visited_cities = visited_cities

    def step(self, action, image=True, override_done=False):
        if isinstance(action, np.ndarray):
            action = action[0]

        assert action in ACTION_LOOKUP
        if not override_done:
            assert not self.done

        reward = R_DEFAULT

        self.arr[self.agent[0], self.agent[1], AGENT] = 0
        if action == LEFT and self.agent[1] > 0:
            self.agent[1] -= 1
        elif action == RIGHT and self.agent[1] < self.size - 1:
            self.agent[1] += 1
        elif action == UP and self.agent[0] > 0:
            self.agent[0] -= 1
        elif action == DOWN and self.agent[0] < self.size - 1:
            self.agent[0] += 1
        self.arr[self.agent[0], self.agent[1], AGENT] = 1

        if self.agent.tolist() in self.cities.tolist():
            ind = np.where((self.cities == self.agent).all(axis=1))[0][0]
            if not self.visited_cities[ind]:
                self.visited_cities[ind] = True
                self.arr[self.agent[0], self.agent[1], VISITED_CITIES] = 1

        if self.visited_cities.all() and \
           np.array_equal(self.agent, self.goal) and not self.done:
            self.done = True
            reward = R_FINAL

        if image:
            state = np.array(self.arr)
        else:
            state = np.array((self.cities.copy(), self.visited_cities.copy(),
                              self.agent.copy(), self.goal.copy()), dtype=object)
        return state, reward, self.done, {}

    def get_cities(self):
        """
        Return the city locations
        """
        return self.cities

    def get_start(self):
        """
        Return the index of the starting city
        """
        return self.target_ind


def all_valid_children(s):
    """
    Get all children of a state
    """
    acts = [0, 1, 2, 3]
    s = s.transpose(1, 2, 0)
    states = []
    for a in acts:
        e = TSPEnv(size=3, targets=2)
        e.set_state(s)
        ns, _, _, _ = e.step(a)
        ns = ns.transpose(2, 0, 1)
        states.append(ns)
    return states


def all_valid_subgoal_children(s):
    """
    Get all valid subgoal-level children of a state. Used for oracle MCTS
    """
    dim = s.shape[1]

    s = s.copy()
    s[AGENT] = np.zeros((dim, dim))

    n_cities_remaining = (s[CITIES]-s[VISITED_CITIES]).sum()
    if n_cities_remaining == 0:
        goal_coords = np.where(s[GOAL] == 1)
        s[AGENT, goal_coords[0], goal_coords[1]] = 1
        return [s]

    cities_to_be_visited = np.array(np.where((s[CITIES] -
                                    s[VISITED_CITIES]) == 1)).T
    children = []
    for i in range(n_cities_remaining.astype(int)):
        _s = s.copy()
        x, y = cities_to_be_visited[i]
        _s[AGENT, x, y] = 1
        _s[VISITED_CITIES, x, y] = 1
        children.append(_s)
    return children


def distance_between_frames(f1, f2):
    """
    Approximation and doesn't cover all types of incorrectly generated frames
    This isn't a problem, as this piece of code is only used together with
    Oracle MCTS that generates correct subgoals
    """
    if isinstance(f1, torch.Tensor):
        n_agents_in_f1 = f1[AGENT].sum().long().item()
        n_agents_in_f2 = f2[AGENT].sum().long().item()
        n_goals_in_f1 = f1[GOAL].sum().long().item()
        n_goals_in_f2 = f2[GOAL].sum().long().item()

        f1 = f1.detach().cpu().numpy()
        f2 = f2.detach().cpu().numpy()
    elif isinstance(f1, np.ndarray):
        n_agents_in_f1 = f1[AGENT].sum().astype(int)
        n_agents_in_f2 = f2[AGENT].sum().astype(int)
        n_goals_in_f1 = f1[GOAL].sum().astype(int)
        n_goals_in_f2 = f2[GOAL].sum().astype(int)

    new_cities_visited = f2[VISITED_CITIES].sum().astype(int) - \
        f1[VISITED_CITIES].sum().astype(int)

    if n_agents_in_f1 != 1 or n_agents_in_f2 != 1:
        return 1000
    if n_goals_in_f1 != 1 or n_goals_in_f2 != 1:
        return 1000
    if np.abs(f1[CITIES] - f2[CITIES]).sum() >= 1e-6:
        return 1000

    if np.bitwise_and(f2[VISITED_CITIES].astype(bool),
                      np.invert(f2[CITIES].astype(bool))).sum():
        return 1000
    if np.bitwise_and(f1[VISITED_CITIES].astype(bool),
                      np.invert(f1[CITIES].astype(bool))).sum():
        return 1000

    if np.bitwise_and(f2[AGENT].astype(bool),
                      np.invert(f2[CITIES].astype(bool))).sum():
        return 1000
    if np.bitwise_and(f2[AGENT].astype(bool),
                      np.invert(f2[VISITED_CITIES].astype(bool))).sum():
        return 1000
    if np.bitwise_and(f1[AGENT].astype(bool),
                      np.invert(f1[CITIES].astype(bool))).sum():
        return 1000
    if np.bitwise_and(f1[AGENT].astype(bool),
                      np.invert(f1[VISITED_CITIES].astype(bool))).sum():
        return 1000

    cities_and_visited_f1 = np.bitwise_or(f1[CITIES].astype(bool),
                                          f1[VISITED_CITIES].astype(bool))
    cities_and_visited_f2 = np.bitwise_or(f2[CITIES].astype(bool),
                                          f2[VISITED_CITIES].astype(bool))
    if cities_and_visited_f1.sum() != cities_and_visited_f2.sum():
        return 1000

    assert cities_and_visited_f1.sum() == 25
    assert cities_and_visited_f2.sum() == 25

    agent_old_coords = np.array(np.where(f1[AGENT] == 1)).T
    agent_new_coords = np.array(np.where(f2[AGENT] == 1)).T
    if new_cities_visited <= 1:
        dist = np.abs(agent_new_coords - agent_old_coords).sum()
    elif new_cities_visited == 2:
        int_stop = np.abs(f2[VISITED_CITIES] - f1[VISITED_CITIES])
        locs = np.array(np.where(int_stop == 1)).T
        if locs[0, 0] == agent_new_coords[0, 0] and \
                locs[0, 1] == agent_new_coords[0, 1]:
            dist = np.abs(locs[1] - agent_old_coords).sum() + \
                np.abs(locs[0] - locs[1]).sum()
        elif locs[1, 0] == agent_new_coords[0, 0] and \
                locs[1, 1] == agent_new_coords[0, 1]:
            dist = np.abs(locs[0] - agent_old_coords).sum() + \
                np.abs(locs[0] - locs[1]).sum()
        else:
            return new_cities_visited * 50
    elif new_cities_visited == 3:
        int_stop = np.abs(f2[VISITED_CITIES] - f1[VISITED_CITIES])
        locs = np.array(np.where(int_stop == 1)).T
        if locs[0, 0] == agent_new_coords[0, 0] and \
                locs[0, 1] == agent_new_coords[0, 1]:
            dist_a = np.abs(locs[1] - agent_old_coords).sum() + \
                np.abs(locs[2] - locs[1]).sum() + \
                np.abs(locs[2] - locs[0]).sum()
            dist_b = np.abs(locs[2] - agent_old_coords).sum() + \
                np.abs(locs[2] - locs[1]).sum() + \
                np.abs(locs[1] - locs[0]).sum()
        elif locs[1, 0] == agent_new_coords[0, 0] and \
                locs[1, 1] == agent_new_coords[0, 1]:
            dist_a = np.abs(locs[0] - agent_old_coords).sum() + \
                np.abs(locs[2] - locs[0]).sum() + \
                np.abs(locs[2] - locs[1]).sum()
            dist_b = np.abs(locs[2] - agent_old_coords).sum() + \
                np.abs(locs[2] - locs[0]).sum() + \
                np.abs(locs[1] - locs[0]).sum()
        elif locs[2, 0] == agent_new_coords[0, 0] and \
                locs[2, 1] == agent_new_coords[0, 1]:
            dist_a = np.abs(locs[0] - agent_old_coords).sum() + \
                np.abs(locs[1] - locs[0]).sum() + \
                np.abs(locs[2] - locs[1]).sum()
            dist_b = np.abs(locs[1] - agent_old_coords).sum() + \
                np.abs(locs[1] - locs[0]).sum() + \
                np.abs(locs[2] - locs[0]).sum()
        else:
            return new_cities_visited * 50
        dist = min(dist_a, dist_b)
    elif new_cities_visited >= 4:
        dist = new_cities_visited * 50
    return dist
