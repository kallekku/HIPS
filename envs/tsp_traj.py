"""
Creates random TSP trajectories
"""
import numpy as np
from .tsp_env import TSPEnv


def get_traj(size=5, targets=4, actions=False, s=None, image=True):
    """
    Code for sampling a TSP trajectory from the teacher dataset
    """

    # Create an environment
    env = TSPEnv(size, targets)
    if s is None:
        obs_init = env.reset(image=image)
    else:
        obs_init = s.numpy()
        env.set_state(s)

    # Get the start city
    cities = env.get_cities()
    start_index = env.get_start()

    # Randomize the order of cities
    arr = list(range(0, start_index)) + list(range(start_index+1, len(cities)))
    np.random.shuffle(arr)
    trajectory = [start_index] + arr + [start_index]

    action_seq = []
    visited = np.asarray([False] * cities.shape[0])

    # Start generating the trajectory by creating the action sequence
    curr_index = trajectory[0]
    traj_index = 1
    while visited.sum() < targets:
        next_index = trajectory[traj_index]

        # If some city has been accidentally visited on the way
        # to another city, do not visit that city again
        while visited[next_index]:
            traj_index += 1
            next_index = trajectory[traj_index]

        to_next = cities[next_index] - cities[curr_index]
        curr_loc = cities[curr_index].copy()

        dist = np.sum(np.abs(to_next))

        to_move = np.asarray([
            -to_next[1] if to_next[1] < 0 else 0,
            to_next[1] if to_next[1] > 0 else 0,
            to_next[0] if to_next[0] > 0 else 0,
            -to_next[0] if to_next[0] < 0 else 0
        ])
        while dist > 0:
            if to_move[0]:
                act = 0
            elif to_move[1]:
                act = 1
            elif to_move[2]:
                act = 2
            elif to_move[3]:
                act = 3
            else:
                raise RuntimeError
            to_move[act] -= 1
            dist -= 1
            action_seq.append(act)

            if act == 0:
                curr_loc[1] -= 1
            elif act == 1:
                curr_loc[1] += 1
            elif act == 2:
                curr_loc[0] += 1
            elif act == 3:
                curr_loc[0] -= 1

            city = np.where((cities == curr_loc).all(axis=1))[0]
            if len(city) > 0:
                if city[0] != start_index or visited.sum() == targets - 1:
                    visited[city[0]] = True

        assert (curr_loc == cities[next_index]).all()
        curr_index = next_index
        traj_index += 1

    # Use the action sequence to get the observations
    obs = [obs_init]
    done = False
    i = 0
    while not done:
        next_state, _, done, _ = env.step(action_seq[i], image=image)
        obs.append(next_state)
        i += 1

    if len(action_seq) != len(obs) - 1:
        action_seq = action_seq[:len(obs)-1]
    act_retval = np.asarray(action_seq) if actions else None

    return np.asarray(obs), act_retval
