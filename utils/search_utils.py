"""
Utility functions for subgoal search
"""
import numpy as np
import torch

from envs import SokobanEnv, SlidingTilePuzzle, BoxWorld, TSPEnv, revert_obs

device = "cuda" if torch.cuda.is_available() else "cpu"

# TSP Channels
CITIES = 0
VISITED_CITIES = 1
AGENT = 2
GOAL = 3


def get_h(state, args):
    """
    Get heuristic value, i.e. predicted distance to goal in low-level actions
    """
    if args.heuristic:
        heuristic = torch.load(args.heuristic)
        if state.ndim == 3:
            predicted_h = heuristic.predict(state[None])
        elif state.ndim >= 4:
            predicted_h = heuristic.predict(state)
        else:
            predicted_h = np.inf
    else:
        predicted_h = None
    return predicted_h


def is_terminal_stp(state):
    """
    Terminal state identification for STP
    """
    env = SlidingTilePuzzle()
    try:
        env.set_image(state)
    except AssertionError:
        return -1
    return env.is_solution()


def is_terminal_bw(state):
    """
    Terminal state identification for BW
    """
    state = state.transpose(1, 2, 0)
    return tuple(state[0, 0]) == (0, 0, 0) and \
        len(np.asarray(np.where((state == (255, 255, 255)).all(axis=2))).T) == 0


def is_terminal_sokoban(state, channels, boxes):
    """
    Terminal state identification for Sokoban
    """
    state = state.transpose((1, 2, 0))
    try:
        state = revert_obs(state, channels)
    except RuntimeError:
        return False
    state = state.transpose((2, 0, 1))
    n_boxes = np.asarray(np.where(state[0] == 142)).T
    n_empty_goals = np.asarray(np.where(state[1] == 126)).T
    n_boxes_on_goals = np.asarray(np.where(state[1] == 95)).T
    player_on_target = np.asarray(np.where(state[0] == 219)).T
    return len(n_boxes) == 0 and len(n_empty_goals) == 0 and \
        len(player_on_target) == 0 and len(n_boxes_on_goals) == boxes


def is_terminal_tsp(s):
    """
    Terminal state identification for TSP
    """
    n_cities_remaining = (s[CITIES]-s[VISITED_CITIES]).sum()
    if n_cities_remaining > 0.5:
        return False
    elif np.abs(s[AGENT] - s[GOAL]).sum() < 1e-6:
        return True
    else:
        return False


def get_valid_children_batched(states, args=None):
    """
    Generate possible children of for given states usign the VQVAE
    """
    if args and args.vqvae:
        net = torch.load(args.vqvae)
    else:
        pass
    K = net.codebook.K
    s_tsr = torch.FloatTensor(states).to(device)

    # Generate a subgoal prediction for each code
    with torch.no_grad():
        if len(list(s_tsr.shape)) == 3:
            s_tsr = s_tsr[None]
        n_states = s_tsr.shape[0]
        s_tsr = s_tsr.repeat_interleave(K, dim=0)
        K_input = torch.from_numpy(np.arange(K)).to(device).repeat(n_states)
        x_tilde = net.decode(s_tsr, K_input)

    if x_tilde.ndim == s_tsr.ndim:
        x_tilde = x_tilde.cpu().numpy()
        x_tilde = (x_tilde > 0.5).astype(int)
    else:
        x_tilde = x_tilde.argmax(dim=-1)
        x_tilde = x_tilde.cpu().numpy()
    x_tilde = x_tilde.reshape(n_states, K, *s_tsr.shape[-3:])
    return x_tilde


def df_wrapper_batched(states, targets, args):
    """
    Given a set of states and targets, find the distances from the states to the targets
    To do this, we perform rollout using the subgoal-conditioned low-level policy
    """
    # States is tensor, targets a list of numpy arrays
    state_list = []
    ll_acts = 0
    model_calls = 0
    tgt_sizes = [t.shape[0] for t in targets]
    for i, state in enumerate(states):
        state_list.append(state.repeat(targets[i].shape[0], 1, 1, 1))
    states = torch.cat(state_list, dim=0)
    targets = torch.FloatTensor(np.concatenate(targets)).to(device)

    n_subgoals = targets.shape[0]
    # 1000 as dummy for unreachable
    dists = np.ones(n_subgoals) * 1000
    dones = torch.zeros(n_subgoals).bool()
    curr_state = states

    policy = torch.load(args.policy)
    if args.model:
        model = torch.load(args.model)
    elif args.env.lower() == 'sokoban':
        envs = [SokobanEnv(s) for s in states]
    elif args.env.lower() == 'stp':
        envs = [SlidingTilePuzzle().set_image(s) for s in states]
    elif args.env.lower() == 'bw':
        envs = []
        for state in states:
            _s = state.permute(1, 2, 0).cpu().numpy()
            agent_pos = np.array(np.where((_s == (128, 128, 128)).all(axis=2))).T.squeeze(axis=0)
            envs.append(BoxWorld(n=12, goal_length=4, num_distractor=3,
                                 distractor_length=1).set(_s, agent_pos))
    elif args.env.lower() == 'tsp':
        envs = [TSPEnv(size=25, targets=25) for s in states]
        for i, state in enumerate(states):
            try:
                envs[i].set_state(state.permute(1, 2, 0))
            except ValueError:
                # Invalid state
                dones[i] = True

    if args.max_subtraj:
        max_subtraj = args.max_subtraj
    elif args.env.lower() == 'tsp':
        max_subtraj = 50
    else:
        max_subtraj = 40

    for j in range(max_subtraj):
        # Predict actions
        with torch.no_grad():
            act = policy(torch.cat((curr_state, targets), dim=1)).argmax(dim=-1)

        # Predict next state with dynamics model
        if args.model:
            with torch.no_grad():
                next_state = model(curr_state, act.float())
                model_calls += 1
            if next_state.ndim == curr_state.ndim:
                next_state = (next_state > 0.5).float()
            else:
                next_state = next_state.argmax(dim=-1).float()

            curr_state = next_state
        else:
            curr_state = []
            for i, env in enumerate(envs):
                if args.env.lower() == 'sokoban':
                    obs = env.step(act[i].item())
                elif args.env.lower() == 'tsp':
                    obs, _, _, _ = env.step(act[i].item(), override_done=True)
                else:
                    obs, _, _, _ = env.step(act[i].item())
                ll_acts += 1
                curr_state.append(torch.FloatTensor(obs).to(device)[None])
            curr_state = torch.cat(curr_state, dim=0)

        if curr_state.shape[-1] != curr_state.shape[-2]:
            curr_state = curr_state.permute(0, 3, 1, 2)
        if targets.shape[-1] != targets.shape[-2]:
            targets = targets.permute(0, 3, 1, 2)

        # Check if a goal was reached
        finished = np.bitwise_or(dones,
                                 (torch.abs(curr_state - targets).flatten(1, 3).
                                  sum(dim=-1) < 0.5).cpu())
        recently_finished = np.bitwise_and(finished, np.invert(dones))

        # If the goal was reached in this frame, update the distance
        dists[np.where(recently_finished)[0]] = j+1
        dones[np.where(finished)[0]] = True
        if dones.all():
            break

    dists = torch.LongTensor(dists).split(tgt_sizes)
    dists = [d.numpy() for d in dists]
    return dists, ll_acts, model_calls


def validate_plan(env, states, tgt_steps, args=None, actions=None, n_acts=None, quiet=False):
    """
    Once a plan has been generated, validate that it works
    """
    if args and args.policy:
        policy = torch.load(args.policy)
    else:
        pass

    states = torch.from_numpy(np.asarray(states)).to(device).float()
    curr_state = states[0]
    sg_i = 1
    steps = 0
    ll_acts = 0

    visited_states = [curr_state]
    while True:
        subgoal = states[sg_i]

        # Predict actions with low-level policy and step the environment
        if args.hybrid and actions[sg_i] < n_acts:
            act = actions[sg_i]
        else:
            with torch.no_grad():
                inp = torch.cat((curr_state[None], subgoal[None]), dim=1).float()
                act = policy(inp).argmax(dim=-1).item()
                if args.env.lower() == 'sokoban':
                    act += 1
        curr_state, _, done, _ = env.step(act)
        steps += 1
        ll_acts += 1
        if args.env.lower() == 'stp':
            curr_state = torch.from_numpy(curr_state).to(device)
        else:
            curr_state = torch.from_numpy(curr_state).permute(2, 0, 1).to(device)
        visited_states.append(curr_state)

        if done:
            break
        if torch.abs(curr_state - subgoal).sum() < 1/2:
            # If subgoal has been encountered, move to the following subgoal
            sg_i += 1
            subgoal = states[sg_i]
        elif steps > tgt_steps + 5:
            if not quiet:
                print("Validation error")
            return False, ll_acts
    return True, ll_acts


class DummyEnv:
    """
    If the subgoal created by the VQVAE is invalid, environment object
    cannot be initialized. Then, we create an instance of DummyEnv instead.
    """
    def __init__(self, state, fdim=False):
        if isinstance(state, torch.Tensor):
            self.state = state.cpu().long().numpy()
        elif isinstance(state, np.ndarray):
            self.state = state.astype(int)
        self.fdim = fdim

    def step(self, act):
        """
        Dummy step, do nothing
        """
        if self.fdim:
            return self.state, None, None, None
        return self.state


def get_envs(states, args):
    """
    Given a list of states, initialize environment objects
    """
    if args.env.lower() == 'sokoban':
        envs = []
        for state in states:
            try:
                envs.append(SokobanEnv(state))
            except ValueError:
                envs.append(DummyEnv(state))
    elif args.env.lower() == 'stp':
        envs = [SlidingTilePuzzle().set_image(s) for s in states]
    elif args.env.lower() == 'bw':
        envs = []
        for state in states:
            try:
                _s = state.permute(1, 2, 0).cpu().numpy()
                agent_pos = np.array(np.where((_s == (128, 128, 128)).all(axis=2))).T.\
                    squeeze(axis=0)
                envs.append(BoxWorld(n=12, goal_length=4, num_distractor=3,
                                     distractor_length=1).set(_s, agent_pos))
            except ValueError:
                envs.append(DummyEnv(state, fdim=True))
    elif args.env.lower() == 'tsp':
        envs = [TSPEnv(size=25, targets=25) for state in states]
        for i, state in enumerate(states):
            try:
                envs[i].set_state(state.permute(1, 2, 0))
            except ValueError:
                pass
    return envs
