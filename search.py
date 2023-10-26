"""
This module implements and executes the search procedure for evaluation
"""

import argparse
import copy
import heapq
import math
import os
import random
import string
import sys
import time
import numpy as np

import gym
import gym_sokoban
import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, set_start_method

from envs import SlidingTilePuzzle, BoxWorld, TSPEnv, SokobanStepWrapper, SokobanDimWrapper
from utils import TreeNode, ChildIdxPair, MultiQueueTreeNode, get_valid_children_batched, \
    is_terminal_sokoban, is_terminal_stp, is_terminal_bw, is_terminal_tsp, df_wrapper_batched, \
    get_envs, validate_plan, get_h, save_subgoals, save_traj

device = "cuda" if torch.cuda.is_available() else "cpu"
N_ACTIONS = 4

# Built on the code from Orseau & Lelis (2021):
# https://github.com/levilelis/h-levin/blob/master/src/search/bfs_levin.py


class BFSLevin():
    """
    Implements a Levin-based search procedure for HIPS
    """
    def __init__(self, use_learned_heuristic=False,
                 estimated_probability_to_go=False, ada=False):
        self._use_learned_heuristic = use_learned_heuristic
        self._estimated_probability_to_go = estimated_probability_to_go
        self._ada = ada
        self.ll_acts = 0
        self.model_calls = 0

    def get_levin_cost_star(self, child_node, predicted_h):
        """
        Use this to get the evaluation function value
        when the derived hybrid heuristic is used
        """
        if self._use_learned_heuristic:
            predicted_h = max(predicted_h, 0)
            mult = (1 + predicted_h / child_node.get_dist())
            return math.log(child_node.get_g() * mult) - mult * child_node.get_pi()
        # This could be expanded to use rule-based heuristics
        raise NotImplementedError

    def get_levin_cost(self, child_node, predicted_h):
        """
        Use this to get the evaluation function value
        when naïve A*-like heuristics are used
        """

        if self._use_learned_heuristic:
            predicted_h = max(predicted_h, 0)
            return math.log(predicted_h + child_node.get_g()) - child_node.get_pi()
        return math.log(child_node.get_g()) - child_node.get_pi()

    def get_children_with_probs(self, state, args):
        """
        Given a state, return its children, the distances to the children and
        their probabilities
        """

        children = get_valid_children_batched(state, args)
        valid_children_sets = [{ChildIdxPair(c, i) for i, c in enumerate(child)}
                               for child in children]

        children_idx, children = [], []
        for valid_children_set in list(valid_children_sets):
            c_list = []
            c_idx_list = []
            for valid_child_pair in valid_children_set:
                if np.abs(valid_child_pair.child - state).sum() > 0.5:
                    c_list.append(valid_child_pair.child)
                    c_idx_list.append(valid_child_pair.idx)
            children_idx.append(np.array(c_idx_list))
            children.append(np.array(c_list))

        state_inp = torch.FloatTensor(state).to(device)
        if len(list(state_inp.shape)) == 3:
            state_inp = state_inp[None]

        if not state_inp.shape[0] or len(children) == 0 or not children[0].shape[0]:
            return None

        n_acts = N_ACTIONS
        if args.hybrid:
            # Append low-level children
            ll_actions = torch.FloatTensor(np.arange(n_acts)).to(device).repeat(state_inp.shape[0])
            state_inp_repeated = state_inp.repeat_interleave(repeats=4, dim=0)

            if args.model and args.no_env:
                model = torch.load(args.model)
                with torch.no_grad():
                    next_states = model(state_inp_repeated, ll_actions)
                    if args.env.lower() == 'tsp':
                        next_states = (next_states > 0.5).float()
                    else:
                        next_states = next_states.argmax(dim=-1).float()
            else:
                envs = get_envs(state_inp_repeated, args)
                next_states = []
                for i, env in enumerate(envs):
                    if args.env.lower() == 'sokoban':
                        obs = env.step(ll_actions[i].item())
                    elif args.env.lower() == 'tsp':
                        obs, _, _, _ = env.step(ll_actions[i].item(), override_done=True)
                    else:
                        obs, _, _, _ = env.step(ll_actions[i].item())
                    self.ll_acts += 1
                    next_states.append(torch.FloatTensor(obs).to(device)[None])
                next_states = torch.cat(next_states, dim=0)
                if next_states.shape[-1] != next_states.shape[-2]:
                    next_states = next_states.permute(0, 3, 1, 2)

            for i in range(state_inp.shape[0]):
                extra_children = next_states[i*n_acts:(i+1)*n_acts].cpu().numpy()
                children[i] = np.concatenate((np.array(extra_children), children[i]))
                children_idx[i] = np.concatenate((np.arange(0, n_acts), children_idx[i] + n_acts))

        dists, ll_acts, model_calls = df_wrapper_batched(state_inp, children, args)
        self.ll_acts += ll_acts
        self.model_calls += model_calls

        if args.prior:
            prior = torch.load(args.prior)
        else:
            prior = None
        with torch.no_grad():
            if args.hybrid and (prior is None or args.astar or args.gbfs):
                prior_probs = torch.zeros((state_inp.shape[0], args.K + n_acts))
            elif prior is None or args.astar or args.gbfs:
                prior_probs = torch.zeros((state_inp.shape[0], args.K))
            elif prior and args.hybrid:
                code_probs, act_probs = prior(state_inp)
                prior_probs = torch.cat((act_probs, code_probs), dim=-1)
            elif prior:
                prior_probs, _ = prior(state_inp)
            else:
                raise NotImplementedError

        prior_probs_list = []
        children_list = []
        children_idx_list = []
        dists_list = []
        for i, dist in enumerate(dists):
            if args.hybrid:
                dist[:n_acts] = 1
            valid_children = np.where(dist < 1000)[0]
            if args.hybrid:
                p_probs = prior_probs[i][children_idx[i]]
                valid_act_children = valid_children[np.where(valid_children < n_acts)[0]]
                valid_code_children = valid_children[np.where(valid_children >= n_acts)[0]]
                norm_act_probs = F.softmax(p_probs[valid_act_children], dim=-1)
                norm_code_probs = F.softmax(p_probs[valid_code_children], dim=-1)
                if not args.ada:
                    norm_probs = torch.cat((args.epsilon * norm_act_probs,
                                           (1 - args.epsilon) * norm_code_probs),
                                           dim=-1).cpu()
                else:
                    norm_probs = torch.cat((norm_act_probs, norm_code_probs), dim=-1).cpu()
            else:
                p_probs = prior_probs[i][children_idx[i]]
                p_probs = p_probs[valid_children]
                norm_probs = F.softmax(p_probs, dim=-1).cpu()
            children_list.append(children[i][valid_children])
            children_idx_list.append(children_idx[i][valid_children])
            dists_list.append(dist[valid_children])
            log_norm_probs = np.log(norm_probs)
            prior_probs_list.append(log_norm_probs)
        return children_list, children_idx_list, dists_list, prior_probs_list

    def init_node(self, ll_node, *args, **kwargs):
        """
        Init search node. Initialization depends on the queue(s) used, i.e.
        if we have only one queue, or two queues, one for high-level nodes and
        another for low-level nodes
        """
        if self._ada:
            return MultiQueueTreeNode(ll_node, *args, **kwargs)
        return TreeNode(*args, **kwargs)

    def search(self, state, args, env=None):
        """
        Performs Best-First search.
        """
        time_limit = args.time_limit
        verbose = args.verbose

        start_time = time.time()
        self.ll_acts = 0
        self.model_calls = 0

        _open = []
        _open_ll = []
        _closed = set()

        expanded = 0
        generated = 0
        expanded_hl, expanded_ll = 0, 0

        otp = self.get_children_with_probs(state, args)
        if otp is not None:
            children, children_idx, dists, action_distribution = otp
        else:
            end_time = time.time()
            print('Emptied Open List during search')
            return -1, 1, 1, end_time - start_time, None, None, None, None, None, \
                self.ll_acts, self.model_calls

        node = self.init_node(False, None, state, 0, 0, 0, -1, 0, N_ACTIONS)

        heapq.heappush(_open, node)
        _closed.add(state.tobytes())

        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []

        while len(_open) > 0 or len(_open_ll) > 0:
            if len(_open) > 0:
                node = heapq.heappop(_open)
                state_str = node.get_game_state().tobytes()
                if args.model and state_str not in _closed:
                    # It is possible that the state has not been closed
                    assert node.only_lls()
                    _closed.add(state_str)
            else:
                node = heapq.heappop(_open_ll)
                state_str = node.get_game_state().tobytes()
                # We might have problematic states, as we're using a model
                if state_str in _closed and not node.only_lls():
                    continue
                if state_str not in _closed:
                    _closed.add(state_str)

            if len(list(node.get_game_state().shape)) <= 2:
                continue

            expanded += 1
            if args.hybrid and 0 <= node.get_action() < N_ACTIONS:
                expanded_ll += 1
            else:
                expanded_hl += 1

            if verbose:
                print("Expanded", expanded, "Queue len", len(_open),
                      "Action", node.get_action(), "Heuristic", node.get_h(),
                      "HL-expansions", expanded_hl, "LL-expansions", expanded_ll,
                      "Levin cost", node.get_levin_cost(), flush=True)

            end_time = time.time()
            if expanded > args.budget > 0 or end_time - start_time > time_limit > 0:
                return -1, expanded, generated, end_time - start_time, None, None, None, \
                    None, None, self.ll_acts, self.model_calls

            otp = self.get_children_with_probs(node.get_game_state(), args)
            if otp is not None:
                children, children_idx, dists, action_distribution = otp
            else:
                continue
            node.set_probability_distribution_children(action_distribution[0])
            node.set_children_distances(dists[0])
            node.set_children(children[0])
            node.set_children_idx(children_idx[0])
            children = node.get_children()
            children_idx = node.get_children_idx()
            action_distribution = node.get_probability_distribution_children()
            dists = node.get_children_distances()

            for i, child in enumerate(children):
                ll_node = args.hybrid and children_idx[i] < N_ACTIONS
                g_increase = dists[i] if args.step_cost else 1

                child_node = self.init_node(ll_node, node, child,
                                            node.get_pi() + action_distribution[i],
                                            node.get_g() + g_increase, -1, children_idx[i],
                                            node.get_dist() + dists[i], N_ACTIONS)
                if args.env.lower() == 'sokoban':
                    terminal = is_terminal_sokoban(child, 4, args.boxes)
                elif args.env.lower() == 'stp':
                    terminal = is_terminal_stp(child)
                    # Invalid child, skipping
                    if terminal == -1:
                        continue
                elif args.env.lower() == 'bw':
                    terminal = is_terminal_bw(child)
                elif args.env.lower() == 'tsp':
                    terminal = is_terminal_tsp(child)

                if terminal:
                    end_time = time.time()
                    inner_node, states, actions = child_node, [], []
                    while inner_node is not None:
                        states.append(inner_node.get_game_state())
                        if inner_node.get_action() is not None:
                            actions.append(inner_node.get_action())
                        inner_node = inner_node.get_parent()

                    if args.model is not None and not args.no_env:
                        if verbose:
                            print("Validating found plan")
                        val_res, ll_acts = validate_plan(copy.deepcopy(env), states[::-1],
                                                         child_node.get_dist(), args,
                                                         actions[::-1], N_ACTIONS, quiet=True)
                        self.ll_acts += ll_acts
                        if not val_res:
                            if verbose:
                                print("Invalid solution found and discarded")
                            continue

                    node = inner_node
                    rnd_str = ''.join(random.choices(string.ascii_letters, k=6))
                    if args.save_subgoals:
                        save_subgoals(rnd_str, args.env.lower(), child_node)
                    if args.save_trajectory:
                        save_traj(rnd_str, args.env.lower(), child_node)
                    if args.save_subgoals or args.save_trajectory:
                        print("Exiting after saving")
                        sys.exit(0)
                    return child_node.get_dist(), expanded, generated, end_time - start_time, \
                        states[::-1], actions[::-1], child_node.get_pi(), expanded_hl, \
                        expanded_ll, self.ll_acts, self.model_calls

                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(child)

            state_inp = np.array(x_input_of_children_to_be_evaluated)
            predicted_h = get_h(state_inp, args)
            for i, child in enumerate(children_to_be_evaluated):
                generated += 1
                if args.astar:
                    levin_cost = predicted_h[i] + child.get_g()
                elif args.gbfs:
                    levin_cost = predicted_h[i]
                elif self._estimated_probability_to_go:
                    levin_cost = self.get_levin_cost_star(child, predicted_h[i])
                elif self._use_learned_heuristic:
                    levin_cost = self.get_levin_cost(child, predicted_h[i])
                else:
                    levin_cost = self.get_levin_cost(child, None)
                child.set_levin_cost(levin_cost)

                if predicted_h is not None:
                    child.set_h(predicted_h[i])

                game_bytes = child.get_game_state().tobytes()
                if (args.model or game_bytes not in _closed) and child.is_ll_queue_node():
                    heapq.heappush(_open_ll, child)
                elif args.model and child.only_lls():
                    heapq.heappush(_open, child)
                elif game_bytes not in _closed:
                    heapq.heappush(_open, child)
                    _closed.add(game_bytes)

            children_to_be_evaluated.clear()
            x_input_of_children_to_be_evaluated.clear()
        print('Emptied Open List during search')
        end_time = time.time()
        return -1, expanded, generated, end_time - start_time, None, None, None, None, None, \
            self.ll_acts, self.model_calls


def search(_id, i, args):
    """
    Function to first configure and then launch the search
    """
    if args.heuristic and args.simple_heuristic:
        bfs_levin = BFSLevin(use_learned_heuristic=True,
                             estimated_probability_to_go=False,
                             ada=args.ada)
    elif args.heuristic:
        bfs_levin = BFSLevin(use_learned_heuristic=True,
                             estimated_probability_to_go=True,
                             ada=args.ada)
    else:
        bfs_levin = BFSLevin()

    if args.env.lower() == 'sokoban':
        assert 3 <= args.boxes <= 6
        sokoban_envs = {
            3: 'Sokoban-v0',
            4: 'Sokoban-v1',
            5: 'Sokoban-v2',
            6: 'Sokoban-v3',
        }

        env = gym.make(sokoban_envs[args.boxes])
        env = SokobanStepWrapper(env)
        env = SokobanDimWrapper(env, 4)
        state = env.reset().transpose(2, 0, 1)
    elif args.env.lower() == 'stp':
        problem_idx = args.start_cntr + _id*50 + i

        with open('./datasets/puzzles_1000', 'r', encoding="utf-8") as problem_file:
            problem = problem_file.readlines()[problem_idx]

        env = SlidingTilePuzzle(problem)
        state = env.get_image_representation().transpose(2, 0, 1)
    elif args.env.lower() == 'bw':
        num_distractor = 4 if args.difficult else 3
        distractor_length = 3 if args.difficult else 1
        env = BoxWorld(n=12, goal_length=4, num_distractor=num_distractor,
                       distractor_length=distractor_length)
        state = env.reset().transpose(2, 0, 1)
    elif args.env.lower() == 'tsp':
        env = TSPEnv(25, args.cities)
        state = env.reset().transpose(2, 0, 1)

    result = bfs_levin.search(state, args, env)
    sol_len, exp, gen, elapsed_time, states, actions, logp, \
        expanded_hl, expanded_ll, ll_acts, model_calls = result

    # sol_len = -1 if no solution was found
    if sol_len > 0:
        actions = np.asarray(actions)
        if args.hybrid:
            # -1 because the first action (expanding the first node) has been set to equal -1
            # Because of that, there is also one action too much, so we do not need +1 anywhere
            n_ll = (actions < N_ACTIONS).sum() - 1
            n_hl = (actions >= N_ACTIONS).sum()
        else:
            n_ll = 0
            n_hl = len(actions)
        if validate_plan(env, states, sol_len, args, actions, N_ACTIONS)[0]:
            print(f"Solved, {sol_len}, {exp}, {gen}, {elapsed_time}, {n_hl}, {n_ll}, " +
                  f"{logp}, {expanded_hl}, {expanded_ll}, {ll_acts}, {model_calls}")
            return 1
        print(f"Validation fail, {sol_len}, {exp}, {gen}, {elapsed_time}, -1, -1, " +
              f"0, -1, -1, {ll_acts}, {model_calls}")
        return 0
    print(f"Search fail, {sol_len}, {exp}, {gen}, {elapsed_time}, -1, -1, " +
          f"0, -1, -1, {ll_acts}, {model_calls}")
    return 0


def process_main(id_no, args):
    """
    Start the search procedure in the subprocess
    """
    # Set a different seed for each subprocess
    np.random.seed((os.getpid() * int(id_no + time.time())) % 123456789)
    torch.manual_seed((os.getpid() * int(id_no + time.time())) % 123456789)
    random.seed((os.getpid() * int(id_no + time.time())) % 123456789)

    if args.env.lower() == 'sokoban':
        n_iters = 100
    elif args.env.lower() == 'stp':
        n_iters = 100
    elif args.env.lower() == 'bw':
        n_iters = 150
    elif args.env.lower() == 'tsp':
        n_iters = 100

    total_solved = 0
    for i in range(n_iters):
        total_solved += search(id_no, i, args)
        solution_pct = total_solved / (i + 1) * 100
        print(f"Id {id_no}, {i+1} iters, Solved {solution_pct:.2f} %", flush=True)


def main():
    """
    Perform subgoal search with HIPS
    """
    parser = argparse.ArgumentParser(description='Discrete Subgoal Search')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--vqvae', type=str, required=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--prior', type=str)
    parser.add_argument('--heuristic', type=str)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--budget', type=int, default=0)
    parser.add_argument('--time_limit', type=int, default=0)
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--boxes', type=int, default=4)
    parser.add_argument('--max_subtraj', type=int)
    parser.add_argument('--start_cntr', type=int, default=0)
    parser.add_argument('--step_cost', action='store_true', default=False)
    parser.add_argument('--hybrid', action='store_true', default=False)
    parser.add_argument('--astar', action='store_true', default=False)
    parser.add_argument('--simple_heuristic', action='store_true', default=False)
    parser.add_argument('--gbfs', action='store_true', default=False)
    parser.add_argument('--difficult', action='store_true', default=False)
    parser.add_argument('--save_subgoals', action='store_true', default=False)
    parser.add_argument('--save_trajectory', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--ada', action='store_true', default=False)
    parser.add_argument('--no_env', action='store_true', default=False)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--cities', type=int, default=25)
    args = parser.parse_args()
    print(args)

    if not args.model:
        print("Warning, running without a trained dynamics model. " +
              "Resorting to environment simulator")

    if not args.prior:
        assert args.K
        print("Warning, running without a prior. Resorting to uniform prior")

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    n_jobs = args.jobs
    processes = []
    for n_job in range(n_jobs):
        proc = Process(target=process_main, args=(n_job, args))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()
