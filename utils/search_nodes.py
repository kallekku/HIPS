"""
Implements classes for the subgoal search
"""

# Built on the code from Orseau & Lelis (2021):
# https://github.com/levilelis/h-levin/blob/master/src/search/bfs_levin.py


class ChildIdxPair:
    """
    Simple class for keeping track of which child corresponds to which index
    when they are held in a set
    """
    def __init__(self, child, idx):
        self.child = child
        self.idx = idx

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return tuple(self.child.flatten()).__hash__()


class TreeNode:
    """
    Node for tree search
    """
    def __init__(self, parent, game_state, pi, g, levin_cost, action, dist, n_acts):
        self._game_state = game_state
        self._pi = pi
        self._g = g
        self._h = None
        self._levin_cost = levin_cost
        self._action = action
        self._parent = parent
        self._probability_distribution_a = None
        self._children = None
        self._children_idx = None
        self._children_distances = None
        self._dist = dist

        # Check if the path to the root only consists of low-level actions
        self._only_lls = self._action < n_acts and (not self._parent or self._parent.only_lls())

    def __eq__(self, other):
        """
        Equality function for priority queue
        """
        return self._game_state == other._game_state

    def __lt__(self, other):
        """
        Comparison function for priority queue
        """
        return self._levin_cost < other._levin_cost

    def __hash__(self):
        """
        Hash function for keeping track of closed nodes
        """
        return self._game_state.tobytes().__hash__()

    def set_probability_distribution_children(self, distr):
        """
        Set pi for node
        """
        self._probability_distribution_a = distr

    def get_probability_distribution_children(self):
        """
        Get pi for node
        """
        return self._probability_distribution_a

    def set_children_distances(self, dists):
        """
        Set distances to children
        """
        self._children_distances = dists

    def get_children_distances(self):
        """
        Get distances to children
        """
        return self._children_distances

    def set_children(self, children):
        """
        Set children
        """
        self._children = children

    def get_children(self):
        """
        Get children
        """
        return self._children

    def set_children_idx(self, children_idx):
        """
        Set the actions (indices of subgoals), which the children correspond to
        """
        self._children_idx = children_idx

    def get_children_idx(self):
        """
        Get the actions (indices of subgoals), which the children correspond to
        """
        return self._children_idx

    def set_levin_cost(self, cost):
        """
        Set cost function to be used in the priority queue
        """
        self._levin_cost = cost
    
    def get_levin_cost(self):
        """
        Get cost function to be used in the priority queue
        """
        return self._levin_cost

    def set_h(self, h):
        """
        Set heuristic value, i.e. estimated distance to solution in terms of low-level steps
        """
        self._h = h

    def get_h(self):
        """
        Get heuristic value, i.e. estimated distance to solution in terms of low-level steps
        """
        return self._h

    def get_dist(self):
        """
        Get distance from root in low-level steps
        """
        return self._dist

    def get_pi(self):
        """
        Get joint node probability, i.e. p(this|parent) * p(parent|its_parent) etc.
        """
        return self._pi

    def get_g(self):
        """
        Get the node cost, denoted by g in the paper
        """
        return self._g

    def get_game_state(self):
        """
        Self-explanatory
        """
        return self._game_state

    def get_parent(self):
        """
        Self-explanatory
        """
        return self._parent

    def get_action(self):
        """
        Get the action taken by the parent to arrive at this node
        """
        return self._action

    def is_ll_queue_node(self):
        """
        This node should not be set in a priority queue that only consists of low-level nodes
        Overwritten in a subclass that expands this class if necessary
        """
        return False

    def only_lls(self):
        """
        Check if the path to the root only consists of low-level actions
        """
        return self._only_lls


class MultiQueueTreeNode(TreeNode):
    """
    Tree node that can be used when there are many queues
    """
    def __init__(self, ll_queue_node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ll_queue_node = ll_queue_node

    def __lt__(self, other):
        """
        Comparison method for priority queue
        """
        if self._ll_queue_node == other._ll_queue_node:
            return self._levin_cost < other._levin_cost
        return self._ll_queue_node < other._ll_queue_node

    def is_ll_queue_node(self):
        """
        Returns if this is a node for the low-level-nodes-only queue
        """
        return self._ll_queue_node
