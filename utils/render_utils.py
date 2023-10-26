"""
Functions for rendering the environments used in this world
"""

import math
import os

import imageio
import numpy as np
import pkg_resources
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from envs import revert_obs

WALLS = 0
GOALS = 1
BOXES = 2
AGENT = 3


def get_img(env, state, comparison_state=None):
    """
    Convert state into a image that can be saved
    """
    if env == 'sokoban':
        state = revert_obs(state.transpose(1, 2, 0), 4)
        return tiny_to_full(state, show=False)
    elif env == 'stp':
        return plot_stp(state.transpose(1, 2, 0), show=False,
                        comparison_state=comparison_state.transpose(1, 2, 0)
                        if comparison_state is not None else None)
    elif env == 'bw':
        return plot_bw(state.transpose(1, 2, 0).astype(np.int32), show=False)
    elif env == 'tsp':
        return plot_tsp(state.transpose(1, 2, 0), show=False)
    else:
        raise NotImplementedError


def save_subgoals(rnd_str, env, node):
    """
    Plot the children of a node
    """
    osdir = './visualizations/' + env + '_' + rnd_str
    os.mkdir(osdir)
    x_dims = {'tsp': 8, 'stp': 8, 'sokoban': 5, 'bw': 5}
    x_dim = x_dims[env]

    while True:
        prev_node = node
        node = node.get_parent()
        if node is None:
            print("Saved all generated subgoals of a successful trajectory with rnd string " +
                  rnd_str)
            return
        img_str = osdir + '/img_n' + str(node.get_g()) + '.pdf'
        images = [get_img(env, node.get_game_state())]

        children = node.get_children()
        children_probs = list(node.get_probability_distribution_children().cpu())
        children = [x for _, x in sorted(zip(children_probs, children),
                                         key=lambda pair: pair[0], reverse=True)]
        chosen_idx = np.argmin(np.abs(prev_node.get_game_state()[None] - np.asarray(children)).
                               sum((1, 2, 3)))
        for child in children:
            images.append(get_img(env, child, node.get_game_state()))
        highlight = chosen_idx+1 if env != 'tsp' else None
        visualize_subgoals(images, img_str, x_dim=x_dim, highlight=highlight)


def save_traj(rnd_str, env, node):
    """
    Plot a subgoal-level trajectory
    """
    path = './visualizations/' + env + '_traj_' + rnd_str + '.pdf'
    images = []
    x_dims = {'tsp': 8, 'stp': 8, 'sokoban': 5, 'bw': 5}
    x_dim = x_dims[env]
    while node is not None:
        parent = node.get_parent()
        images.append(get_img(env, node.get_game_state(),
                              parent.get_game_state() if parent is not None else None))
        node = parent
    images.reverse()
    save_subplots(images, path, x_dim=x_dim)


def visualize_subgoals(images, filename, x_dim=None, titles=False, highlight=None):
    """
    Helper function for generating a grid of images
    """
    if not x_dim:
        dim = math.ceil(np.sqrt(len(images)))
        x_dim = dim
        y_dim = dim
    else:
        y_dim = math.ceil((len(images) - 1) / (x_dim - 1))
    tgt_fig, tgt_axs = plt.subplots(y_dim, x_dim, constrained_layout=True)
    tgt_fig.set_size_inches(x_dim*3 + 2, y_dim*3 + 2)

    init_ctr = 1
    if y_dim > 1:
        tgt_axs[0, 0].imshow(images[0])
        for a in ['top', 'bottom', 'left', 'right']:
            tgt_axs[0, 0].spines[a].set_linewidth(4)
            tgt_axs[0, 0].spines[a].set_color('blue')

        ctr = init_ctr
        for i in range(y_dim):
            for j in range(1, x_dim):
                if ctr < len(images):
                    tgt_axs[i, j].imshow(images[ctr])
                    if titles:
                        tgt_axs[i, j].set_title("Frame " + str(ctr))
                    if ctr == highlight:
                        for a in ['top', 'bottom', 'left', 'right']:
                            tgt_axs[i, j].spines[a].set_linewidth(4)
                            tgt_axs[i, j].spines[a].set_color('red')
                else:
                    tgt_fig.delaxes(tgt_axs[i, j])
                ctr += 1
        for i in range(y_dim):
            for j in range(x_dim):
                tgt_axs[i, j].set_xticks([])
                tgt_axs[i, j].set_yticks([])

        for i in range(1, y_dim):
            tgt_axs[i, 0].remove()
    else:
        # The first image is the current state
        tgt_axs[0].imshow(images[0])
        for a in ['top', 'bottom', 'left', 'right']:
            tgt_axs[0].spines[a].set_linewidth(4)
            tgt_axs[0].spines[a].set_color('blue')

        ctr = init_ctr
        for j in range(1, x_dim):
            if ctr < len(images):
                tgt_axs[j].imshow(images[ctr])
                if titles:
                    tgt_axs[j].set_title("Frame " + str(ctr))
                if ctr == highlight:
                    for a in ['top', 'bottom', 'left', 'right']:
                        tgt_axs[j].spines[a].set_linewidth(4)
                        tgt_axs[j].spines[a].set_color('red')
            else:
                tgt_fig.delaxes(tgt_axs[j])
            ctr += 1
        for j in range(x_dim):
            tgt_axs[j].set_xticks([])
            tgt_axs[j].set_yticks([])

    plt.savefig(filename)
    tgt_fig.clear()
    plt.close(tgt_fig)
    plt.clf()
    plt.cla()


def save_subplots(images, filename, x_dim=None, init_ctr=0, titles=False, highlight=None):
    """
    Helper function for saving images
    """
    if not x_dim:
        dim = math.ceil(np.sqrt(len(images)))
        x_dim = dim
        y_dim = dim
    else:
        y_dim = (len(images) - 1) // x_dim + 1
    tgt_fig, tgt_axs = plt.subplots(y_dim, x_dim, constrained_layout=True)
    tgt_fig.set_size_inches(x_dim*3 + 2, y_dim*3 + 2)

    if y_dim > 1:
        ctr = init_ctr
        for i in range(y_dim):
            for j in range(x_dim):
                if ctr < len(images):
                    tgt_axs[i, j].imshow(images[ctr])
                    if titles:
                        tgt_axs[i, j].set_title("Frame " + str(ctr))
                    if ctr == highlight:
                        for a in ['top', 'bottom', 'left', 'right']:
                            tgt_axs[i, j].spines[a].set_linewidth(6)
                            tgt_axs[i, j].spines[a].set_color('red')
                else:
                    tgt_fig.delaxes(tgt_axs[i, j])
                ctr += 1
        for i in range(y_dim):
            for j in range(x_dim):
                tgt_axs[i, j].set_xticks([])
                tgt_axs[i, j].set_yticks([])
    else:
        ctr = init_ctr
        for j in range(x_dim):
            if ctr < len(images):
                tgt_axs[j].imshow(images[ctr])
                if titles:
                    tgt_axs[j].set_title("Frame " + str(ctr))
                if ctr == highlight:
                    for a in ['top', 'bottom', 'left', 'right']:
                        tgt_axs[j].spines[a].set_linewidth(6)
                        tgt_axs[j].spines[a].set_color('red')
            else:
                tgt_fig.delaxes(tgt_axs[j])
            ctr += 1
        for j in range(x_dim):
            tgt_axs[j].set_xticks([])
            tgt_axs[j].set_yticks([])
    plt.savefig(filename)
    tgt_fig.clear()
    plt.close(tgt_fig)
    plt.clf()
    plt.cla()


def plot_bw(image, show=True, path=None):
    """
    Visualize Box-World environment
    """
    fig, ax = plt.subplots(1, figsize=(5,5))
    ax.set(
        xlim=(-0.5, 13.5), ylim=(13.5, -0.5),
        xticklabels=[],
        yticklabels=[]
    )

    ax.imshow(image)
    ax.autoscale(False)

    # Visualize keys and locks
    key_img = plt.imread("./surface/key_grey.png")
    lock_img = plt.imread("./surface/lock_grey.png")
    coin_img = plt.imread("./surface/coin.png")
    agent_img = plt.imread("./surface/agent.png")

    agent = (image[:,:,0] == 128) & (image[:,:,1] == 128) & (image[:,:,2] == 128)
    y, x = np.where(agent)
    y, x = y[0], x[0]
    extent = (x-0.51, x+0.49, y+0.5, y-0.485)
    ax.imshow(agent_img, extent=extent, interpolation=None)

    gem = (image[:,:,0] == 255) & (image[:,:,1] == 255) & (image[:,:,2] == 255)
    key_or_lock = (
        (image[:,:,0] != 220) | (image[:,:,1] != 220) | (image[:,:,2] != 220)
    ) & ~(
        (image[:,:,0] == 0) & (image[:,:,1] == 0) & (image[:,:,2] == 0)
    ) & ~agent & ~gem
    ys, xs = np.where(key_or_lock)

    same_row = ((ys[:, None] - ys[None,:]) == 0)
    xdiff = (xs[:, None] - xs[None,:])
    klpairs = np.tril(same_row & (xdiff == 1))
    lock_ixs, key_ixs = np.where(klpairs)
    locks = list(zip(ys[lock_ixs], xs[lock_ixs]))
    keys = list(zip(ys[key_ixs], xs[key_ixs]))

    # The lock next to the gem
    if np.any(gem):
        y, x = np.where(gem)
        y, x = y[0], x[0]
        extent = (x-0.51, x+0.49, y+0.5, y-0.485)
        ax.imshow(coin_img, extent=extent, interpolation=None)

        x = x + 1
        if key_or_lock[y, x]:
            locks.append((y, x))

    # The key that the agent has
    if np.any(image[0, 0, :]):
        keys.append((0, 0))

    # The loose key
    all_keys_or_locks = set(zip(*np.where(key_or_lock)))
    keys.extend(list(all_keys_or_locks - set(locks) - set(keys)))

    for (y, x) in locks:
        extent = (x-0.51, x+0.49, y+0.5, y-0.485)
        ax.imshow(lock_img, extent=extent, interpolation=None)

    for (y, x) in keys:
        extent = (x-0.51, x+0.49, y+0.5, y-0.485)
        ax.imshow(key_img, extent=extent, interpolation=None)

    if show:
        plt.show()
    if path:
        plt.savefig(path)

    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ret = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    plt.clf()
    plt.cla()
    return ret


def plot_stp(state, show=True, path=None, comparison_state=None):
    """
    Visualize STP environment
    """
    fig, ax = plt.subplots(1, figsize=(5,5))
    ax.set(
        xlim=(0, 5), ylim=(5, 0),
        xticks=np.arange(6), xticklabels=[],
        yticks=np.arange(6), yticklabels=[]
    )
    ax.grid(True, which='major', linewidth=2)
    ax.tick_params(which='major', length=0)

    ax.grid(True, which='minor')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(which='minor', length=0)

    for r in range(5):
        for c in range(5):
            ix = np.argmax(state[r, c])
            if comparison_state is not None:
                cix = np.argmax(comparison_state[r, c])
            else:
                cix = None
            if ix:
                px, py = c+0.5, r+0.5
                if cix is not None and cix != ix:
                    ax.text(px, py, ix, ha='center', va='center', fontsize=40, color='red')
                else:
                    ax.text(px, py, ix, ha='center', va='center', fontsize=40)

    if show or path:
        if show:
            plt.show()
        if path:
            plt.savefig(path)

    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ret = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    plt.clf()
    plt.cla()
    return ret


def plot_tsp(state, show=True, path=None):
    """
    Visualize TSP environment
    """
    fig, ax = plt.subplots(1, figsize=(5,5))
    agent = np.where(state[:,:,2])
    packed_state = np.packbits(state[:,:,[0, 1, 3]].astype(bool),
                               axis=2, bitorder='little').squeeze()

    cmap = ['w'] * 8
    cmap[0] = 'w'
    cmap[1] = 'r'
    cmap[3] = 'g'
    cmap[7] = 'k'
    cmap = ListedColormap(cmap)
    ax.matshow(packed_state, cmap=cmap)

    agent_circle = plt.Circle((agent[1], agent[0]), 1, color='k', fill=False, linewidth=2)
    ax.add_patch(agent_circle)

    ax.set_yticks([])
    ax.set_xticks([])
    if show or path:
        if show:
            plt.show()
        if path:
            plt.savefig(path, bbox_inches='tight')

    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ret = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    plt.clf()
    plt.cla()
    return ret


def tiny_to_full(image_repr, show=True, path=None):
    """
    Creates an RGB image of the Sokoban state.
    :param image_repr:
    :return:
    """
    resource_package = __name__

    box_filename = pkg_resources.resource_filename(resource_package,
                                                   '/'.join(('surface', 'box.png')))
    box = imageio.imread(box_filename)

    box_on_target_filename = \
        pkg_resources.resource_filename(resource_package,
                                        '/'.join(('surface', 'box_on_target.png')))
    box_on_target = imageio.imread(box_on_target_filename)

    box_target_filename = pkg_resources.resource_filename(resource_package,
                                                          '/'.join(('surface', 'box_target.png')))
    box_target = imageio.imread(box_target_filename)

    floor_filename = pkg_resources.resource_filename(resource_package,
                                                     '/'.join(('surface', 'floor.png')))
    floor = imageio.imread(floor_filename)

    player_filename = pkg_resources.resource_filename(resource_package,
                                                      '/'.join(('surface', 'player.png')))
    player = imageio.imread(player_filename)

    player_on_target_filename = \
        pkg_resources.resource_filename(resource_package,
                                        '/'.join(('surface', 'player_on_target.png')))
    player_on_target = imageio.imread(player_on_target_filename)

    wall_filename = pkg_resources.resource_filename(resource_package,
                                                    '/'.join(('surface', 'wall.png')))
    wall = imageio.imread(wall_filename)

    surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

    wall = [0, 0, 0]
    floor = [243, 248, 238]
    box_target = [254, 126, 125]
    box_on_target = [254, 95, 56]
    box = [142, 121, 56]
    player = [160, 212, 56]
    player_on_target = [219, 212, 56]

    # Assemble the new rgb_state, with all loaded images
    state_rgb = np.zeros(shape=(image_repr.shape[0] * 16,
                                image_repr.shape[1] * 16, 3), dtype=np.uint8)
    for i in range(image_repr.shape[0]):
        x_i = i * 16

        for j in range(image_repr.shape[1]):
            y_j = j * 16

            channels = list(image_repr[i, j])

            if channels == wall:
                surface_id = 0
            elif channels == floor:
                surface_id = 1
            elif channels == box_target:
                surface_id = 2
            elif channels == box_on_target:
                surface_id = 3
            elif channels == box:
                surface_id = 4
            elif channels == player:
                surface_id = 5
            elif channels == player_on_target:
                surface_id = 6
            else:
                print(channels)
                raise RuntimeError

            state_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = surfaces[surface_id]

    if show or path:
        plt.imshow(state_rgb)
        if show:
            plt.show()
        if path:
            plt.savefig(path)
        plt.close()
        plt.clf()
        plt.cla()
    return state_rgb


TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

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
