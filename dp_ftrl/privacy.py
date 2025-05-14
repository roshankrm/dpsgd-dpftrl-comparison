# -----------------------------------------------------------------------------
# This code uses the privacy computation and optimizer from:
#
#   google-research/DP-FTRL  
#   https://github.com/google-research/DP-FTRL

# -----------------------------------------------------------------------------


"""Privacy computation for DP-FTRL."""
import numpy as np

def convert_gaussian_renyi_to_dp(sigma, delta, verbose=True):
    """
    Convert from RDP to DP for a Gaussian mechanism.
    :param sigma: the algorithm guarantees (alpha, alpha/(2*sigma^2))-RDP
    :param delta: target DP delta
    :param verbose: whether to print message
    :return: the DP epsilon
    """
    
    # Candidate Renyi orders α ∈ (1, 200)
    alphas = np.arange(1, 200, 0.1)[1:]
    
    # For each α compute ε(α) via the RDP→DP conversion formula
    epss = alphas / 2 / sigma**2 - (np.log(delta*(alphas - 1)) - alphas * np.log(1 - 1/alphas)) / (alphas - 1)
    
    # Pick α that minimizes ε
    idx = np.nanargmin(epss)
    if verbose and idx == len(alphas) - 1:
        print('The best alpha is the last one. Consider increasing the range of alpha.')
    eps = epss[idx]
    return eps

def get_total_sensitivity_sq_same_order(steps_per_epoch, epochs, extra_steps):
    """
    Get the squared sensitivty for a tree where we fix the order of batches for all epochs.
    :param steps_per_epoch: number of steps per epoch
    :param epochs: number of epochs in the tree
    :param extra_steps: number of virtual steps
    :return: squared sensitivity
    """
    # get first layer as a list of counters
    layer = []
    for _ in range(epochs):
        layer += [{ss: 1} for ss in range(steps_per_epoch)]
    layer += [{-1: 1} for _ in range(extra_steps)]  # extra steps denoted as -1

    # sensitivity_sq[i] will record the total sensitivity wrt batch i
    sensitivity_sq_all = [0] * steps_per_epoch

    # update sensitivity_sq with a given layer
    def update_sensitivity_sq(current_layer):
        # Add each node’s squared count to its batch index
        for node in current_layer:
            for ss in node:
                if ss != -1:
                    sensitivity_sq_all[ss] += node[ss] ** 2

    update_sensitivity_sq(layer)  # get sensitivity for the first layer

    while len(layer) > 1:
        layer_new = []  # merge every two consecutive nodes to get the next layer
        length = len(layer)
        for i in range(0, length, 2):
            if i + 1 < length:
                # Merge two sibling nodes by summing their counts
                merged = {}
                for d in [layer[i], layer[i + 1]]:
                    for k, v in d.items():
                        merged[k] = merged.get(k, 0) + v
                layer_new.append(merged)
        layer = layer_new
        update_sensitivity_sq(layer)

    # The worst-case (max) over all batches is the squared sensitivity
    return max(sensitivity_sq_all)

def compute_epsilon_tree(num_batches, epochs_between_restarts, noise, delta, verbose=True):
    """
    Compute the overall (ε, δ)-DP guarantee for DP-FTRL using tree-aggregated
    sensitivity.
    :param num_batches: number of batches per epoch
    :param epochs_between_restarts: number of epochs between each restart, e.g. [2, 1] means epoch1, epoch2, restart, epoch3
    :param noise: noise multiplier for each step
    :param delta: target DP delta
    :param verbose: whether to print message
    :return: the DP epsilon for DP-FTRL
    """
    if noise < 1e-20:
        return float('inf')

    sensitivity_sq = 0  # total sensitivity^2
    mem = {}  # cache for results

    for i, epochs in enumerate(epochs_between_restarts):
        if epochs == 0:
            continue

        extra_steps = 0

        key = (num_batches, epochs, extra_steps)
        if key not in mem:
            mem[key] = get_total_sensitivity_sq_same_order(num_batches, epochs, extra_steps)
        sensitivity_sq += mem[key]

    effective_sigma = noise / np.sqrt(sensitivity_sq)
    eps = convert_gaussian_renyi_to_dp(effective_sigma, delta, verbose)
    return eps
