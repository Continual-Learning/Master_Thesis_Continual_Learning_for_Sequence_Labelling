import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

# Required: download the cca_core.py script from https://github.com/google/svcca
# copy it in this directory or elsewhere in your PYTHONPATH (it is gitignored in this repo)
from cca_core import get_cca_similarity
from display_names import dataset_display_name, model_display_name, metric_display_name

logger = logging.getLogger('shared_encoder:correlation_utils')


def sequence_js_divergence(p, q):
    """Computes the avereage JS divergence between attention distributions corresponding to words in two sequences
    :param p    attention distributions for a sequence output by a self-attention head. Shape: num_tokens^3.
    :param q    attention distributions for a sequence output by a self-attention head. Shape: num_tokens^3.

    :returns    Average JS divergence over the attention distributions over each token in the sequence
    """
    # Transpose ensures it works along axis 0 (using axis=1 may give errors on the cluster)
    p = np.transpose(p)
    q = np.transpose(q)
    m = (p + q) / 2
    return np.mean(0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2)))


def cca(hidden_rep, hidden_rep_cmp, eps=1e-10):
    """Compute correlation between layers as average correlation of neurons computed using CCA"""
    cca_similarity = get_cca_similarity(hidden_rep, hidden_rep_cmp, epsilon=eps)
    return np.mean(cca_similarity["cca_coef1"])


def svd(a, retain_variance=0.99):
    """Computes singular value decomposition of matrix a,
    keeping only the dimensions that explain at least 99% of the variance in A

    :param a                2D array of dimensions num_neurons x num_samples
    :param retain_variance  Fraction of variance the retained dimensions need to explain

    :returns    A projected onto the dimensions that explain at least a % of the variance equal to retain_variance
    """
    a_cent = a - np.mean(a, axis=1, keepdims=True)
    u, s, _ = np.linalg.svd(a_cent, full_matrices=False)
    s_sq = s ** 2
    tot_variance = sum(s_sq)
    curr_variance = 0.0
    i = 0
    while curr_variance / tot_variance < retain_variance:
        curr_variance += s_sq[i]
        i += 1
    logger.info("Reduced dimensions from %i to %i", len(s_sq), i)
    # Project onto main SV directions
    return np.dot(u.T[:i], a_cent)


def svcca(hidden_rep, hidden_rep_cmp):
    hidden_rep_p = svd(np.transpose(hidden_rep))
    hidden_rep_cmp_p = svd(np.transpose(hidden_rep_cmp))

    return cca(hidden_rep_p, hidden_rep_cmp_p)

def visualize_attention_map(correlations, plot_measure, model_name, dataset, out_filename, vmax=1.0, cmap='Spectral_r'):
    """Save a heatmap containing the correlation values of each layer and head"""
    plt.rcParams.update({'font.size': 20})
    _, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlations, cmap=cmap, vmin=0, vmax=vmax)

    ax.set_title(f"{model_display_name(model_name)}\n({dataset_display_name(dataset)})")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(metric_display_name(plot_measure))

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(correlations.shape[0]))
    ax.set_yticks(np.arange(correlations.shape[1]))
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Transformer layer")

    plt.savefig(out_filename, format='jpg', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
