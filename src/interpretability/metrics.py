import math

import numpy as np

from interpretability.interpretability_util import get_clz_proba
from matplotlib import pyplot as plt


class MetricError(ValueError):
    pass


def get_top_k_mask(instance_attributions, clz_instance_targets, k=None):
    if k is None:
        k = np.count_nonzero(clz_instance_targets == 1)
    if k == 0:
        raise MetricError('No instances found to support this class')
    sorted_order = np.argsort(instance_attributions)[::-1]
    top_k_idxs = []
    top_k_mask = []
    i = 0
    while len(top_k_mask) < k:
        idx = sorted_order[i]
        t = clz_instance_targets[idx]
        if t is not None:
            top_k_idxs.append(idx)
            top_k_mask.append(t)
        i += 1
    top_k_mask = np.asarray(top_k_mask)
    return top_k_idxs, top_k_mask, k


def topk_acc(instance_attributions, clz_instance_targets, k=None):
    _, top_k_mask, k = get_top_k_mask(instance_attributions, clz_instance_targets, k=k)
    acc = 1 if 1 in top_k_mask else 0
    return acc


def precision_at_k(instance_attributions, clz_instance_targets, k=None):
    _, top_k_mask, k = get_top_k_mask(instance_attributions, clz_instance_targets, k=k)
    precision = (np.count_nonzero(top_k_mask == 1)/k)
    return precision


def normalized_discounted_cumulative_gain(instance_attributions, clz_instance_targets, k=None):
    _, top_k_mask, k = get_top_k_mask(instance_attributions, clz_instance_targets, k=k)
    dcg = 0
    norm = 0
    for i, m in enumerate(top_k_mask):
        dcg += m/math.log2(i + 2)
        norm += 1/math.log2(i + 2)
    ndcg = dcg/norm
    return ndcg


def perturbation_metric(instance_attributions, bag, model, pred, clz, n_random=10, n_perturbations=None, plot=False):
    n_perturbations = n_perturbations if n_perturbations is not None else len(bag)
    if n_perturbations > (len(bag)):
        raise ValueError('Cannot generate {:d} perturbations for bag of size {:d}. Max is {:d}.'
                         .format(n_perturbations, len(bag), len(bag)))

    assert n_perturbations <= len(bag)

    idxs_by_attribution = np.argsort(instance_attributions)[::-1]
    orig_pred = pred[clz].detach().cpu().item()
    if orig_pred == 0:
        raise MetricError('Cannot measure perturbations when original prediction is zero.')

    n_rows = n_random + 1
    idx_grid = np.zeros((n_rows, len(bag)), dtype=int)
    idx_grid[0, :] = idxs_by_attribution
    for i in range(n_random):
        idxs_by_random = list(range(len(bag)))
        np.random.shuffle(idxs_by_random)
        idx_grid[i + 1, :] = idxs_by_random

    perturbation_preds = np.zeros(n_perturbations + 1)
    perturbation_preds[0] = orig_pred
    diff_grid = np.zeros_like(idx_grid, dtype=float)
    for c in range(1, n_perturbations):
        for r in range(n_rows):
            reduced_coalition = np.ones(len(bag))
            idxs_to_remove = idx_grid[r, :c]
            reduced_coalition[idxs_to_remove] = 0
            reduced_bag = bag[reduced_coalition == 1]
            new_pred = get_clz_proba(reduced_bag, model, clz)
            diff_grid[r, c] = orig_pred - new_pred
            if r == 0:
                perturbation_preds[c] = new_pred

    aopc_r = _aopc_r_at_p(diff_grid, n_random, n_perturbations)

    if plot:
        print('AOPC-R:', aopc_r)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
        aopc_r_by_n_perturbation = [_aopc_r_at_p(diff_grid, n_random, p + 1) for p in range(n_perturbations)]
        aopc_r_by_n_perturbation.insert(0, 0)

        axes[0].plot(range(n_perturbations + 1), perturbation_preds)
        axes[0].set_xlim(0, n_perturbations)
        axes[0].set_ylim(0, orig_pred * 1.1)
        axes[0].set_xlabel('Number of removed instances')
        axes[0].set_ylabel('Model Prediction')

        axes[1].plot(range(n_perturbations + 1), aopc_r_by_n_perturbation)
        axes[1].set_xlim(0, n_perturbations)
        axes[1].set_xlabel('Number of removed instances')
        axes[1].set_ylabel('AOPC-R')
        plt.show()

    return aopc_r


def _aopc_r_at_p(diff_grid, n_random, p):
    aopc_m = np.mean(diff_grid[0, :p])
    aopc_diffs = np.zeros(n_random)
    for i in range(n_random):
        aopc_diffs[i] = aopc_m - diff_grid[1 + i, :p].mean()
    aopc_r = aopc_diffs.mean()
    return aopc_r
