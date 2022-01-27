import warnings
from abc import ABC, abstractmethod
from itertools import combinations
from math import factorial

import numpy as np
from tqdm import tqdm

from interpretability import local_surrogate
from interpretability.instance_attribution.base_instance_attribution import InstanceAttributionMethod
from interpretability.instance_attribution.independent_instance_attribution import get_independent_instance_method
from interpretability.interpretability_util import get_clz_proba


def comb(n, r):
    f = factorial
    c = f(n) // f(r) // f(n-r)
    return c


def powerset(s):
    x = len(s)
    for i in range(1 << x):
        mask = [j for j in range(x) if (i & (1 << j))]
        yield s[mask]


def shap_kernel(m, z):
    # Change types to fix overflow issues
    m = int(m)
    z = int(z)
    c = int(comb(m, z))
    return (m - 1) / (c * z * (m - z))


def get_coalition_weights(max_size, coalitions):
    weights = []
    for coalition in coalitions:
        weight = shap_kernel(max_size, (coalition == 1).sum())
        weights.append(weight)
    weights = np.asarray(weights)
    return weights


def shap_guided_gen(s):
    """
    Generate subsets from set s according to SHAP weighting
    Ignore empty set and complete set s, i.e., all subsets of length l, where 1 < l < |s| - 1
    Highest weighted subsets are at the extremes, and lower weighted subsets are in the middle
       i.e., highest weighted are single instance sets, and n-1 sets
    Sample alternatively from forward generator (sets getting bigger) and backward generator (sets getting smaller)
    Stop when they meet in the middle
    """
    # Calculate midpoint of set (where the two generators will meet in the middle)
    mid = len(s) // 2

    # Forward generator - sets of length 1 to sets of length mid-1
    def forward_gen():
        for c in [combinations(s, r) for r in range(1, mid)]:
            for x in c:
                yield list(x)

    # Backward generator - sets of length n-1 to sets of length mid
    def backward_gen():
        for c in [combinations(s, r) for r in reversed(range(mid, len(s)))]:
            for x in c:
                yield list(x)

    # Create both generators
    f = forward_gen()
    b = backward_gen()

    # Track if both internal generators have stopped (to raise StopIterator for the wrapper generator)
    f_stopped = False
    b_stopped = False

    # Yield alternatively from forward and backward generators until both exhausted
    while not f_stopped or not b_stopped:
        if not f_stopped:
            try:
                yield next(f)
            except StopIteration:
                f_stopped = True
        if not b_stopped:
            try:
                yield next(b)
            except StopIteration:
                b_stopped = True


class KernelSHAP(InstanceAttributionMethod, ABC):

    def __init__(self, n_samples, allow_repeats=False, verbose=False):
        self.n_samples = n_samples
        self.verbose = verbose
        self.allow_repeats = allow_repeats

    def get_instance_clz_attributions(self, bag, model, original_pred, clz):
        coalitions = self._generate_coalitions(bag, model, original_pred)
        sampled_probas = self.get_coalition_probas(bag, model, clz, coalitions)
        sampled_weights = get_coalition_weights(len(bag), coalitions)
        shapley_values = local_surrogate.fit_weighted_linear_model(coalitions, sampled_probas, sampled_weights)
        return shapley_values

    def get_coalition_probas(self, original_bag, model, clz, coalitions):
        probas = []
        for coalition in tqdm(coalitions, desc="Getting probas", disable=not self.verbose):
            bag = original_bag[coalition == 1]
            proba = get_clz_proba(bag, model, clz)
            probas.append(proba)
        probas = np.asarray(probas)
        return probas

    def get_convergence_values(self, bag, model, original_pred, clz):
        coalitions = self._generate_coalitions(bag, model, original_pred)
        sampled_probas = self.get_coalition_probas(bag, model, clz, coalitions)
        sampled_weights = get_coalition_weights(len(bag), coalitions)
        convergence_values = np.zeros((self.n_samples + 1, len(bag)))
        for n in range(1, self.n_samples + 1):
            step_shapley_values = local_surrogate.fit_weighted_linear_model(coalitions[:n, :], sampled_probas[:n],
                                                                            sampled_weights[:n])
            convergence_values[n, :] = step_shapley_values
        return convergence_values

    @abstractmethod
    def _generate_coalitions(self, bag, model, orig_pred):
        pass


class RandomKernelSHAP(KernelSHAP):

    def _generate_coalitions(self, bag, model, orig_pred):
        return local_surrogate.generate_random_coalitions(bag, self.n_samples, allow_repeats=self.allow_repeats)


class GuidedKernelSHAP(KernelSHAP):

    def _generate_coalitions(self, bag, model, orig_pred):
        coalition_size = len(bag)
        coalitions = []
        gen = shap_guided_gen(list(range(coalition_size)))
        limit = 2 ** coalition_size - 2
        if limit < self.n_samples and not self.allow_repeats:
            warnings.warn("Number of requested samples ({:d}) is greater than the total number of possible coalitions,"
                          " and repeat sampling is not allowed. The upper limit of {:d} will be used instead."
                          .format(self.n_samples, limit))
        else:
            limit = self.n_samples
        while len(coalitions) < limit:
            try:
                selected_idxs = next(gen)
            except StopIteration:
                # If allowing repeats, restart the generator
                gen = shap_guided_gen(list(range(coalition_size)))
                selected_idxs = next(gen)
            coalition = np.zeros(coalition_size)
            coalition[selected_idxs] = 1
            coalitions.append(coalition)
        coalitions = np.asarray(coalitions)
        return coalitions


class WeightedKernelSHAP(KernelSHAP):

    def __init__(self, n_samples, method='single', alpha=0.1):
        super().__init__(n_samples)
        self.initial_method = get_independent_instance_method(method)
        self.alpha = alpha

    def _generate_coalitions(self, bag, model, orig_pred):
        initial_weights = 1 - self.initial_method.get_instance_clz_attributions(bag, model, orig_pred, 0)
        initial_order = np.argsort(initial_weights)[::-1]
        coalition_size = len(bag)
        weights = np.zeros_like(initial_weights)
        for i in range(coalition_size):
            w = (1 - 2 * self.alpha) / (coalition_size - 1) * i + self.alpha
            idx = initial_order[i]
            weights[idx] = w
        return local_surrogate.generate_weighted_coalitions(bag, weights, self.n_samples)
