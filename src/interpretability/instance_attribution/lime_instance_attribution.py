import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import euclidean, cosine
from tqdm import tqdm

from interpretability import local_surrogate
from interpretability.instance_attribution.base_instance_attribution import InstanceAttributionMethod
from interpretability.instance_attribution.independent_instance_attribution import get_independent_instance_method
from interpretability.interpretability_util import get_clz_proba
from itertools import combinations


def l2_dist(z):
    x = np.ones_like(z)
    return euclidean(z, x)


def get_l2_kernel_width(mean_bag_size):
    half_coalition = np.ones(mean_bag_size)
    half_coalition[:mean_bag_size // 2] = 0
    l2_kernel_width = np.sqrt(- l2_dist(half_coalition) ** 2 / np.log(0.25))
    return l2_kernel_width


def cosine_dist(z):
    x = np.ones_like(z)
    return cosine(z, x)


def get_cosine_kernel_width(mean_bag_size):
    half_coalition = np.ones(mean_bag_size)
    half_coalition[:mean_bag_size // 2] = 0
    cosine_kernel_width = np.sqrt(- cosine_dist(half_coalition) ** 2 / np.log(0.25))
    return cosine_kernel_width


def lime_kernel(z, dist_func, kernel_width):
    d = dist_func(z)
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def get_coalition_weights(coalitions, dist_func, kernel_width):
    weights = []
    for coalition in coalitions:
        weight = lime_kernel(coalition, dist_func, kernel_width)
        weights.append(weight)
    weights = np.asarray(weights)
    return weights


def lime_guided_gen(s):
    """
    Generate subsets from set s according to LIME weighting.
    Ignore empty set and complete set s, i.e., all subsets of length l, where 1 < l < |s| - 1.
    Highest weighted subsets are the biggest, and lower weighted subsets are the smallest.
        This assumes the LIME kernel monotonically increases with subset size,
        which it does for L2 and cosines distance measures.
    Sample from a backward generator (sets getting smaller).
    """
    while True:
        for c in [combinations(s, r) for r in reversed(range(1, len(s)))]:
            for x in c:
                yield list(x)


class LinearLIME(InstanceAttributionMethod, ABC):

    def __init__(self, n_samples, dist_func_name='l2', kernel_width=1, allow_repeats=False, verbose=False):
        self.n_samples = n_samples
        self.verbose = verbose
        self.dist_func = self._get_dist_func(dist_func_name)
        self.kernel_width = kernel_width
        self.allow_repeats = allow_repeats

    def get_instance_clz_attributions(self, bag, model, original_pred, clz):
        coalitions = self._generate_coalitions(bag, model, original_pred)
        sampled_probas = self.get_coalition_probas(bag, model, clz, coalitions)
        sampled_weights = get_coalition_weights(coalitions, self.dist_func, self.kernel_width)
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
        sampled_weights = get_coalition_weights(coalitions, self.dist_func, self.kernel_width)
        convergence_values = np.zeros((self.n_samples + 1, len(bag)))
        for n in range(1, self.n_samples + 1):
            step_shapley_values = local_surrogate.fit_weighted_linear_model(coalitions[:n, :], sampled_probas[:n],
                                                                            sampled_weights[:n])
            convergence_values[n, :] = step_shapley_values
        return convergence_values

    @abstractmethod
    def _generate_coalitions(self, bag, model, orig_pred):
        pass

    @staticmethod
    def _get_dist_func(dist_func_name):
        if dist_func_name == 'l2':
            return l2_dist
        if dist_func_name == 'cosine':
            return cosine_dist
        raise ValueError('Invalid distance function for LinearLIME: {:s}'.format(dist_func_name))


class RandomLinearLIME(LinearLIME):

    def _generate_coalitions(self, bag, model, orig_pred):
        return local_surrogate.generate_random_coalitions(bag, self.n_samples, allow_repeats=self.allow_repeats)


class GuidedLinearLIME(LinearLIME):

    def _generate_coalitions(self, bag, model, orig_pred):
        coalition_size = len(bag)
        coalitions = []
        gen = lime_guided_gen(list(range(coalition_size)))
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
                gen = lime_guided_gen(list(range(coalition_size)))
                selected_idxs = next(gen)
            coalition = np.zeros(coalition_size)
            coalition[selected_idxs] = 1
            coalitions.append(coalition)
        coalitions = np.asarray(coalitions)
        return coalitions


class WeightedLinearLIME(LinearLIME):

    def __init__(self, n_samples, method='single', alpha=0.1, dist_func_name='l2', kernel_width=1):
        super().__init__(n_samples, dist_func_name=dist_func_name, kernel_width=kernel_width)
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
