import math

import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from interpretability import local_surrogate
from interpretability.instance_attribution.base_instance_attribution import InstanceAttributionMethod
from interpretability.instance_attribution.independent_instance_attribution import get_independent_instance_method
from interpretability.interpretability_util import get_clz_proba


def milli_func(x, n_instances, alpha, beta):
    k = n_instances - 1
    # Ensure that positive beta always produces larger coalitions
    if alpha > 0.5:
        beta = -beta
    if beta >= 0:
        return (2 * alpha - 1) * (1 - x/k) * math.exp(-beta * x) + 1 - alpha
    else:
        return (1 - 2 * alpha) * (1 + (x - k)/k) * math.exp(abs(beta) * (x - k)) + alpha


def milli_integral(n_instances, alpha, beta):
    k = n_instances - 1
    # Ensure that positive beta always produces larger coalitions
    if alpha > 0.5:
        beta = -beta
    if beta > 0:
        return ((2 * alpha - 1) / (k * beta ** 2)) * (math.exp(-beta * k) + beta * k - 1) + k * (1 - alpha)
    elif beta < 0:
        return ((1 - 2 * alpha) / (abs(beta) ** 2 * k)) * (math.exp(-abs(beta) * k) + abs(beta) * k - 1) + k * alpha
    else:
        return 0.5 * k


def find_beta(n_instances, alpha, target_expected_coalition_size):
    print('Finding beta for alpha = {:.4f} and target coalition size = {:.4f}'
          .format(alpha, float(target_expected_coalition_size)))

    def f(b):
        return abs(milli_integral(n_instances, alpha, b) - target_expected_coalition_size)

    res = minimize_scalar(f)
    best_beta = res.x

    print('Best beta found: {:.4f}'.format(best_beta))

    return best_beta


def calculate_milli_weights(initial_ordering, alpha, beta):
    n_instances = len(initial_ordering)
    weights = np.zeros_like(initial_ordering, dtype=float)
    for i in range(n_instances):
        w = milli_func(i, n_instances, alpha, beta)
        idx = initial_ordering[i]
        weights[idx] = w
    return weights


def milli_kernel(z, milli_weights):
    s = (z * milli_weights).sum() / z.sum()
    return s


def get_coalition_weights(coalitions, milli_weights):
    weights = []
    for coalition in coalitions:
        weight = milli_kernel(coalition, milli_weights)
        weights.append(weight)
    weights = np.asarray(weights)
    return weights


class Milli(InstanceAttributionMethod):

    def __init__(self, n_samples, method='single', alpha=0.1, beta=1.0, verbose=False, allow_repeats=False,
                 paired_sampling=False):
        self.initial_method = get_independent_instance_method(method)
        self.n_samples = n_samples
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.allow_repeats = allow_repeats
        self.paired_sampling = paired_sampling

    def get_instance_clz_attributions(self, bag, model, original_pred, clz):
        milli_weights = self.get_milli_weights(bag, model, original_pred)
        coalitions = self._generate_coalitions(bag, milli_weights)
        sampled_probas = self.get_coalition_probas(bag, model, clz, coalitions)
        sampled_weights = get_coalition_weights(coalitions, milli_weights)
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

    def get_milli_weights(self, bag, model, orig_pred):
        initial_weights = 1 - self.initial_method.get_instance_clz_attributions(bag, model, orig_pred, 0)
        initial_order = np.argsort(initial_weights)[::-1]
        weights = calculate_milli_weights(initial_order, self.alpha, self.beta)
        return weights

    def _generate_coalitions(self, bag, milli_weights):
        coalitions = local_surrogate.generate_weighted_coalitions(bag, milli_weights, self.n_samples,
                                                                  allow_repeats=self.allow_repeats,
                                                                  paired_sampling=self.paired_sampling)
        return coalitions


if __name__ == "__main__":
    find_beta(264, 0.008, 2)
