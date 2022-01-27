import numpy as np
from sklearn.linear_model import Ridge


def random_gen(coalition_size):
    while True:
        coalition = np.random.randint(2, size=coalition_size)
        # Can't have all ones or all zeros
        if 1 in coalition and 0 in coalition:
            selected_idxs = [x for i, x in enumerate(list(range(coalition_size))) if coalition[i] == 1]
            yield selected_idxs


def generate_random_coalitions(bag, n_samples, allow_repeats=False):
    coalition_size = len(bag)
    gen = random_gen(coalition_size)
    coalitions = []
    while len(coalitions) < n_samples:
        selected_idxs = next(gen)
        coalition = np.zeros(coalition_size)
        coalition[selected_idxs] = 1
        if allow_repeats or not any(all(c == coalition) for c in coalitions):
            coalitions.append(list(coalition))
    coalitions = np.asarray(coalitions)
    return coalitions


def weighted_gen(coalition_size, weights):
    while True:
        selected_idxs = []
        for idx in range(coalition_size):
            w = weights[idx]
            p = np.random.random(1)[0]
            if p < w:
                selected_idxs.append(idx)
        if selected_idxs and len(selected_idxs) < coalition_size:
            yield selected_idxs


def generate_weighted_coalitions(bag, weights, n_samples, allow_repeats=False, paired_sampling=False):
    coalition_size = len(bag)
    gen = weighted_gen(coalition_size, weights)
    coalitions = []
    while len(coalitions) < n_samples:
        selected_idxs = next(gen)
        coalition = np.zeros(coalition_size)
        coalition[selected_idxs] = 1
        if allow_repeats or not any(all(c == coalition) for c in coalitions):
            coalitions.append(list(coalition))
        if paired_sampling:
            paired_coalition = np.ones(coalition_size)
            paired_coalition[selected_idxs] = 0
            if allow_repeats or not any(all(c == paired_coalition) for c in coalitions):
                coalitions.append(list(paired_coalition))
    coalitions = np.asarray(coalitions)

    # TODO output histogram of selected indices
    # print(np.sum(coalitions, axis=0))

    return coalitions


def get_coalition_bags(original_bag, coalitions):
    bags = []
    for coalition in coalitions:
        bag = original_bag[coalition == 1]
        bags.append(bag)
    return bags


def fit_weighted_linear_model(coalitions, probas, weights):
    # reg = LinearRegression().fit(coalitions, probas, sample_weight=weights)
    reg = Ridge(alpha=1e-10).fit(coalitions, probas, sample_weight=weights)
    shapley_values = reg.coef_
    return shapley_values
