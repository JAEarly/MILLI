import numpy as np
import torch

from interpretability.interpretability_util import get_pred, get_clz_proba
from interpretability.instance_attribution.base_instance_attribution import InstanceAttributionMethod


def get_independent_instance_method(method_name):
    if method_name == 'single':
        return SingleInstanceAttribution()
    if method_name == 'one_removed':
        return OneRemovedInstanceAttribution()
    if method_name == 'combined':
        return CombinedInstanceAttribution(method='mean')
    raise NotImplementedError('Invalid instance independent method name: {:s}'.format(method_name))


class SingleInstanceAttribution(InstanceAttributionMethod):

    def get_instance_clz_attributions(self, bag, model, original_pred, clz):
        attributions = np.zeros(len(bag))
        for i, instance in enumerate(bag):
            single_instance_bag = instance.unsqueeze(0)
            attributions[i] = get_clz_proba(single_instance_bag, model, clz)
        return attributions


class OneRemovedInstanceAttribution(InstanceAttributionMethod):

    def get_instance_clz_attributions(self, bag, model, original_pred, clz):
        attributions = np.zeros(len(bag))
        for i, instance in enumerate(bag):
            one_removed_bag = torch.cat([bag[0:i], bag[i + 1:]])
            one_removed_pred = get_pred(one_removed_bag, model)
            attributions[i] = original_pred[clz] - one_removed_pred[clz]
        return attributions


class CombinedInstanceAttribution(InstanceAttributionMethod):

    def __init__(self, method):
        self.single_method = SingleInstanceAttribution()
        self.one_removed_method = OneRemovedInstanceAttribution()
        self.method = method

    def get_instance_clz_attributions(self, bag, model, orig_pred, clz):
        single_attr = self.single_method.get_instance_clz_attributions(bag, model, orig_pred, clz)
        one_removed_attr = self.one_removed_method.get_instance_clz_attributions(bag, model, orig_pred, clz)
        if self.method == 'mean':
            attributions = (single_attr + one_removed_attr) / 2.0
        elif self.method == 'max':
            attributions = np.max([single_attr, one_removed_attr], axis=0)
        elif self.method == 'vote':
            def get_votes(attr):
                # Sort attr idxs from min to max
                n_instances = len(attr)
                sorted_idxs = np.argsort(attr)
                votes = np.zeros_like(attr)
                # Each idx gets one more vote than the previous, from min to max
                for i in range(n_instances):
                    votes[sorted_idxs[i]] = i
                return votes
            single_votes = get_votes(single_attr)
            one_removed_votes = get_votes(one_removed_attr)
            attributions = single_votes + one_removed_votes
        else:
            raise NotImplementedError('Invalid combined method: {:s}'.format(self.method))
        return attributions
