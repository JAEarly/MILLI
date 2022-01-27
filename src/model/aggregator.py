import torch
from torch import nn

from model import modules as mod
from abc import ABC


class Aggregator(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _parse_agg_method(agg_func_name):
        if agg_func_name == 'mean':
            return lambda x: torch.mean(x, dim=0)
        if agg_func_name == 'max':
            return lambda x: torch.max(x, dim=0)[0]
        raise ValueError('Invalid aggregation function name for Instance Aggregator: {:s}'.format(agg_func_name))


class InstanceAggregator(Aggregator):

    def __init__(self, d_in, ds_hid, n_classes, dropout, agg_func_name):
        super().__init__()
        self.instance_classifier = mod.FullyConnectedStack(d_in, ds_hid, n_classes, dropout, raw_last=True)
        self.aggregation_func = self._parse_agg_method(agg_func_name)

    def forward(self, instance_embeddings):
        instance_predictions = self.instance_classifier(instance_embeddings)
        bag_prediction = self.aggregation_func(instance_predictions)
        return bag_prediction, instance_predictions


class EmbeddingAggregator(Aggregator):

    def __init__(self, d_in, ds_hid, n_classes, dropout, agg_func_name):
        super().__init__()
        self.aggregation_func = self._parse_agg_method(agg_func_name)
        self.embedding_classifier = mod.FullyConnectedStack(d_in, ds_hid, n_classes, dropout, raw_last=True)

    def forward(self, instance_embeddings):
        bag_embedding = self.aggregation_func(instance_embeddings)
        bag_prediction = self.embedding_classifier(bag_embedding)
        return bag_prediction, None


class AttentionAggregator(Aggregator):

    def __init__(self, d_in, ds_hid, d_attn, n_classes, dropout):
        super().__init__()
        self.attention_aggregator = mod.AttentionBlock(d_in, d_attn, dropout)
        self.embedding_classifier = mod.FullyConnectedStack(d_in, ds_hid, n_classes, dropout, raw_last=True)

    def forward(self, instance_embeddings):
        bag_embedding, attn = self.attention_aggregator(instance_embeddings)
        bag_prediction = self.embedding_classifier(bag_embedding)
        return bag_prediction, attn
