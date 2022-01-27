from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data.data import Data
from torch_geometric.nn import SAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from model import modules as mod


class MultipleInstanceModel(nn.Module, ABC):

    def __init__(self, device, n_classes, n_expec_dims):
        super().__init__()
        self.device = device
        self.n_classes = n_classes
        self.n_expec_dims = n_expec_dims

    @abstractmethod
    def forward(self, model_input):
        pass

    @abstractmethod
    def forward_verbose(self, bags):
        pass

    def suggest_train_params(self):
        return {}


class MultipleInstanceNN(MultipleInstanceModel, ABC):

    def forward(self, model_input):
        # Check if input is unbatched bag (n_expec_dims) or batched (n_expec_dims + 1)
        input_shape = model_input.shape
        unbatched_bag = len(input_shape) == self.n_expec_dims

        # Batch if single bag
        bags = model_input.unsqueeze(0) if unbatched_bag else model_input

        # Actually pass the input through the model
        #  We don't care about any interpretability output here
        bag_predictions, _ = self.forward_verbose(bags)

        # Return single pred if unbatched_bag bag else multiple preds
        if unbatched_bag:
            return bag_predictions.squeeze()
        return bag_predictions


class InstanceSpaceNN(MultipleInstanceNN):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def forward_verbose(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        bag_instance_predictions = []
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            bag_prediction, instance_predictions = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_instance_predictions.append(instance_predictions)
        return bag_predictions, bag_instance_predictions


class EmbeddedSpaceNN(MultipleInstanceNN):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def forward_verbose(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            bag_prediction, _ = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
        return bag_predictions, None


class AttentionNN(MultipleInstanceNN):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def forward_verbose(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        bag_attns = []
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            bag_prediction, attn = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_attns.append(attn)
        return bag_predictions, bag_attns


class ClusterGNN(MultipleInstanceModel):

    def __init__(self, device, n_classes, n_expec_dims, encoder, d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout):
        super().__init__(device, n_classes, n_expec_dims)
        self.n_clusters = 1
        self.encoder = encoder
        self.gnn_stack = mod.GNNConvStack(d_enc, ds_gnn_hid, d_gnn, dropout, SAGEConv, raw_last=False)
        # TODO this should be called GNN cluster but the trained models are stuck with gnn_pool
        self.gnn_pool = SAGEConv(d_enc, self.n_clusters)
        self.classifier = mod.FullyConnectedStack(d_gnn, ds_fc_hid, n_classes, dropout, raw_last=True)

    def forward(self, model_input):
        # Unbatched input could be Data type, or raw tensor without Data structure
        unbatched_bag = type(model_input) is Data or \
                        (type(model_input) is torch.Tensor and len(model_input.shape) == self.n_expec_dims)

        # Batch if single bag
        bags = [model_input] if unbatched_bag else model_input

        # Actually pass the input through the model
        #  We don't care about any interpretability output here
        bag_predictions, _ = self.forward_verbose(bags)

        # Return single pred if unbatched_bag bag else multiple preds
        if unbatched_bag:
            return bag_predictions.squeeze()
        return bag_predictions

    def forward_verbose(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        bag_cluster_weights = []
        for i, bag in enumerate(bags):
            # If given Tensor and not Data type, add fully connected graph to data
            if type(bag) is not Data:
                bag = self.add_graph(bag)
            bag = bag.to(self.device)
            x, edge_index = bag.x, bag.edge_index
    
            # Embed instances
            instance_embeddings = self.encoder(x)

            # Run instance embeddings through GNN stack for message passing
            updated_instance_embeddings, _ = self.gnn_stack((instance_embeddings, edge_index))

            # Clustering
            cluster_weights = F.softmax(self.gnn_pool(instance_embeddings, edge_index), dim=0)

            # Reduce to single representation of the graph
            graph_embedding, _, _, _ = dense_diff_pool(updated_instance_embeddings, to_dense_adj(edge_index),
                                                       cluster_weights)
            graph_embedding = graph_embedding.squeeze(dim=0)

            # Classify
            bag_prediction = self.classifier(graph_embedding)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_cluster_weights.append(cluster_weights.T)
        return bag_predictions, bag_cluster_weights

    @staticmethod
    def add_graph(bag):
        edge_index = torch.ones((len(bag), len(bag)))
        edge_index, _ = dense_to_sparse(edge_index)
        edge_index = edge_index.long().contiguous()
        graph_bag = Data(x=bag, edge_index=edge_index)
        return graph_bag
