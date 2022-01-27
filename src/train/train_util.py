import random

from torch_geometric.data.data import Data


class GraphDataloader:

    def __init__(self, graph_dataset):
        self.graph_dataset = graph_dataset
        self.n_graphs = len(self.graph_dataset.bags)

    def __iter__(self):
        self.idx = 0
        self.order = list(range(self.n_graphs))
        random.shuffle(self.order)
        return self

    def __next__(self):
        if self.idx >= self.n_graphs:
            raise StopIteration
        o_idx = self.order[self.idx]
        edge_index = self.graph_dataset.edge_indexs[o_idx]
        instances, target = self.graph_dataset[o_idx]
        data = Data(x=instances, edge_index=edge_index)
        self.idx += 1
        return [data], target.unsqueeze(0)

    def __len__(self):
        return self.n_graphs
