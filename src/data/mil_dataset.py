from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


class MilDataset(Dataset):

    def __init__(self, bags, targets, instance_targets):
        super(Dataset, self).__init__()
        self.bags = bags
        self.targets = torch.as_tensor(targets).float()
        self.instance_targets = instance_targets

    def add_edges(self, eta=None):
        self.edge_indexs = []
        for bag in self.bags:
            # If eta not set, create fully connected graph
            if eta is None:
                edge_index = torch.ones((len(bag), len(bag)))
                edge_index, _ = dense_to_sparse(edge_index)
                edge_index = edge_index.long().contiguous()
            else:
                dist = torch.cdist(bag, bag)
                b_dist = torch.zeros_like(dist)
                b_dist[dist < eta] = 1
                edge_index = (dist < eta).nonzero().t()
            edge_index = edge_index.long().contiguous()
            self.edge_indexs.append(edge_index)

    def summarise(self):
        clz_dist = Counter(np.asarray(self.targets))
        print('- MIL Dataset Summary -')
        print(' Class Distribution')
        for clz in sorted(clz_dist.keys()):
            print('  {:d}: {:d} ({:.2f}%)'.format(int(clz), clz_dist[clz], clz_dist[clz]/len(self)*100))

        bag_sizes = [len(b) for b in self.bags]
        print(' Bag Sizes')
        print('  Min: {:d}'.format(min(bag_sizes)))
        print('  Avg: {:.1f}'.format(np.mean(bag_sizes)))
        print('  Max: {:d}'.format(max(bag_sizes)))

    def get_bag_verbose(self, index):
        bag = self.bags[index]
        target = self.targets[index]
        instance_targets = self.instance_targets[index] if self.instance_targets is not None else None
        return bag, target, instance_targets

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        target = self.targets[index]
        return bag, target
