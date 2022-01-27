from abc import ABC

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.sival.sival_dataset import create_datasets, SIVAL_N_CLASSES
from model.sival_models import SivalGNN
from train.train_base import Trainer
from train.train_util import GraphDataloader


class SivalTrainer(Trainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, SIVAL_N_CLASSES, "models/sival", train_params)
        self.model_clz = model_clz
        self.model_params = model_params

    def create_model(self):
        if self.model_params is not None:
            return self.model_clz(self.device, **self.model_params)
        return self.model_clz(self.device)

    def get_model_name(self):
        return self.model_clz.__name__

    def create_optimizer(self, model):
        lr = self.get_train_param('lr')
        weight_decay = self.get_train_param('weight_decay')
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class SivalNetTrainer(SivalTrainer):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, train_params, model_clz, model_params=model_params)
        assert model_clz != SivalGNN

    def load_datasets(self, seed=None):
        train_dataset, val_dataset, test_dataset = create_datasets(random_state=seed)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)
        return train_dataloader, val_dataloader, test_dataloader


class SivalGNNTrainer(SivalTrainer):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, train_params, model_clz, model_params=model_params)
        assert model_clz == SivalGNN

    def load_datasets(self, seed=None):
        train_dataset, val_dataset, test_dataset = create_datasets(random_state=seed)
        train_dataset.add_edges()
        val_dataset.add_edges()
        test_dataset.add_edges()
        train_dataloader = GraphDataloader(train_dataset)
        val_dataloader = GraphDataloader(val_dataset)
        test_dataloader = GraphDataloader(test_dataset)
        return train_dataloader, val_dataloader, test_dataloader
