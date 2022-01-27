from abc import ABC

from model import mnist_models
from train.mnist_training import MnistNetTrainer, MnistGNNTrainer
from tuning.tune_base import Tuner


def get_tuner(mnist_model_clz):
    if mnist_model_clz == mnist_models.MnistInstanceSpaceNN:
        return MnistInstanceSpaceNNTuner
    elif mnist_model_clz == mnist_models.MnistEmbeddingSpaceNN:
        return MnistEmbeddingSpaceNNTuner
    elif mnist_model_clz == mnist_models.MnistAttentionNN:
        return MnistAttentionNNTuner
    elif mnist_model_clz == mnist_models.MnistGNN:
        return MnistGNNTuner
    raise NotImplementedError('No tuner implement for class {:}'.format(mnist_model_clz))


class MnistTuner(Tuner, ABC):

    def __init__(self, device, model_clz):
        super().__init__(device)
        self.model_clz = model_clz

    def create_trainer(self, train_params, model_params):
        return MnistNetTrainer(self.device, train_params, self.model_clz, model_params)

    def generate_train_params(self, trial):
        lr = trial.suggest_categorical('lr', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        wd = trial.suggest_categorical('wd', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        train_params = {
            'lr': lr,
            'weight_decay': wd,
        }
        return train_params


class MnistInstanceSpaceNNTuner(MnistTuner):

    def __init__(self, device):
        super().__init__(device, mnist_models.MnistInstanceSpaceNN)

    def generate_model_params(self, trial):
        d_enc = trial.suggest_categorical('d_enc', [64, 128, 256, 512])
        ds_enc_hid = self.suggest_layers(trial, "enc_hid", "ds_enc_hid", 0, 2, [64, 128, 256])
        ds_agg_hid = self.suggest_layers(trial, "agg_hid", "ds_agg_hid", 0, 2, [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.05)
        agg_func_name = trial.suggest_categorical('agg_func', ['mean', 'max'])
        model_params = {
            'd_enc': d_enc,
            'ds_enc_hid': ds_enc_hid,
            'ds_agg_hid': ds_agg_hid,
            'dropout': dropout,
            'agg_func_name': agg_func_name,
        }
        return model_params


class MnistEmbeddingSpaceNNTuner(MnistTuner):

    def __init__(self, device):
        super().__init__(device, mnist_models.MnistEmbeddingSpaceNN)

    def generate_model_params(self, trial):
        d_enc = trial.suggest_categorical('d_enc', [64, 128, 256, 512])
        ds_enc_hid = self.suggest_layers(trial, "enc_hid", "ds_enc_hid", 0, 2, [64, 128, 256])
        ds_agg_hid = self.suggest_layers(trial, "agg_hid", "ds_agg_hid", 0, 2, [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.05)
        agg_func_name = trial.suggest_categorical('agg_func', ['mean', 'max'])
        model_params = {
            'd_enc': d_enc,
            'ds_enc_hid': ds_enc_hid,
            'ds_agg_hid': ds_agg_hid,
            'dropout': dropout,
            'agg_func_name': agg_func_name,
        }
        return model_params


class MnistAttentionNNTuner(MnistTuner):

    def __init__(self, device):
        super().__init__(device, mnist_models.MnistAttentionNN)

    def generate_model_params(self, trial):
        d_enc = trial.suggest_categorical('d_enc', [64, 128, 256, 512])
        ds_enc_hid = self.suggest_layers(trial, "enc_hid", "ds_enc_hid", 0, 2, [64, 128, 256])
        ds_agg_hid = self.suggest_layers(trial, "agg_hid", "ds_agg_hid", 0, 2, [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.05)
        d_attn = trial.suggest_categorical('d_attn', [64, 128, 256])
        model_params = {
            'd_enc': d_enc,
            'ds_enc_hid': ds_enc_hid,
            'ds_agg_hid': ds_agg_hid,
            'dropout': dropout,
            'd_attn': d_attn,
        }
        return model_params


class MnistGNNTuner(MnistTuner):

    def __init__(self, device):
        super().__init__(device, mnist_models.MnistGNN)

    def create_trainer(self, train_params, model_params):
        return MnistGNNTrainer(self.device, train_params, self.model_clz, model_params)

    def generate_model_params(self, trial):
        d_enc = trial.suggest_categorical('d_enc', [64, 128, 256, 512])
        ds_enc_hid = self.suggest_layers(trial, "enc_hid", "ds_enc_hid", 0, 2, [64, 128, 256])
        d_gnn = trial.suggest_categorical('d_gnn', [64, 128, 256, 512])
        ds_gnn_hid = self.suggest_layers(trial, "gnn_hid", "ds_gnn_hid", 0, 2, [64, 128, 256])
        ds_fc_hid = self.suggest_layers(trial, "fc_hid", "ds_fc_hid", 0, 2, [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.05)
        model_params = {
            'd_enc': d_enc,
            'ds_enc_hid': ds_enc_hid,
            'd_gnn': d_gnn,
            'ds_gnn_hid': ds_gnn_hid,
            'ds_fc_hid': ds_fc_hid,
            'dropout': dropout,
        }
        return model_params
