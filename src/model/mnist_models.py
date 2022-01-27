import model.base_models as bm
from data.mnist_bags import MNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, MNIST_FV_SIZE

from model import modules as mod
from model import aggregator as agg

from torch import nn
from overrides import overrides


class MnistEncoder(nn.Module):

    def __init__(self, ds_enc_hid, d_enc, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=1, c_out=20, kernel_size=5, stride=1, padding=0, dropout=dropout)
        conv2 = mod.ConvBlock(c_in=20, c_out=50, kernel_size=5, stride=1, padding=0, dropout=dropout)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(MNIST_FV_SIZE, ds_enc_hid, d_enc, dropout, raw_last=False)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class MnistInstanceSpaceNN(bm.InstanceSpaceNN):

    def __init__(self, device, d_enc=512, ds_enc_hid=(), ds_agg_hid=(128, 64,), dropout=0.3, agg_func_name='mean'):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, MNIST_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, MNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        }


class MnistEmbeddingSpaceNN(bm.EmbeddedSpaceNN):

    def __init__(self, device, d_enc=512, ds_enc_hid=(128,), ds_agg_hid=(), dropout=0.3, agg_func_name='mean'):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, MNIST_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, MNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-3,
        }


class MnistAttentionNN(bm.AttentionNN):

    def __init__(self, device, d_enc=256, ds_enc_hid=(64,), ds_agg_hid=(64,), dropout=0.15, d_attn=64):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.AttentionAggregator(d_enc, ds_agg_hid, d_attn, MNIST_N_CLASSES, dropout)
        super().__init__(device, MNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        }


class MnistGNN(bm.ClusterGNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(64,), d_gnn=128, ds_gnn_hid=(128, 128), ds_fc_hid=(64,), dropout=0.3):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        super().__init__(device, MNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, encoder,
                         d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-5,
            'weight_decay': 1e-5,
        }
