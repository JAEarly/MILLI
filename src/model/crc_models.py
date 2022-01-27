import model.base_models as bm
from data.crc.crc_dataset import CRC_N_EXPECTED_DIMS, CRC_FV_SIZE

from model import modules as mod
from model import aggregator as agg

from torch import nn
from overrides import overrides


class CrcEncoder(nn.Module):

    def __init__(self, ds_enc_hid, d_enc, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0, dropout=dropout)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0, dropout=dropout)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(CRC_FV_SIZE, ds_enc_hid, d_enc, dropout, raw_last=False)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class CrcInstanceSpaceNN(bm.InstanceSpaceNN):

    def __init__(self, device, n_classes, d_enc=64, ds_enc_hid=(64,), ds_agg_hid=(64,),
                 dropout=0.25, agg_func_name='max'):
        encoder = CrcEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, n_classes, dropout, agg_func_name)
        super().__init__(device, n_classes, CRC_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-2,
        }


class CrcEmbeddingSpaceNN(bm.EmbeddedSpaceNN):

    def __init__(self, device, n_classes, d_enc=512, ds_enc_hid=(64,), ds_agg_hid=(128, 64),
                 dropout=0.3, agg_func_name='max'):
        encoder = CrcEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, n_classes, dropout, agg_func_name)
        super().__init__(device, n_classes, CRC_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-3,
        }


class CrcAttentionNN(bm.AttentionNN):

    def __init__(self, device, n_classes, d_enc=256, ds_enc_hid=(64, 64), ds_agg_hid=(), dropout=0.2, d_attn=128):
        encoder = CrcEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.AttentionAggregator(d_enc, ds_agg_hid, d_attn, n_classes, dropout)
        super().__init__(device, n_classes, CRC_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-6,
        }

class CrcGNN(bm.ClusterGNN):

    def __init__(self, device, n_classes, d_enc=128, ds_enc_hid=(64,), d_gnn=128, ds_gnn_hid=(), ds_fc_hid=(128,),
                 dropout=0.35):
        encoder = CrcEncoder(ds_enc_hid, d_enc, dropout)
        super().__init__(device, n_classes, CRC_N_EXPECTED_DIMS, encoder, d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-2,
        }
