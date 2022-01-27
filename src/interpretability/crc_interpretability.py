import numpy as np
import torch

from data.crc.crc_dataset import CRC_N_CLASSES, load_crc
from interpretability.base_interpretability import Model, InterpretabilityStudy, Method, Metric
from interpretability.instance_attribution import lime_instance_attribution as lime
from interpretability.instance_attribution import milli_instance_attribution as milli
from interpretability.instance_attribution import shap_instance_attribution as shap
from interpretability.instance_attribution.base_instance_attribution import InherentAttribution
from model import crc_models
from interpretability import metrics as met


class CrcInterpretabilityStudy(InterpretabilityStudy):

    def __init__(self, device, n_repeats=None):
        super().__init__(device, CRC_N_CLASSES, "out/interpretability_studies/crc", n_repeats=n_repeats)

    @staticmethod
    def get_clz_target_mask(instance_targets, clz):
        mask = []
        for t in instance_targets:
            mask.append(1 if clz in t else (0 if t else None))
        return np.asarray(mask)

    def get_model_grid(self):
        model_grid = [
            Model('EmbeddedSpace-Net', crc_models.CrcEmbeddingSpaceNN,
                  'models/crc/CrcEmbeddingSpaceNN/CrcEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', crc_models.CrcInstanceSpaceNN,
                  'models/crc/CrcInstanceSpaceNN/CrcInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', crc_models.CrcAttentionNN,
                  'models/crc/CrcAttentionNN/CrcAttentionNN_{:d}.pkl'),
            Model('MI-GNN', crc_models.CrcGNN,
                  'models/crc/CrcGNN/CrcGNN_{:d}.pkl'),
        ]
        return model_grid

    def get_attribution_method_grid(self):
        l2_kernel_width = lime.get_l2_kernel_width(264)
        cosine_kernel_width = lime.get_cosine_kernel_width(264)
        attribution_grid = [
            # Method('Inherent', InherentAttribution()),
            # Method('Single', indep.SingleInstanceAttribution()),
            # Method('One Removed', indep.OneRemovedInstanceAttribution()),
            # Method('Combined', indep.CombinedInstanceAttribution('mean')),
            # Method('RandomSHAP', shap.RandomKernelSHAP(n_samples=1000)),
            # Method('GuidedSHAP', shap.GuidedKernelSHAP(n_samples=1000)),
            # Method('RandomLIME-L2', lime.RandomLinearLIME(n_samples=1000, dist_func_name='l2',
            #                                               kernel_width=l2_kernel_width)),
            # Method('GuidedLIME-L2', lime.GuidedLinearLIME(n_samples=1000, dist_func_name='l2',
            #                                               kernel_width=l2_kernel_width)),
            # Method('WeightedLIME-L2', lime.WeightedLinearLIME(n_samples=1000, dist_func_name='l2',
            #                                                   kernel_width=l2_kernel_width,
            #                                                   alpha=0.008, method='single')),
            # Method('RandomLIME-Cosine', lime.RandomLinearLIME(n_samples=1000, dist_func_name='cosine',
            #                                                   kernel_width=cosine_kernel_width)),
            # Method('GuidedLIME-Cosine', lime.GuidedLinearLIME(n_samples=1000, dist_func_name='cosine',
            #                                                   kernel_width=cosine_kernel_width)),
            # Method('WeightedLIME-Cosine', lime.WeightedLinearLIME(n_samples=1000, dist_func_name='cosine',
            #                                                       kernel_width=cosine_kernel_width,
            #                                                       alpha=0.008, method='single')),
            # Method('MILLI', milli.Milli(n_samples=1000, alpha=0.008, beta=-5, method='single')),
            Method('Paired-MILLI', milli.Milli(n_samples=1000, alpha=0.008, beta=-5, method='single',
                                               paired_sampling=True)),
        ]
        return attribution_grid

    def load_model(self, model_clz, path):
        model = model_clz(self.device, CRC_N_CLASSES)
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        model.eval()
        return model

    def load_test_dataset(self, seed):
        _, _, test_dataset = load_crc(random_state=seed, verbose=False)
        return test_dataset

    def get_attribution_metric_grid(self):
        evaluation_grid = [
            Metric('NDCG@N', met.normalized_discounted_cumulative_gain),
        ]
        return evaluation_grid
