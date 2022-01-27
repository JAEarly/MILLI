import numpy as np
import torch

from data.sival.sival_dataset import create_datasets, SIVAL_N_CLASSES
from interpretability.base_interpretability import Model, InterpretabilityStudy, Method
from interpretability.instance_attribution import independent_instance_attribution as indep
from interpretability.instance_attribution import lime_instance_attribution as lime
from interpretability.instance_attribution import milli_instance_attribution as milli
from interpretability.instance_attribution import shap_instance_attribution as shap
from interpretability.instance_attribution.base_instance_attribution import InherentAttribution
from model import sival_models


class SivalInterpretabilityStudy(InterpretabilityStudy):

    def __init__(self, device, n_repeats=None):
        super().__init__(device, SIVAL_N_CLASSES,  "out/interpretability_studies/sival", n_repeats=n_repeats,
                         all_clz_attribution=False)

    @staticmethod
    def get_clz_target_mask(instance_targets, clz):
        mask_positive_idxs = (instance_targets == clz).nonzero(as_tuple=True)[0]
        mask = np.zeros(len(instance_targets))
        mask[mask_positive_idxs] = 1
        return mask

    def get_model_grid(self):
        model_grid = [
            Model('EmbeddedSpace-Net', sival_models.SivalEmbeddingSpaceNN,
                  'models/sival/SivalEmbeddingSpaceNN/SivalEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', sival_models.SivalInstanceSpaceNN,
                  'models/sival/SivalInstanceSpaceNN/SivalInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', sival_models.SivalAttentionNN,
                  'models/sival/SivalAttentionNN/SivalAttentionNN_{:d}.pkl'),
            Model('MI-GNN', sival_models.SivalGNN,
                  'models/sival/SivalGNN/SivalGNN_{:d}.pkl'),
        ]
        return model_grid

    def get_attribution_method_grid(self):
        l2_kernel_width = lime.get_l2_kernel_width(30)
        attribution_grid = [
            # Method('Inherent', InherentAttribution()),
            # Method('Single', indep.SingleInstanceAttribution()),
            # Method('One Removed', indep.OneRemovedInstanceAttribution()),
            # Method('Combined', indep.CombinedInstanceAttribution('mean')),
            # Method('RandomSHAP', shap.RandomKernelSHAP(n_samples=200)),
            # Method('GuidedSHAP', shap.GuidedKernelSHAP(n_samples=200)),
            # Method('WeightedSHAP', shap.WeightedKernelSHAP(n_samples=200, alpha=0.05, method='single')),
            # Method('RandomLIME-L2', lime.RandomLinearLIME(n_samples=200, dist_func_name='l2',
            #                                               kernel_width=l2_kernel_width)),
            # Method('GuidedLIME-L2', lime.GuidedLinearLIME(n_samples=200, dist_func_name='l2',
            #                                               kernel_width=l2_kernel_width)),
            # Method('WeightedLIME-L2', lime.WeightedLinearLIME(n_samples=200, dist_func_name='l2',
            #                                                   kernel_width=l2_kernel_width,
            #                                                   alpha=0.05, method='single')),
            # Method('MILLI', milli.Milli(n_samples=200, alpha=0.05, beta=-0.01, method='single')),
            Method('Paired-MILLI', milli.Milli(n_samples=200, alpha=0.05, beta=-0.01, method='single',
                                               paired_sampling=True)),
        ]

        return attribution_grid

    def load_model(self, model_clz, path):
        model = model_clz(self.device)
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        model.eval()
        return model

    def load_test_dataset(self, seed):
        _, _, test_dataset = create_datasets(random_state=seed)
        return test_dataset
