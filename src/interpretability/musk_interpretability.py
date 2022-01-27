from functools import partial

import torch

from data.musk_dataset import create_datasets, MUSK_N_CLASSES
from interpretability import metrics as met
from interpretability.base_interpretability import Model, InterpretabilityStudy, Method, Metric
from interpretability.instance_attribution import independent_instance_attribution as indep
from interpretability.instance_attribution import lime_instance_attribution as lime
from interpretability.instance_attribution import milli_instance_attribution as milli
from interpretability.instance_attribution import shap_instance_attribution as shap
from interpretability.instance_attribution.base_instance_attribution import InherentAttribution
from model import musk_models


class MuskInterpretabilityStudy(InterpretabilityStudy):

    def __init__(self, device, n_repeats=None):
        super().__init__(device, MUSK_N_CLASSES, "out/interpretability_studies/musk", n_repeats=n_repeats)

    @staticmethod
    def get_clz_target_mask(instance_targets, clz):
        return None

    def get_model_grid(self):
        model_grid = [
            Model('EmbeddedSpace-Net', musk_models.MuskEmbeddingSpaceNN,
                  'models/musk1/MuskEmbeddingSpaceNN/MuskEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', musk_models.MuskInstanceSpaceNN,
                  'models/musk1/MuskInstanceSpaceNN/MuskInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', musk_models.MuskAttentionNN,
                  'models/musk1/MuskAttentionNN/MuskAttentionNN_{:d}.pkl'),
            Model('MI-GNN', musk_models.MuskGNN,
                  'models/musk1/MuskGNN/MuskGNN_{:d}.pkl'),
        ]
        return model_grid

    def get_attribution_method_grid(self):
        l2_kernel_width = lime.get_l2_kernel_width(5)
        attribution_grid = [
            # Method('Inherent', InherentAttribution()),
            # Method('Single', indep.SingleInstanceAttribution()),
            # Method('One Removed', indep.OneRemovedInstanceAttribution()),
            # Method('Combined', indep.CombinedInstanceAttribution('mean')),
            # Method('RandomSHAP', shap.RandomKernelSHAP(n_samples=150, allow_repeats=True)),
            # Method('GuidedSHAP', shap.GuidedKernelSHAP(n_samples=150, allow_repeats=True)),
            # Method('RandomLIME-L2', lime.RandomLinearLIME(n_samples=150, dist_func_name='l2', allow_repeats=True,
            #                                               kernel_width=l2_kernel_width)),
            # Method('GuidedLIME-L2', lime.GuidedLinearLIME(n_samples=150, dist_func_name='l2', allow_repeats=True,
            #                                               kernel_width=l2_kernel_width)),
            # Method('MILLI', milli.Milli(n_samples=150, alpha=0.3, beta=-1.0, method='single', allow_repeats=True)),
            Method('Paired-MILLI', milli.Milli(n_samples=150, alpha=0.3, beta=-1.0, method='single',
                                               allow_repeats=True, paired_sampling=True)),
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

    def get_attribution_metric_grid(self):
        evaluation_grid = [
            Metric('AOPC', partial(met.perturbation_metric, n_random=10)),
        ]
        return evaluation_grid
