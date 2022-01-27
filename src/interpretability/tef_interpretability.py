from functools import partial

import torch

from data.tef_dataset import create_datasets, TEF_N_CLASSES
from interpretability import metrics as met
from interpretability.base_interpretability import Model, InterpretabilityStudy, Method, Metric
from interpretability.instance_attribution import independent_instance_attribution as indep
from interpretability.instance_attribution import lime_instance_attribution as lime
from interpretability.instance_attribution import milli_instance_attribution as milli
from interpretability.instance_attribution import shap_instance_attribution as shap
from interpretability.instance_attribution.base_instance_attribution import InherentAttribution
from model import tef_models


class TefInterpretabilityStudy(InterpretabilityStudy):

    def __init__(self, device, dataset_name, n_repeats=None):
        self.dataset_name = dataset_name
        super().__init__(device, TEF_N_CLASSES, "out/interpretability_studies/tef", n_repeats=n_repeats)

    @staticmethod
    def get_clz_target_mask(instance_targets, clz):
        return None

    def get_model_grid(self):
        model_grid = [
            Model('EmbeddedSpace-Net', tef_models.TefEmbeddingSpaceNN,
                  'models/' + self.dataset_name + '/TefEmbeddingSpaceNN/TefEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', tef_models.TefInstanceSpaceNN,
                  'models/' + self.dataset_name + '/TefInstanceSpaceNN/TefInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', tef_models.TefAttentionNN,
                  'models/' + self.dataset_name + '/TefAttentionNN/TefAttentionNN_{:d}.pkl'),
            Model('MI-GNN', tef_models.TefGNN,
                  'models/' + self.dataset_name + '/TefGNN/TefGNN_{:d}.pkl'),
        ]
        return model_grid

    def get_attribution_method_grid(self):
        l2_kernel_width = lime.get_l2_kernel_width(6)
        attribution_grid = [
            # Method('Inherent', InherentAttribution()),
            # Method('Single', indep.SingleInstanceAttribution()),
            # Method('One Removed', indep.OneRemovedInstanceAttribution()),
            # Method('Combined', indep.CombinedInstanceAttribution('mean')),
            # Method('RandomSHAP', shap.RandomKernelSHAP(n_samples=150, allow_repeats=True)),
            # Method('GuidedSHAP', shap.GuidedKernelSHAP(n_samples=150, allow_repeats=True)),
            # Method('RandomLIME-L2', lime.RandomLinearLIME(n_samples=150, dist_func_name='l2',
            #                                               kernel_width=l2_kernel_width, allow_repeats=True)),
            # Method('GuidedLIME-L2', lime.GuidedLinearLIME(n_samples=150, dist_func_name='l2',
            #                                               kernel_width=l2_kernel_width, allow_repeats=True)),
            # Method('MILLI', milli.Milli(n_samples=150, alpha=0.3, beta=0.01, method='single', allow_repeats=True)),
            Method('Paired-MILLI', milli.Milli(n_samples=150, alpha=0.3, beta=0.01, method='single',
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
        _, _, test_dataset = create_datasets(self.dataset_name, random_state=seed)
        return test_dataset

    def get_attribution_metric_grid(self):
        evaluation_grid = [
            Metric('AOPC', partial(met.perturbation_metric, n_random=10)),
        ]
        return evaluation_grid
