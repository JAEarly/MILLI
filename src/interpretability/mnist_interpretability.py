import numpy as np
import torch

from data.mnist_bags import create_andmil_datasets, MNIST_N_CLASSES
from interpretability import metrics as met
from interpretability.base_interpretability import Model, InterpretabilityStudy, Method, Metric
from interpretability.instance_attribution import independent_instance_attribution as indep
from interpretability.instance_attribution import lime_instance_attribution as lime
from interpretability.instance_attribution import milli_instance_attribution as milli
from interpretability.instance_attribution import shap_instance_attribution as shap
from interpretability.instance_attribution.base_instance_attribution import InherentAttribution
from model import mnist_models
from functools import partial


class MnistInterpretabilityStudy(InterpretabilityStudy):

    def __init__(self, device, n_repeats=None):
        super().__init__(device, MNIST_N_CLASSES, "out/interpretability_studies/mnist", n_repeats=n_repeats)

    @staticmethod
    def get_clz_target_mask(instance_targets, clz):
        mask_negative_idxs = []
        if clz == 3:
            mask_positive_idxs = (instance_targets != 0).nonzero()[0]
        elif clz == 2:
            mask_positive_idxs = (instance_targets == 2).nonzero()[0]
            mask_negative_idxs = (instance_targets == 1).nonzero()[0]
        elif clz == 1:
            mask_positive_idxs = (instance_targets == 1).nonzero()[0]
            mask_negative_idxs = (instance_targets == 2).nonzero()[0]
        elif clz == 0:
            mask_positive_idxs = (instance_targets == 0).nonzero()[0]
        else:
            raise ValueError('Invalid MNIST class {:}'.format(clz))
        mask = np.zeros(len(instance_targets))
        mask[mask_positive_idxs] = 1
        mask[mask_negative_idxs] = -1
        return mask

    def get_model_grid(self):
        model_grid = [
            Model('EmbeddedSpace-Net', mnist_models.MnistEmbeddingSpaceNN,
                  'models/mnist/MnistEmbeddingSpaceNN/MnistEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', mnist_models.MnistInstanceSpaceNN,
                  'models/mnist/MnistInstanceSpaceNN/MnistInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', mnist_models.MnistAttentionNN,
                  'models/mnist/MnistAttentionNN/MnistAttentionNN_{:d}.pkl'),
            # Model('MI-GNN', mnist_models.MnistGNN,
            #       'models/mnist/MnistGNN/MnistGNN_{:d}.pkl'),
        ]
        return model_grid

    def get_attribution_method_grid(self):
        l2_kernel_width = lime.get_l2_kernel_width(30)
        attribution_grid = [
            # Method('Inherent', InherentAttribution()),
            # Method('Single', indep.SingleInstanceAttribution()),
            # Method('One Removed', indep.OneRemovedInstanceAttribution()),
            # Method('Combined', indep.CombinedInstanceAttribution('mean')),
            # Method('RandomSHAP', shap.RandomKernelSHAP(n_samples=150)),
            # Method('GuidedSHAP', shap.GuidedKernelSHAP(n_samples=150)),
            # Method('WeightedSHAP', shap.WeightedKernelSHAP(n_samples=150, alpha=0.05, method='single')),
            # Method('RandomLIME-L2', lime.RandomLinearLIME(n_samples=150, dist_func_name='l2',
            #                                               kernel_width=l2_kernel_width)),
            # Method('GuidedLIME-L2', lime.GuidedLinearLIME(n_samples=150, dist_func_name='l2',
            #                                               kernel_width=l2_kernel_width)),
            # Method('WeightedLIME-L2', lime.WeightedLinearLIME(n_samples=150, dist_func_name='l2',
            #                                                   kernel_width=l2_kernel_width,
            #                                                   alpha=0.05, method='single')),
            # Method('MILLI', milli.Milli(n_samples=150, alpha=0.05, beta=0.01, method='single')),
            Method('Paired-MILLI', milli.Milli(n_samples=150, alpha=0.05, beta=0.01, method='single',
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
        _, _, test_dataset = create_andmil_datasets(30, 2, 2500, random_state=seed, verbose=False, num_test_bags=100)
        return test_dataset

    def get_attribution_metric_grid(self):
        evaluation_grid = [
            # Metric('NDCG@N', met.normalized_discounted_cumulative_gain),
            Metric('AOPC', partial(met.perturbation_metric, n_random=10)),
        ]
        return evaluation_grid
