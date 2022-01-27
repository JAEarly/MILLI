import pickle as pkl

import numpy as np
import torch
from matplotlib import pyplot as plt

from interpretability.metrics import normalized_discounted_cumulative_gain
from interpretability.instance_attribution import lime_instance_attribution as lime
from interpretability.base_interpretability import Method, Metric, Model
from interpretability.mnist_interpretability import MnistInterpretabilityStudy
from interpretability.sival_interpretability import SivalInterpretabilityStudy
from interpretability.crc_interpretability import CrcInterpretabilityStudy
from model import mnist_models, sival_models, crc_models
from functools import partial

plt.style.use(['science', 'bright', 'grid'])


def run_experiment(save_path, dataset_name, dist_func_name):
    print('Running kernel width experiment for {:s}'.format(dataset_name))
    print('Save path: {:s}'.format(save_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_repeats = 1

    study_details = get_study_details(dataset_name, device, n_repeats)
    model_info, study, n_samples, alpha, mean_bag_size = study_details

    metrics = [
        Metric('NDCG@N', normalized_discounted_cumulative_gain),
    ]

    method_clzs, widths = get_method_clzs_and_widths(dist_func_name, n_samples, alpha, mean_bag_size)

    print('Kernel widths:', widths)
    print('  {:d} total'.format(len(widths)))

    results = {}
    for method in method_clzs:
        for metric in metrics:
            for width in widths:
                results[(method.name, metric.name, width)] = []

    for width in widths:
        print('Running for width: {:f}'.format(width))
        study.metrics = metrics
        methods = []
        for m in method_clzs:
            methods.append(Method(m.name, m.method(kernel_width=width)))

        study.attribution_methods = methods

        for repeat in range(n_repeats):
            print('Repeat {:d}/{:d}'.format(repeat + 1, n_repeats))
            _, run_results = study.evaluate_model_single(model_info, do_discrim=False, do_attribution=True,
                                                         repeat_num=repeat)
            for (method_name, metric_name), scores in run_results.items():
                if (method_name, metric_name, width) not in results:
                    results[(method_name, metric_name, width)] = []
                results[method_name, metric_name, width].extend(scores)

    with open(save_path, 'wb+') as f:
        pkl.dump(results, f)


def get_study_details(dataset_name, device, n_repeats):
    if dataset_name == 'mnist':
        model_info = Model('MI-Attn', mnist_models.MnistAttentionNN,
                           'models/mnist/MnistAttentionNN/MnistAttentionNN_{:d}.pkl')
        study = MnistInterpretabilityStudy(device, n_repeats=n_repeats)
        n_samples = 150
        alpha = 0.1
        mean_bag_size = 30
        return model_info, study, n_samples, alpha, mean_bag_size
    if dataset_name == 'sival':
        model_info = Model('MI-Attn', sival_models.SivalAttentionNN,
                           'models/sival/SivalAttentionNN/SivalAttentionNN_{:d}.pkl')
        study = SivalInterpretabilityStudy(device, n_repeats=n_repeats)
        return model_info, study
    if dataset_name == 'crc':
        model_info = Model('MI-Attn', crc_models.CrcAttentionNN,
                           'models/crc/CrcAttentionNN/CrcAttentionNN_{:d}.pkl')
        study = CrcInterpretabilityStudy(device, n_repeats=n_repeats)
        return model_info, study
    raise NotImplementedError('No study and model setup for dataset {:s}'.format(dataset_name))


def get_method_clzs_and_widths(dist_func_name, n_samples, alpha, mean_bag_size):
    if dist_func_name == 'l2':
        method_clzs = [
            Method("RandomLIME-L2", partial(lime.RandomLinearLIME, n_samples=n_samples, dist_func_name='l2')),
            Method("GuidedLIME-L2", partial(lime.GuidedLinearLIME, n_samples=n_samples, dist_func_name='l2')),
            Method("WeightedSHAP-L2", partial(lime.WeightedLinearLIME, n_samples=n_samples, dist_func_name='l2',
                                              alpha=alpha, method='single'))]
        half_coalition = np.ones(mean_bag_size)
        half_coalition[:mean_bag_size//2] = 0
        l2_kernel_width = np.sqrt(- lime.l2_dist(half_coalition) ** 2 / np.log(0.25))
        min_width = 0.25
        widths = np.linspace(min_width, 2 * l2_kernel_width - min_width, 5)
        return method_clzs, widths
    if dist_func_name == 'cosine':
        method_clzs = [
            Method("RandomLIME-Cosine", partial(lime.RandomLinearLIME, n_samples=n_samples, dist_func_name='cosine')),
            Method("GuidedLIME-Cosine", partial(lime.GuidedLinearLIME, n_samples=n_samples, dist_func_name='cosine')),
            Method("WeightedSHAP-Cosine", partial(lime.WeightedLinearLIME, n_samples=n_samples, dist_func_name='cosine',
                                                  alpha=alpha, method='single')),
        ]

        half_coalition = np.ones(mean_bag_size)
        half_coalition[:mean_bag_size//2] = 0
        cosine_kernel_width = np.sqrt(- lime.cosine_dist(half_coalition) ** 2 / np.log(0.25))
        min_width = 0.1
        widths = np.linspace(min_width, 2 * cosine_kernel_width - min_width, 5)
        return method_clzs, widths
    raise NotImplementedError('No setup found for distance function {:s}'.format(dist_func_name))


def output_results(path):
    with open(path, 'rb') as f:
        results = pkl.load(f)

    method_names = []
    metric_names = []
    widths = []
    for key in results.keys():
        method, metric, width = key
        if method not in method_names:
            method_names.append(method)
        if metric not in metric_names:
            metric_names.append(metric)
        if width not in widths:
            widths.append(width)
    widths = sorted(widths)

    print('Found methods:', method_names)
    print('Found metrics:', metric_names)
    print('Found widths:', widths)

    metric_names = ['NDCG@N']

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(7, 3))
    for metric_idx, metric in enumerate(metric_names):
        plot_methods(axis, method_names, widths, results, metric)

    plt.tight_layout()
    fig_path = path[:path.rindex(".pkl")] + ".png"
    fig.savefig(fig_path, format='png', dpi=1200)
    plt.show()


def plot_methods(axis, method_names, widths, results, metric):
    for method in method_names:
        y_means = []
        y_errs = []
        for width in widths:
            y = results[(method, metric, width)]
            y_means.append(np.mean(y))
            y_errs.append(np.std(y) / np.sqrt(len(y)))

        y_means = np.asarray(y_means)
        y_errs = np.asarray(y_errs)

        axis.plot(widths, y_means, label=method)
        axis.fill_between(widths, y_means - y_errs, y_means + y_errs, alpha=0.5)

        axis.set_xlim(min(widths), max(widths))
        # axis.set_ylim(0.6, 0.9)
        axis.set_xlabel('Kernel Width')
        axis.set_ylabel(metric)

        # axis.grid()

        axis.legend(loc='lower right')


if __name__ == "__main__":
    _gather_data = True
    _dataset_name = 'mnist'
    _dist_func_name = 'l2'
    _path = 'out/kernel_width/{:s}_{:s}_Attn_KernelWidth.pkl'.format(_dataset_name, _dist_func_name)
    if _gather_data:
        run_experiment(_path, _dataset_name, _dist_func_name)
    output_results(_path)
