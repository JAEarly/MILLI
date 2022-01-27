import pickle as pkl

import numpy as np
import torch
from matplotlib import pyplot as plt

from interpretability.metrics import normalized_discounted_cumulative_gain, perturbation_metric
from interpretability.instance_attribution import lime_instance_attribution as lime
from interpretability.instance_attribution import milli_instance_attribution as milli
from interpretability.instance_attribution import shap_instance_attribution as shap
from interpretability.base_interpretability import Method, Metric
from interpretability.mnist_interpretability import MnistInterpretabilityStudy
from interpretability.sival_interpretability import SivalInterpretabilityStudy
from interpretability.crc_interpretability import CrcInterpretabilityStudy
from interpretability.musk_interpretability import MuskInterpretabilityStudy
from functools import partial

plt.style.use(['science', 'bright', 'grid'])


def run_experiment(save_path, dataset_name, model_name):
    print('Running sample size experiment for {:s}'.format(dataset_name))
    print('Save path: {:s}'.format(save_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_repeats = 3

    study, model_info, method_clzs = get_setup(dataset_name, model_name, device, n_repeats)

    if dataset_name == 'sival' or dataset_name == 'mnist':
        sample_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    elif dataset_name == 'musk':
        sample_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    elif dataset_name == 'crc':
        sample_sizes = [5, 50, 100, 250, 500, 750, 1000, 1250, 1500]
    else:
        raise NotImplementedError('No sample sizes setup for {:s}'.format(dataset_name))

    if dataset_name == 'musk':
        metrics = [Metric('AOPC', partial(perturbation_metric, n_random=10))]
    else:
        metrics = [Metric('NDCG@N', normalized_discounted_cumulative_gain)]

    print('Sample sizes:', sample_sizes)
    print('  {:d} total'.format(len(sample_sizes)))

    results = {}
    for method in method_clzs:
        for metric in metrics:
            for sample_size in sample_sizes:
                results[(method.name, metric.name, sample_size)] = []

    for n_samples in sample_sizes:
        print('Running for n_samples: {:d}'.format(n_samples))
        study.attribution_metrics = metrics
        methods = []
        for m in method_clzs:
            methods.append(Method(m.name, m.method(n_samples=n_samples)))

        study.attribution_methods = methods

        for repeat in range(n_repeats):
            print('Repeat {:d}/{:d}'.format(repeat + 1, n_repeats))
            run_results = study.evaluate_model_single(model_info, repeat_num=repeat)
            for (method_name, metric_name), scores in run_results.items():
                if (method_name, metric_name, n_samples) not in results:
                    results[(method_name, metric_name, n_samples)] = []
                results[method_name, metric_name, n_samples].extend(scores)

    with open(save_path, 'wb+') as f:
        pkl.dump(results, f)


def get_setup(dataset_name, model_name, device, n_repeats):
    if dataset_name == 'mnist':
        study = MnistInterpretabilityStudy(device, n_repeats=n_repeats)
        model_info = get_model(study, model_name)
        l2_kernel_width = lime.get_l2_kernel_width(30)
        method_clzs = [
            Method("RandomSHAP", shap.RandomKernelSHAP),
            Method("GuidedSHAP", shap.GuidedKernelSHAP),
            Method("RandomLIME", partial(lime.RandomLinearLIME, dist_func_name='l2', kernel_width=l2_kernel_width)),
            Method("GuidedLIME", partial(lime.GuidedLinearLIME, dist_func_name='l2', kernel_width=l2_kernel_width)),
            Method("MILLI", partial(milli.Milli, alpha=0.1, beta=-0.01, method='single')),
        ]
        return study, model_info, method_clzs
    if dataset_name == 'sival':
        study = SivalInterpretabilityStudy(device, n_repeats=n_repeats)
        model_info = get_model(study, model_name)
        l2_kernel_width = lime.get_l2_kernel_width(30)
        method_clzs = [
            Method("RandomSHAP", shap.RandomKernelSHAP),
            Method("GuidedSHAP", shap.GuidedKernelSHAP),
            Method("RandomLIME", partial(lime.RandomLinearLIME, dist_func_name='l2', kernel_width=l2_kernel_width)),
            Method("GuidedLIME", partial(lime.GuidedLinearLIME, dist_func_name='l2', kernel_width=l2_kernel_width)),
            Method("MILLI", partial(milli.Milli, alpha=0.05, beta=-0.01, method='single')),
        ]
        return study, model_info, method_clzs
    if dataset_name == 'crc':
        study = CrcInterpretabilityStudy(device, n_repeats=n_repeats)
        model_info = get_model(study, model_name)
        l2_kernel_width = lime.get_l2_kernel_width(264)
        method_clzs = [
            Method('RandomSHAP', shap.RandomKernelSHAP),
            Method('GuidedSHAP', shap.GuidedKernelSHAP),
            Method('RandomLIME-L2', partial(lime.RandomLinearLIME, dist_func_name='l2', kernel_width=l2_kernel_width)),
            Method('GuidedLIME-L2', partial(lime.GuidedLinearLIME, dist_func_name='l2', kernel_width=l2_kernel_width)),
            Method('MILLI', partial(milli.Milli, alpha=0.008, beta=-5, method='single')),
        ]
        return study, model_info, method_clzs
    if dataset_name == 'musk':
        study = MuskInterpretabilityStudy(device, n_repeats=n_repeats)
        model_info = get_model(study, model_name)
        l2_kernel_width = lime.get_l2_kernel_width(5)
        method_clzs = [
            Method("RandomSHAP", partial(shap.RandomKernelSHAP, allow_repeats=True)),
            Method("GuidedSHAP", partial(shap.GuidedKernelSHAP, allow_repeats=True)),
            Method("RandomLIME", partial(lime.RandomLinearLIME, dist_func_name='l2', kernel_width=l2_kernel_width,
                                         allow_repeats=True)),
            Method("GuidedLIME", partial(lime.GuidedLinearLIME, dist_func_name='l2', kernel_width=l2_kernel_width,
                                         allow_repeats=True)),
            Method("MILLI", partial(milli.Milli, alpha=0.1, beta=-0.1, method='single', allow_repeats=True)),
        ]
        return study, model_info, method_clzs
    raise NotImplementedError('No study and model setup for dataset {:s}'.format(dataset_name))


def get_model(study, model_name):
    for m in study.models:
        if m.name == model_name:
            return m
    raise ValueError('No model for name {:s}'.format(model_name))


def output_single_results(path, dataset_name):
    results, method_names, _, sample_sizes = parse_results_file(path)

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
    metric = 'AOPC' if dataset_name == 'musk' else 'NDCG@N'
    plot_results(axis, method_names, sample_sizes, metric, results)

    axis.set_title(title)

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=[1.0, 0.53])

    plt.tight_layout(rect=[0, 0, 0.65, 1])
    # fig_path = path[:path.rindex(".pkl")] + ".png"
    # fig.savefig(fig_path, format='png', dpi=1200)
    plt.show()


def output_multiple_results(model_name):
    dataset_names = ['sival', 'mnist', 'crc']
    ylims = [[0.5, 0.9], [0.4, 1.0], [0.6, 0.9]]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 2))

    for idx, dataset_name in enumerate(dataset_names):
        path = get_path(dataset_name, model_name)
        try:
            results, method_names, _, sample_sizes = parse_results_file(path)
        except FileNotFoundError:
            print('No data for {:s}'.format(dataset_name))
            continue
        ylim_min, ylim_max = ylims[idx]
        axis = axes[idx]
        plot_results(axis, method_names, sample_sizes, 'NDCG@N', results, ylim_min=ylim_min, ylim_max=ylim_max,
                     label_x=idx == 1, label_y=idx == 0)

        title = dataset_name.upper() if dataset_name != 'mnist' else '4-MNIST-Bags'
        axis.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=[1.0, 0.53])
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    fig_path = 'out/sample_size/multi_{:s}_SampleSize.png'.format(model_name)
    fig.savefig(fig_path, format='png', dpi=400)
    plt.show()


def parse_results_file(path):
    print('Parsing {:s}'.format(path))
    with open(path, 'rb') as f:
        results = pkl.load(f)
    method_names = []
    metric_names = []
    sample_sizes = []
    for key in results.keys():
        method, metric, n_samples = key
        if method not in method_names:
            method_names.append(method)
        if metric not in metric_names:
            metric_names.append(metric)
        if n_samples not in sample_sizes:
            sample_sizes.append(int(n_samples))
    sample_sizes = sorted(sample_sizes)
    print('Found methods:', method_names)
    print('Found metrics:', metric_names)
    print('Found sample_sizes:', sample_sizes)
    return results, method_names, metric_names, sample_sizes


def plot_results(axis, method_names, sample_sizes, metric, results, ylim_min=None, ylim_max=None,
                 label_x=True, label_y=True):
    min_y = None
    max_y = None
    for method in method_names:
        y_means = []
        y_errs = []
        for n_samples in sample_sizes:
            y = results[(method, metric, n_samples)]
            y_means.append(np.mean(y))
            y_errs.append(np.std(y) / np.sqrt(len(y)))

        y_means = np.asarray(y_means)
        axis.plot(sample_sizes, y_means, label=method)

        # y_errs = np.asarray(y_errs)
        # axis.fill_between(xs, y_means - y_errs, y_means + y_errs, alpha=0.5)

        min_y = min(y_means) if min_y is None or min(y_means) < min_y else min_y
        max_y = max(y_means) if max_y is None or max(y_means) > max_y else max_y

    axis.set_xlim(min(sample_sizes), max(sample_sizes))
    axis.set_ylim(ylim_min if ylim_min is not None else 0.95 * min_y,
                  ylim_max if ylim_max is not None else 1.05 * max_y)
    if label_x:
        axis.set_xlabel('Number of Samples')
    if label_y:
        axis.set_ylabel(metric)

    ticks = axis.get_xticks()
    ticks[0] = 5
    axis.set_xticks(ticks)


def get_path(dataset_name, model_name):
    return 'out/sample_size/{:s}_{:s}_SampleSize.pkl'.format(dataset_name, model_name)


if __name__ == "__main__":
    _model_name = 'MI-GNN'

    output_multiple_results(_model_name)
    #
    # _gather_data = True
    # _dataset_name = 'crc'
    # _path = get_path(_dataset_name, _model_name)
    # if _gather_data:
    #     run_experiment(_path, _dataset_name, _model_name)
    # output_single_results(_path, _dataset_name)
