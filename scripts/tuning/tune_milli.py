import pickle as pkl

import numpy as np
import torch
from matplotlib import pyplot as plt
from texttable import Texttable

from interpretability import metrics as met
from interpretability.base_interpretability import Method, Model, Metric
from interpretability.crc_interpretability import CrcInterpretabilityStudy
from interpretability.instance_attribution import milli_instance_attribution as milli
from interpretability.mnist_interpretability import MnistInterpretabilityStudy
from interpretability.sival_interpretability import SivalInterpretabilityStudy
from interpretability.musk_interpretability import MuskInterpretabilityStudy
from interpretability.tef_interpretability import TefInterpretabilityStudy
from model import mnist_models, sival_models, crc_models, musk_models, tef_models
from functools import partial


def run(dataset_name, save_path, paired_sampling):
    print('Running MILLI tuning experiment for {:s}'.format(dataset_name))
    n_repeats = 1

    alphas = [0.005, 0.01, 0.02, 0.98, 0.99, 0.995]
    betas = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    n_combinations = len(alphas) * len(betas)

    print('alphas:', alphas)
    print('  {:d} total'.format(len(alphas)))
    print('betas:', betas)
    print('  {:d} total'.format(len(betas)))
    print('{:d} combinations'.format(n_combinations))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_infos, study, n_samples = get_study_and_model(dataset_name, device, n_repeats)

    if dataset_name in ['musk', 'tiger']:
        study.attribution_metrics = [Metric('AOPC', partial(met.perturbation_metric, n_random=10))]
        allow_repeats = True
    else:
        study.attribution_metrics = [Metric('NDCG@N', met.normalized_discounted_cumulative_gain)]
        allow_repeats = False

    alpha_05_result = None

    metric = 'AOPC' if dataset_name in ['musk', 'tiger'] else 'NDCG@N'

    results = {}
    i = 1
    for alpha in alphas:
        for beta in betas:
            print('Setup {:d}/{:d} - alpha = {:.2f}, beta = {:.2f}'.format(i, n_combinations, alpha, beta))

            if alpha != 0.5 or alpha_05_result is None:
                method = Method("MILLI", milli.Milli(method='single', alpha=alpha, beta=beta,
                                                     n_samples=n_samples, allow_repeats=allow_repeats,
                                                     paired_sampling=paired_sampling))
                study.attribution_methods = [method]
                run_results = []
                for model_info in model_infos:
                    print('Running for model: {:s}'.format(model_info.name))
                    for repeat in range(n_repeats):
                        print('Repeat {:d}/{:d}'.format(repeat + 1, n_repeats))
                        model_attribution_results = study.evaluate_model_single(model_info, repeat_num=repeat)
                        run_results.extend(model_attribution_results[('MILLI', metric)])

                mean = np.mean(run_results)
                err = np.std(run_results) / np.sqrt(len(run_results))

                if alpha == 0.5 and alpha_05_result is None:
                    alpha_05_result = (mean, err)
            else:
                print('Skipping as we already have results for alpha = 0.5')
                (mean, err) = alpha_05_result

            results[(alpha, beta)] = (mean, err)
            i += 1

    with open(save_path, 'wb+') as f:
        pkl.dump(results, f)
    print('Saved to {:s}'.format(save_path))


def output_results(path):
    with open(path, 'rb') as f:
        results = pkl.load(f)

    alphas = sorted(list(set(k[0] for k in results.keys())))
    betas = sorted(list(set(k[1] for k in results.keys())))
    n_combinations = len(alphas) * len(betas)
    print('alphas:', alphas)
    print('  {:d} total'.format(len(alphas)))
    print('betas:', betas)
    print('  {:d} total'.format(len(betas)))
    print('{:d} combinations'.format(n_combinations))

    flat_results = []

    rows = []
    header = ["{:.2f}".format(alpha) for alpha in alphas]
    header.insert(0, 'b\\a')
    rows.append(header)
    for beta in betas:
        row = ["{:.2f}".format(beta)]
        for alpha in alphas:
            score = results[alpha, beta][0]
            row.append("{:.4f}".format(score))
            flat_results.append(((alpha, beta), score))
        rows.append(row)

    table = Texttable()
    table.set_cols_dtype(['t'] * (len(alphas) + 1))
    table.add_rows(rows)
    table.set_max_width(0)

    print('\n-- Results Table --')
    print(table.draw())

    sorted_results = sorted(flat_results, key=lambda x: x[1], reverse=True)
    print('\n-- Top 5 --')
    for i in range(min(5, len(sorted_results))):
        print(i + 1, sorted_results[i][0], "{:.4f}".format(sorted_results[i][1]))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

    for beta in betas:
        ys = [results[(alpha, beta)][0] for alpha in alphas]
        axes[0].plot(alphas, ys, label='beta = {:.2f}'.format(beta))
    axes[0].set_xlim(min(alphas), max(alphas))
    axes[0].set_xlabel('alpha')
    axes[0].set_ylabel('NDCG@n')
    axes[0].legend(loc='best')

    for alpha in alphas:
        ys = [results[(alpha, beta)][0] for beta in betas]
        axes[1].plot(betas, ys, label='alpha = {:.2f}'.format(alpha))
    axes[1].set_xlim(min(betas), max(betas))
    axes[1].set_xlabel('beta')
    axes[1].set_ylabel('NDCG@n')
    axes[1].legend(loc='best')

    plt.tight_layout()
    plt.show()


def get_study_and_model(dataset_name, device, n_repeats):
    if dataset_name == 'mnist':
        model_infos = [
            Model('EmbeddedSpace-Net', mnist_models.MnistEmbeddingSpaceNN,
                  'models/mnist/MnistEmbeddingSpaceNN/MnistEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', mnist_models.MnistInstanceSpaceNN,
                  'models/mnist/MnistInstanceSpaceNN/MnistInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', mnist_models.MnistAttentionNN,
                  'models/mnist/MnistAttentionNN/MnistAttentionNN_{:d}.pkl'),
            Model('MI-GNN', mnist_models.MnistGNN,
                  'models/mnist/MnistGNN/MnistGNN_{:d}.pkl'),
        ]
        study = MnistInterpretabilityStudy(device, n_repeats=n_repeats)
        n_samples = 150
        return model_infos, study, n_samples
    if dataset_name == 'sival':
        model_infos = [
            Model('EmbeddedSpace-Net', sival_models.SivalEmbeddingSpaceNN,
                  'models/sival/SivalEmbeddingSpaceNN/SivalEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', sival_models.SivalInstanceSpaceNN,
                  'models/sival/SivalInstanceSpaceNN/SivalInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', sival_models.SivalAttentionNN,
                  'models/sival/SivalAttentionNN/SivalAttentionNN_{:d}.pkl'),
            Model('MI-GNN', sival_models.SivalGNN,
                  'models/sival/SivalGNN/SivalGNN_{:d}.pkl'),
        ]
        study = SivalInterpretabilityStudy(device, n_repeats=n_repeats)
        n_samples = 200
        return model_infos, study, n_samples
    if dataset_name == 'crc':
        model_infos = [
            Model('MI-Attn', crc_models.CrcAttentionNN,
                  'models/crc/CrcAttentionNN/CrcAttentionNN_{:d}.pkl'),
        ]
        study = CrcInterpretabilityStudy(device, n_repeats=n_repeats)
        n_samples = 1000
        return model_infos, study, n_samples
    if dataset_name == 'musk':
        model_infos = [
            Model('EmbeddedSpace-Net', musk_models.MuskEmbeddingSpaceNN,
                  'models/musk1/MuskEmbeddingSpaceNN/MuskEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', musk_models.MuskInstanceSpaceNN,
                  'models/musk1/MuskInstanceSpaceNN/MuskInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', musk_models.MuskAttentionNN,
                  'models/musk1/MuskAttentionNN/MuskAttentionNN_{:d}.pkl'),
            Model('MI-GNN', musk_models.MuskGNN,
                  'models/musk1/MuskGNN/MuskGNN_{:d}.pkl'),
        ]
        study = MuskInterpretabilityStudy(device, n_repeats=n_repeats)
        n_samples = 150
        return model_infos, study, n_samples
    if dataset_name == 'tiger':
        model_infos = [
            Model('EmbeddedSpace-Net', tef_models.TefEmbeddingSpaceNN,
                  'models/' + dataset_name + '/TefEmbeddingSpaceNN/TefEmbeddingSpaceNN_{:d}.pkl'),
            Model('InstanceSpace-Net', tef_models.TefInstanceSpaceNN,
                  'models/' + dataset_name + '/TefInstanceSpaceNN/TefInstanceSpaceNN_{:d}.pkl'),
            Model('MI-Attn', tef_models.TefAttentionNN,
                  'models/' + dataset_name + '/TefAttentionNN/TefAttentionNN_{:d}.pkl'),
            Model('MI-GNN', tef_models.TefGNN,
                  'models/' + dataset_name + '/TefGNN/TefGNN_{:d}.pkl'),
        ]
        study = TefInterpretabilityStudy(device, dataset_name, n_repeats=n_repeats)
        n_samples = 150
        return model_infos, study, n_samples
    raise NotImplementedError('No study and model setup for dataset {:s}'.format(dataset_name))


if __name__ == "__main__":
    _gather_data = True
    _dataset_name = 'crc'
    _paired_sampling = True

    _save_path = "out/tune/tune_{:s}MILLI_{:s}_results.pkl".format("Paired-" if _paired_sampling else "", _dataset_name)
    if _gather_data:
        run(_dataset_name, _save_path, _paired_sampling)
    output_results(_save_path)
