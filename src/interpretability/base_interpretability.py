import os
import pickle as pkl
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
from texttable import Texttable
from tqdm import tqdm

from interpretability import metrics as met
from interpretability.instance_attribution.base_instance_attribution import InherentInterpretabilityError
from interpretability.interpretability_util import get_pred
from functools import partial


Method = namedtuple('Method', ['name', 'method'])
Metric = namedtuple('Metric', ['name', 'metric'])
Model = namedtuple('Model', ['name', 'clz', 'path_fmt'])


class InterpretabilityStudy(ABC):

    def __init__(self, device, n_classes, out_dir, n_repeats=None, all_clz_attribution=True):
        self.device = device
        self.n_classes = n_classes
        self.attribution_methods = self.get_attribution_method_grid()
        self.attribution_metrics = self.get_attribution_metric_grid()
        self.models = self.get_model_grid()
        self.seeds = self.get_seeds()
        self.n_repeats = n_repeats if n_repeats is not None else len(self.seeds)
        self.all_clz_attribution = all_clz_attribution
        self.out_dir = out_dir

    @staticmethod
    @abstractmethod
    def get_clz_target_mask(instance_targets, clz):
        pass

    @abstractmethod
    def get_model_grid(self):
        pass

    @abstractmethod
    def get_attribution_method_grid(self):
        pass

    @abstractmethod
    def load_model(self, model_clz, clz):
        pass

    @abstractmethod
    def load_test_dataset(self, seed):
        pass

    def run_study(self):
        for model_info in self.models:
            print('Running for model {:s}'.format(model_info.name))
            self.evaluate_model_with_repeats(model_info)

    def output_study(self, n_repeats):
        model_attribution_dicts = []
        for model_info in self.models:
            model_attribution_results = {}
            model_dir = self.out_dir + "/{:s}".format(model_info.name)
            for repeat in range(n_repeats):
                with open(model_dir + "/{:d}_attribution.pkl".format(repeat), 'rb') as f:
                    repeat_attribution_results = pkl.load(f)
                    for key, values in repeat_attribution_results.items():
                        if key not in model_attribution_results:
                            model_attribution_results[key] = []
                        model_attribution_results[key].extend(values)
            model_attribution_dicts.append(model_attribution_results)

        print('\n\n### ATTRIBUTION RESULTS ###')
        for metric in self.attribution_metrics:
            print('\n--- {:s} Results ---'.format(metric.name))
            self.output_results_by_metric(model_attribution_dicts, metric.name, self.attribution_methods)

    def evaluate_model_with_repeats(self, model_info):
        # Run for n_repeats.
        #   Different model and test dataset for each repeat,
        #   but seeded to make sure its the true test data for that model.
        for repeat in range(self.n_repeats):
            print('Repeat {:d}/{:d}'.format(repeat + 1, self.n_repeats))

            model_attribution_results = self.evaluate_model_single(model_info, repeat_num=repeat)

            model_dir = self.out_dir + "/{:s}".format(model_info.name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            with open(model_dir + "/{:d}_attribution.pkl".format(repeat), 'wb+') as f:
                pkl.dump(model_attribution_results, f)

    def evaluate_model_single(self, model_info, repeat_num=0):
        # Setup attribution results output
        model_attribution_results = {}
        for method in self.attribution_methods:
            for metric in self.attribution_metrics:
                model_attribution_results[(method.name, metric.name)] = []

        model = self.load_model(model_info.clz, model_info.path_fmt.format(repeat_num))
        dataset = self.load_test_dataset(self.seeds[repeat_num])
        self.evaluate_model_clz_attributions(model, dataset, model_attribution_results)

        return model_attribution_results

    def evaluate_model_clz_attributions(self, model, dataset, model_results):
        for idx in tqdm(range(len(dataset)), desc='Getting instance attribution scores'):
            bag, label, instance_targets = dataset.get_bag_verbose(idx)
            label = int(label)
            pred = get_pred(bag, model)
            if len(bag) <= 2:
                continue

            if self.all_clz_attribution:
                clzs = range(self.n_classes)
            else:
                if label == 0:
                    clzs = [0]
                else:
                    clzs = [0, label]
            for clz in clzs:
                clz_instance_targets = self.get_clz_target_mask(instance_targets, clz)
                # Get scores for each method
                for method in self.attribution_methods:
                    try:
                        instance_attributions = method.method.get_instance_clz_attributions(bag, model, pred, clz)
                        for metric in self.attribution_metrics:
                            try:
                                if metric.name == 'AOPC':
                                    value = metric.metric(instance_attributions, bag, model, pred, clz)
                                else:
                                    value = metric.metric(instance_attributions, clz_instance_targets)
                                model_results[(method.name, metric.name)].append(value)
                            # Skip if there aren't any instances that support this class
                            except met.MetricError:
                                pass
                    except InherentInterpretabilityError:
                        for metric in self.attribution_metrics:
                            model_results[(method.name, metric.name)].append(0)


    def output_results_by_metric(self, results_grouped_by_model, metric_name, methods):
        # Regroup results by metric rather than by model
        results_grouped_by_metric = {}
        for model_info in self.models:
            for method in methods:
                results_grouped_by_metric[(model_info.name, method.name)] = []
        for model_idx, model_dict in enumerate(results_grouped_by_model):
            model_info = self.models[model_idx]
            for key, values in model_dict.items():
                if key[1] == metric_name:
                    results_grouped_by_metric[(model_info.name, key[0])].extend(values)

        # Output results grouped by metric
        header = ['Methods'] + [m.name for m in self.models] + ['Overall']
        rows = [header]
        for method in methods:
            row = [method.name]
            means = []
            sems = []
            for model_info in self.models:
                model_method_results = results_grouped_by_metric[(model_info.name, method.name)]
                mean = np.average(model_method_results)
                if mean == 0:
                    row += ['N/A']
                else:
                    sem = np.std(model_method_results) / np.sqrt(len(model_method_results))
                    row += ['{:.4f} +- {:.4f}'.format(mean, sem)]
                    means.append(mean)
                    sems.append(sem)
            row_mean = np.mean(means)
            row_sem = np.sqrt(np.mean([s ** 2 for s in sems]))
            row += ['{:.4f} +- {:.4f}'.format(row_mean, row_sem)]
            rows.append(row)
        n_cols = len(header)
        table = Texttable()
        table.set_cols_dtype(['t'] * n_cols)
        table.set_cols_align(['c'] * n_cols)
        table.add_rows(rows)
        table.set_max_width(0)
        print(table.draw())

    def get_attribution_metric_grid(self):
        evaluation_grid = [
            Metric('NDCG@N', met.normalized_discounted_cumulative_gain),
            Metric('AOPC', partial(met.perturbation_metric, n_random=10)),
        ]
        return evaluation_grid

    def get_seeds(self):
        return [868, 207, 702, 999, 119, 401, 74, 9, 741, 744]
