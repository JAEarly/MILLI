import csv
from texttable import Texttable
import latextable
import numpy as np


ndcg_files = ["results/crc/crc_interpretability_NDCG.txt", "results/mnist/MNIST_Interpretability_NDCG.txt",
              "results/sival/SIVAL_Interpretability_NDCG.txt"]
aopc_files = ["results/mnist/MNIST_Interpretability_AOPC.txt", "results/sival/SIVAL_Interpretability_AOPC.txt",
              "results/musk/musk1_Interpretability.txt", "results/tef/tiger/tiger_Interpretability.txt",
              "results/tef/elephant/elephant_Interpretability.txt", "results/tef/fox/fox_Interpretability.txt"]

methods = ['Inherent', 'Single', 'One Removed', 'Combined', 'RandomSHAP', 'GuidedSHAP',
           'RandomLIME-L2', 'GuidedLIME-L2', 'MILLI']


def run():
    ndcg_results = parse_files(ndcg_files)
    aopc_results = parse_files(aopc_files)
    output_table(ndcg_results, aopc_results)


def parse_files(files):
    results = {}
    for file in files:
        with open(file) as f:
            reader = csv.reader(f, delimiter='|')
            for line in reader:
                parse_line(line, results)
    return results


def parse_line(line, results):
    line = [l.strip() for l in line]
    if len(line) == 0 or line[0] != '' or line[1] == 'Methods' or line[1] not in methods:
        return
    method_name = line[1]
    mean, sem = [float(s.strip()) for s in line[-2].split('+-')]
    if method_name not in results:
        results[method_name] = []
    results[method_name].append((mean, sem))


def generate_table_row(results, n_expected):
    mean_results = {}
    for method, method_results in results.items():
        assert len(method_results) == n_expected
        means = [x[0] for x in method_results]
        sems = [x[1] for x in method_results]
        print(means)
        mean = np.mean(means)
        sem = np.sqrt(np.mean([s ** 2 for s in sems]))
        mean_results[method] = (mean, sem)
    line = ['{:.3f} +- {:.3f}'.format(*mean_results[m]) for m in methods]
    return line


def output_table(ndcg_results, aopc_results):
    ndcg_line = generate_table_row(ndcg_results, len(ndcg_files))
    aopc_line = generate_table_row(aopc_results, len(aopc_files))

    header = ["Metric"] + methods
    ndcg_line = ["NDCG@n"] + ndcg_line
    aopc_line = ["AOPC-R"] + aopc_line

    lines = [header, ndcg_line, aopc_line]

    n_cols = len(header)

    table = Texttable()
    table.set_cols_dtype(['t']*n_cols)
    table.set_cols_align(['c']*n_cols)
    table.add_rows(lines)
    table.set_max_width(0)
    print(table.draw())

    # print(latextable.draw_latex(table, use_booktabs=True, drop_columns=['Train Loss', 'Val Loss', 'Test Loss']))


if __name__ == "__main__":
    run()
