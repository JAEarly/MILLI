from collections import Counter

import numpy as np
import torch

from data.crc.crc_dataset import load_crc
from data.mnist_bags import create_andmil_datasets
from data.sival.sival_dataset import create_datasets


def run(dataset_name):
    train_dataset, val_dataset, test_dataset = get_datasets_and_n_clzs(dataset_name)
    get_wr(train_dataset, val_dataset, test_dataset)


def get_wr(train_dataset, val_dataset, test_dataset):
    wrs = []
    wrs.extend(get_dataset_wr(train_dataset))
    wrs.extend(get_dataset_wr(val_dataset))
    wrs.extend(get_dataset_wr(test_dataset))
    print('WR: {:.2f}%'.format(np.mean(wrs)))


def get_dataset_wr(dataset):
    wrs = []
    for bag_instance_targets in dataset.instance_targets:
        flat_targets = []
        for target in bag_instance_targets:
            if type(target) is list:
                flat_targets.extend(target)
            elif type(target) is torch.Tensor:
                flat_targets.append(target.item())
            else:
                flat_targets.append(target)
        c = Counter(flat_targets)
        wr = (1 - c[0] / len(flat_targets)) * 100
        wrs.append(wr)
    return wrs


def get_datasets_and_n_clzs(dataset_name):
    if dataset_name == 'mnist':
        return create_andmil_datasets(30, 2, 2500, random_state=5, verbose=False, num_test_bags=100)
    if dataset_name == 'sival':
        return create_datasets()
    if dataset_name == 'crc':
        return load_crc()
    raise ValueError('Invalid dataset name: {:s}'.format(dataset_name))


if __name__ == "__main__":
    print('MNIST')
    run('mnist')
    print('\nSIVAL')
    run('sival')
    print('\nCRC')
    run('crc')
