import numpy as np
import torch
from texttable import Texttable
from torch.utils.data import DataLoader

from data import musk_dataset
from data import tef_dataset
from model import musk_models, tef_models
from train.musk_training import MuskNetTrainer, MuskGNNTrainer
from train.tef_training import TefNetTrainer, TefGNNTrainer

SEEDS = [868, 207, 702, 999, 119, 401, 74, 9, 741, 744]


def run(dataset_name):
    print('Getting results for dataset {:s}'.format(dataset_name))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_data = get_model_data(dataset_name)
    for (model_clz, fmt_path) in model_data:
        print('Running for', model_clz)
        trainer = get_trainer(device, dataset_name, model_clz)
        run_single(device, dataset_name, model_clz, fmt_path, trainer)


def run_single(device, dataset_name, model_clz, fmt_path, trainer):
    results = get_results(device, dataset_name, model_clz, fmt_path, trainer)
    print(model_clz)
    output_results(results)


def get_results(device, dataset_name, model_clz, fmt_path, trainer):
    results = []
    for i in range(len(SEEDS)):
        print('Model {:d}/{:d}'.format(i + 1, len(SEEDS)))
        train_dataloader, val_dataloader, test_dataloader = load_datasets(dataset_name, seed=SEEDS[i])
        path = fmt_path.format(i)
        model = load_model(device, model_clz, path)
        final_results = trainer.eval_complete(model, train_dataloader, val_dataloader, test_dataloader, verbose=False)
        train_results, val_results, test_results = final_results
        results.append([train_results[0], train_results[1],
                        val_results[0], val_results[1],
                        test_results[0], test_results[1]])
    return results


def get_model_data(dataset_name):
    if dataset_name == 'musk':
        return [
            (musk_models.MuskEmbeddingSpaceNN, 'models/musk1/MuskEmbeddingSpaceNN/MuskEmbeddingSpaceNN_{:d}.pkl'),
            (musk_models.MuskInstanceSpaceNN, 'models/musk1/MuskInstanceSpaceNN/MuskInstanceSpaceNN_{:d}.pkl'),
            (musk_models.MuskAttentionNN, 'models/musk1/MuskAttentionNN/MuskAttentionNN_{:d}.pkl'),
            (musk_models.MuskGNN, 'models/musk1/MuskGNN/MuskGNN_{:d}.pkl'),
        ]
    if dataset_name in ['tiger', 'elephant', 'fox']:
        return [
            (tef_models.TefEmbeddingSpaceNN,
             'models/' + dataset_name + '/TefEmbeddingSpaceNN/TefEmbeddingSpaceNN_{:d}.pkl'),
            (tef_models.TefInstanceSpaceNN,
             'models/' + dataset_name + '/TefInstanceSpaceNN/TefInstanceSpaceNN_{:d}.pkl'),
            (tef_models.TefAttentionNN,
             'models/' + dataset_name + '/TefAttentionNN/TefAttentionNN_{:d}.pkl'),
            (tef_models.TefGNN,
             'models/' + dataset_name + '/TefGNN/TefGNN_{:d}.pkl'),
        ]
    raise ValueError('Invalid dataset name: {:s}'.format(dataset_name))


def get_trainer(device, dataset_name, model_clz):
    if dataset_name == 'musk':
        trainer_clz = MuskGNNTrainer if model_clz == musk_models.MuskGNN else MuskNetTrainer
        return trainer_clz(device, {}, model_clz)
    elif dataset_name in ['tiger', 'elephant', 'fox']:
        trainer_clz = TefGNNTrainer if model_clz == tef_models.TefGNN else TefNetTrainer
        return trainer_clz(device, {}, model_clz, dataset_name)
    raise ValueError('Invalid dataset name: {:s}'.format(dataset_name))


def load_model(device, model_clz, path):
    model = model_clz(device)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def load_datasets(dataset_name, seed=None):
    if dataset_name == 'musk':
        train_dataset, val_dataset, test_dataset = musk_dataset.create_datasets(musk_two=False, random_state=seed)
    elif dataset_name in ['tiger', 'elephant', 'fox']:
        train_dataset, val_dataset, test_dataset = tef_dataset.create_datasets(dataset_name, random_state=seed)
    else:
        raise ValueError('Invalid dataset name: {:s}'.format(dataset_name))
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)
    return train_dataloader, val_dataloader, test_dataloader


def output_results(results):
    results = np.asarray(results)
    rows = [['Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss', 'Test Accuracy', 'Test Loss']]
    results_row = []
    for i in range(6):
        values = results[:, i]
        mean = np.mean(values)
        sem = np.std(values) / np.sqrt(len(values))
        results_row.append('{:.4f} +- {:.4f}'.format(mean, sem))
    rows.append(results_row)
    table = Texttable()
    table.set_cols_dtype(['t'] * 6)
    table.set_cols_align(['c'] * 6)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())


if __name__ == "__main__":
    run("fox")
