import copy
import os
from abc import ABC, abstractmethod

import latextable
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from texttable import Texttable
from torch import nn
from tqdm import tqdm


class Trainer(ABC):

    def __init__(self, device, n_classes, save_dir, train_params):
        self.device = device
        self.n_classes = n_classes
        self.save_dir = save_dir
        self.train_params = train_params
        self.update_train_params(self.get_default_train_params())

    @abstractmethod
    def create_optimizer(self, model):
        pass

    @abstractmethod
    def get_criterion(self):
        pass

    @abstractmethod
    def load_datasets(self, seed=None):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_default_train_params(self):
        pass

    def get_train_param(self, key):
        return self.train_params[key]

    def update_train_params(self, new_params, override=False):
        for key, value in new_params.items():
            if override or key not in self.train_params:
                self.train_params[key] = value

    def train_epoch(self, model, optimizer, criterion, train_dataloader, val_dataloader):
        model.train()
        epoch_train_loss = 0
        for data in train_dataloader:
            bags, targets = data[0], data[1].to(self.device)
            optimizer.zero_grad()
            outputs = model(bags)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_dataloader)
        epoch_train_loss = epoch_train_loss
        epoch_val_metrics = self.eval_model(model, val_dataloader)
        return epoch_train_loss, epoch_val_metrics

    def train_model(self, model, train_dataloader, val_dataloader, trial=None):
        # Override current parameters with model suggested parameters
        self.update_train_params(model.suggest_train_params(), override=True)

        model.to(self.device)
        model.train()

        optimizer = self.create_optimizer(model)
        criterion = self.get_criterion()

        early_stopped = False

        n_epochs = self.get_train_param('n_epochs')
        patience = self.get_train_param('patience')

        train_losses = []
        val_metrics = []

        best_model = None
        best_val_loss = float("inf")

        with tqdm(total=n_epochs, desc='Training model', leave=False) as t:
            for epoch in range(n_epochs):
                # Train model for an epoch
                epoch_outputs = self.train_epoch(model, optimizer, criterion, train_dataloader, val_dataloader)
                epoch_train_loss, epoch_val_metrics = epoch_outputs

                # Early stopping
                if patience is not None:
                    if epoch_val_metrics[1] < best_val_loss:
                        best_val_loss = epoch_val_metrics[1]
                        best_model = copy.deepcopy(model)
                        patience_tracker = 0
                    else:
                        patience_tracker += 1
                        if patience_tracker == patience:
                            early_stopped = True
                            break
                else:
                    best_model = copy.deepcopy(model)

                # Update progress bar
                train_losses.append(epoch_train_loss)
                val_metrics.append(epoch_val_metrics)
                t.set_postfix(train_loss=epoch_train_loss, val_loss=epoch_val_metrics[1])
                t.update()

                # Update Optuna
                if trial is not None:
                    trial.report(epoch_val_metrics[0], epoch)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        return best_model, train_losses, val_metrics, early_stopped

    def train_single(self, seed=5, save_model=True, show_plot=True, verbose=True, trial=None):
        train_dataloader, val_dataloader, test_dataloader = self.load_datasets(seed=seed)
        model = self.create_model()
        train_outputs = self.train_model(model, train_dataloader, val_dataloader, trial=trial)
        del model
        best_model, train_losses, val_metrics, early_stopped = train_outputs
        train_results, val_results, test_results = self.eval_complete(best_model, train_dataloader, val_dataloader,
                                                                      test_dataloader, verbose=verbose)

        if save_model:
            save_path = '{:s}/{:s}.pkl'.format(self.save_dir, self.get_model_name())
            print('Saving model to {:s}'.format(save_path))
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            torch.save(best_model.state_dict(), save_path)

        if show_plot:
            self.plot_training(train_losses, val_metrics)

        return best_model, train_results, val_results, test_results, early_stopped

    def train_multiple(self, num_repeats=10, seed=5):
        print('Training model with repeats')
        np.random.seed(seed=seed)

        model_save_dir = '{:s}/{:s}'.format(self.save_dir, self.get_model_name())
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # Train multiple models
        results = []
        for i in range(num_repeats):
            print('Repeat {:d}/{:d}'.format(i + 1, num_repeats))
            repeat_seed = np.random.randint(low=1, high=1000)
            print('Seed: {:d}'.format(repeat_seed))
            train_dataloader, val_dataloader, test_dataloader = self.load_datasets(seed=repeat_seed)
            model = self.create_model()
            best_model, _, _, _ = self.train_model(model, train_dataloader, val_dataloader)
            del model
            final_results = self.eval_complete(best_model, train_dataloader, val_dataloader, test_dataloader,
                                               verbose=False)
            train_results, val_results, test_results = final_results
            results.append([train_results[0], train_results[1],
                            val_results[0], val_results[1],
                            test_results[0], test_results[1]])
            torch.save(best_model.state_dict(), '{:s}/{:s}_{:d}.pkl'.format(model_save_dir, self.get_model_name(), i))

        # Output results in table
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
        print(latextable.draw_latex(table))
        print('Done!')

    def eval_complete(self, model, train_dataloader, val_dataloader, test_dataloader, verbose=False):
        if verbose:
            print('\n-- Train Results --')
        train_results = self.eval_model(model, train_dataloader, verbose=verbose)
        if verbose:
            print('\n-- Val Results --')
        val_results = self.eval_model(model, val_dataloader, verbose=verbose)
        if verbose:
            print('\n-- Test Results --')
        test_results = self.eval_model(model, test_dataloader, verbose=verbose)
        return train_results, val_results, test_results

    def eval_model(self, model, dataloader, verbose=False):
        model.eval()
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            labels = list(range(self.n_classes))
            all_probas = []
            all_targets = []
            for data in dataloader:
                bags, targets = data[0], data[1]
                bag_probas = model(bags)
                all_probas.append(bag_probas.detach().cpu())
                all_targets.append(targets.detach().cpu())
            all_targets = torch.cat(all_targets).long()
            all_probas = torch.cat(all_probas)
            _, all_preds = torch.max(F.softmax(all_probas, dim=1), dim=1)
            acc = accuracy_score(all_targets, all_preds)
            loss = criterion(all_probas, all_targets).item()
            if verbose:
                conf_mat = pd.DataFrame(
                    confusion_matrix(all_targets, all_preds, labels=labels),
                    index=pd.Index(labels, name='Actual'),
                    columns=pd.Index(labels, name='Predicted')
                )
                print(' Acc: {:.3f}'.format(acc))
                print('Loss: {:.3f}'.format(loss))
                print(conf_mat)
        return acc, loss

    def plot_training(self, train_losses, val_metrics):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        x_range = range(len(train_losses))
        axes[0].plot(x_range, train_losses, label='Train')
        axes[0].plot(x_range, [m[1] for m in val_metrics], label='Validation')
        axes[0].set_xlim(0, len(x_range))
        axes[0].set_ylim(min(min(train_losses), min([m[1] for m in val_metrics])) * 0.95,
                         max(max(train_losses), max([m[1] for m in val_metrics])) * 1.05)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(x_range, [m[0] for m in val_metrics], label='Validation')
        axes[1].set_xlim(0, len(x_range))
        axes[1].set_ylim(min([m[0] for m in val_metrics]) * 0.95, max([m[0] for m in val_metrics]) * 1.05)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(loc='best')
        plt.show()
