from abc import ABC, abstractmethod


class Tuner(ABC):

    def __init__(self, device):
        self.device = device

    @abstractmethod
    def generate_train_params(self, trial):
        pass

    @abstractmethod
    def generate_model_params(self, trial):
        pass

    @abstractmethod
    def create_trainer(self, train_params, model_params):
        pass

    @staticmethod
    def suggest_layers(trial, layer_name, param_name, min_n_layers, max_n_layers, layer_options):
        n_layers = trial.suggest_int('n_{:s}'.format(layer_name), min_n_layers, max_n_layers)
        layers_values = []
        for i in range(n_layers):
            layers_values.append(trial.suggest_categorical('{:s}_{:d}'.format(param_name, i), layer_options))
        return layers_values

    def __call__(self, trial):
        train_params = self.generate_train_params(trial)
        model_params = self.generate_model_params(trial)
        trainer = self.create_trainer(train_params, model_params)
        model, _, _, test_results, early_stopped = trainer.train_single(save_model=False, show_plot=False,
                                                                        verbose=False, trial=trial)
        test_acc = test_results[0]
        trial.set_user_attr('early_stopped', early_stopped)
        return test_acc
