import torch

from model import mnist_models
from tuning import mnist_tuning
from tuning.tune_util import setup_study

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clz = mnist_models.MnistGNN
    tuner_clz = mnist_tuning.get_tuner(model_clz)

    print('Tuning model {:}'.format(model_clz.__name__))
    print('Using device {:}'.format(device))
    study = setup_study("Optimise-{:s}".format(model_clz.__name__))
    tuner = tuner_clz(device)
    study.optimize(tuner, n_trials=100)
