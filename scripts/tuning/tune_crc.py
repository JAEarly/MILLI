import torch

from model import crc_models
from tuning import crc_tuning
from tuning.tune_util import setup_study

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clz = crc_models.CrcEmbeddingSpaceNN
    tuner_clz = crc_tuning.get_tuner(model_clz)

    print('Tuning model {:}'.format(model_clz.__name__))
    print('Using device {:}'.format(device))
    study = setup_study("Optimise-{:s}".format(model_clz.__name__))
    tuner = tuner_clz(device)
    study.optimize(tuner, n_trials=100)
