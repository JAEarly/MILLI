import torch

from model import tef_models
from train.tef_training import TefNetTrainer, TefGNNTrainer


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clz = tef_models.TefInstanceSpaceNN

    dataset_names = ["tiger", "elephant", "fox"]

    for dataset_name in dataset_names:
        print("Running training on {:s} dataset".format(dataset_name))
        trainer_clz = TefGNNTrainer if model_clz == tef_models.TefGNN else TefNetTrainer
        trainer = trainer_clz(device, {}, model_clz, dataset_name)

        print('Using model {:}'.format(model_clz.__name__))
        print('Using device {:}'.format(device))
        trainer.train_multiple()
