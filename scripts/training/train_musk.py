import torch

from model import musk_models
from train.musk_training import MuskNetTrainer, MuskGNNTrainer


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clz = musk_models.MuskInstanceSpaceNN

    trainer_clz = MuskGNNTrainer if model_clz == musk_models.MuskGNN else MuskNetTrainer
    trainer = trainer_clz(device, {}, model_clz)

    print('Using model {:}'.format(model_clz.__name__))
    print('Using device {:}'.format(device))
    trainer.train_single()
