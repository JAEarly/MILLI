import torch

from model import sival_models
from train.sival_training import SivalNetTrainer, SivalGNNTrainer


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clz = sival_models.SivalAttentionNN

    trainer_clz = SivalGNNTrainer if model_clz == sival_models.SivalGNN else SivalNetTrainer
    trainer = trainer_clz(device, {}, model_clz)

    print('Using model {:}'.format(model_clz.__name__))
    print('Using device {:}'.format(device))
    trainer.train_single()
