import torch

from model import crc_models
from train.crc_training import CrcNetTrainer, CrcGNNTrainer


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clz = crc_models.CrcAttentionNN
    trainer_clz = CrcGNNTrainer if model_clz == crc_models.CrcGNN else CrcNetTrainer
    trainer = trainer_clz(device, {}, model_clz)

    print('Using model {:}'.format(model_clz.__name__))
    print('Using device {:}'.format(device))
    trainer.train_multiple()
