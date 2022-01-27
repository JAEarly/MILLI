import torch

from model import mnist_models
from train.mnist_training import MnistNetTrainer, MnistGNNTrainer


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clz = mnist_models.MnistGNN
    trainer_clz = MnistGNNTrainer if model_clz == mnist_models.MnistGNN else MnistNetTrainer
    trainer = trainer_clz(device, {}, model_clz)

    print('Using model {:}'.format(model_clz.__name__))
    print('Using device {:}'.format(device))
    trainer.train_multiple()
