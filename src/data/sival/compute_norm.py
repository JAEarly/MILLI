import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


from data.sival.sival_dataset import parse_data_from_file

from data.bach_dataset import load_bach_bags

_, bags, _, _ = parse_data_from_file()
print(bags[0].shape)

bags = torch.cat(bags)

print(bags.shape)

arrs_mean = torch.mean(bags, dim=0)
arrs_std = torch.std(bags, dim=0)

print(arrs_mean)
print(arrs_std)
