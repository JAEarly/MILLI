import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from data.crc.crc_dataset import load_crc_bags

bags, _, _ = load_crc_bags()

avgs = []
transformation = transforms.ToTensor()


for bag in tqdm(bags):
    for file_name in bag:
        with open(file_name, 'rb') as f:
            img = Image.open(f)
            instance = img.convert('RGB')
            avg = torch.mean(transformation(instance), dim=(1, 2))
            avgs.append(avg)

arrs = torch.stack(avgs)
print(arrs.shape)

arrs_mean = torch.mean(arrs, dim=0)
arrs_std = torch.std(arrs, dim=0)

print(arrs_mean)
print(arrs_std)
