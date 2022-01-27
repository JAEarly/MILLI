import csv
import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from data.mil_dataset import MilDataset

cell_types = ['others', 'inflammatory', 'fibroblast', 'epithelial']
binary_clz_names = ['non-epithelial', 'epithelial']

orig_path = 'data/CRC/orig'
raw_path = 'data/CRC/raw'
csv_path = 'data/CRC/crc_classes.csv'

CRC_N_CLASSES = 2
CRC_FV_SIZE = 1200
CRC_N_EXPECTED_DIMS = 4  # i x c x h x w


class Rotate90:

    def __call__(self, x):
        angle = random.choice([0, 90, 190, 270])
        return TF.rotate(x, angle)


basic_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.8035, 0.6499, 0.8348), (0.0858, 0.1079, 0.0731))])

augmentation_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),
                                             Rotate90(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.8035, 0.6499, 0.8348), (0.0858, 0.1079, 0.0731))])


class CRCDataset(MilDataset):

    def __init__(self, bags, targets, ids, transform, instance_labels):
        super().__init__(bags, targets, instance_labels)
        self.transform = transform
        self.ids = ids

    def __getitem__(self, index):
        instances = self._load_instances(index)
        target = self.targets[index]
        return instances, target

    def get_bag_verbose(self, index):
        instances = self._load_instances(index)
        target = self.targets[index]
        instance_targets = self.instance_targets[index]
        return instances, target, instance_targets

    def _load_instances(self, bag_idx):
        instances = []
        bag = self.bags[bag_idx]
        for file_name in bag:
            with open(file_name, 'rb') as f:
                img = Image.open(f)
                instance = img.convert('RGB')
                if self.transform is not None:
                    instance = self.transform(instance)
                instances.append(instance)
        instances = torch.stack(instances)
        return instances


def load_crc_classes():
    binary_clzs = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_id, binary_clz, _ = [int(i) for i in row]
            binary_clzs[image_id] = binary_clz
    return binary_clzs


def load_crc_bags(patch_size=27, verbose=False):
    binary_clzs = load_crc_classes()
    if verbose:
        print('Loading CRC data')

    label_dict = binary_clzs
    clz_names = binary_clz_names

    bags = []
    targets = []
    ids = []

    n_r = int(500 / patch_size)
    n_c = int(500 / patch_size)

    for img_id, clz in label_dict.items():
        clz_name = clz_names[clz]
        patch_dir = 'data/CRC/patch_{:d}/{:s}'.format(patch_size, clz_name)

        bag = []
        for r in range(n_r):
            for c in range(n_c):
                file_name = 'img{:d}_{:d}_{:d}.png'.format(img_id, r, c)
                file_path = '{:s}/{:s}'.format(patch_dir, file_name)
                if os.path.exists(file_path):
                    bag.append(file_path)
        if len(bag) == 0:
            if verbose:
                print('Omitting image {:d} as it has zero foreground patches'.format(img_id))
        else:
            bags.append(bag)
            ids.append(img_id)
            targets.append(clz)

    if verbose:
        print('Loaded {:d} bags'.format(len(bags)))
    return bags, targets, ids


def load_crc_instance_labels(patch_size):
    img_id_to_instance_labels = {}
    for i in range(100):
        img_id = i + 1
        label_csv_path = "data/CRC/patch_{:d}/instance_labels/img{:d}_instance_labels.csv".format(patch_size, img_id)
        bag_instance_labels = {}
        with open(label_csv_path, newline='', mode='r') as f:
            r = csv.reader(f)
            next(r)
            for line in r:
                x = int(line[0])
                y = int(line[1])
                labels = line[2:]
                label_clzs = []
                for label in labels:
                    label_clzs.append(_binary_label_to_id(label))
                label_clzs = list(set(label_clzs))
                bag_instance_labels[(x, y)] = label_clzs
        img_id_to_instance_labels[img_id] = bag_instance_labels
    return img_id_to_instance_labels


def _binary_label_to_id(label):
    if label == 'others' or label == 'inflammatory' or label == 'fibroblast':
        return 0
    if label == 'epithelial':
        return 1
    raise ValueError('Invalid label: {:s}'.format(label))


def _get_instance_targets_for_bags(bags, instance_label_dict):
    all_instance_labels = []
    for bag in bags:
        bag_instance_labels = []
        for file_name in bag:
            info_str = file_name[file_name.rindex('img')+3:-4]
            img_id, x, y = [int(x) for x in info_str.split('_')]
            if (x, y) in instance_label_dict[img_id]:
                instance_labels = instance_label_dict[img_id][(x, y)]
            else:
                instance_labels = []
            bag_instance_labels.append(instance_labels)
        all_instance_labels.append(bag_instance_labels)
    return all_instance_labels


def load_crc(patch_size=27, augment_train=True, random_state=12, verbose=False):
    bags, targets, ids = load_crc_bags(patch_size, verbose=verbose)

    train_bags, test_bags, train_targets, test_targets, train_ids, test_ids = \
        train_test_split(bags, targets, ids, train_size=0.6, stratify=targets, random_state=random_state)
    val_bags, test_bags, val_targets, test_targets, val_ids, test_ids = \
        train_test_split(test_bags, test_targets, test_ids, train_size=0.5, stratify=test_targets,
                         random_state=random_state)

    img_id_to_instance_labels = load_crc_instance_labels(patch_size)
    train_instance_labels = _get_instance_targets_for_bags(train_bags, img_id_to_instance_labels)
    val_instance_labels = _get_instance_targets_for_bags(val_bags, img_id_to_instance_labels)
    test_instance_labels = _get_instance_targets_for_bags(test_bags, img_id_to_instance_labels)

    train_dataset = CRCDataset(train_bags, train_targets, train_ids,
                               augmentation_transform if augment_train else basic_transform, train_instance_labels)
    val_dataset = CRCDataset(val_bags, val_targets, val_ids, basic_transform, val_instance_labels)
    test_dataset = CRCDataset(test_bags, test_targets, test_ids, basic_transform, test_instance_labels)

    if verbose:
        print('\n-- Train dataset --')
        train_dataset.summarise()
        print('\n-- Val dataset --')
        val_dataset.summarise()
        print('\n-- Test dataset --')
        test_dataset.summarise()

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    dataset, _, _ = load_crc()
