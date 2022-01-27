import csv

import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from data.mil_dataset import MilDataset

raw_dir = "data/SIVAL/raw"
input_file = "data/SIVAL/processed.data"

all_clzs = ['ajaxorange', 'apple', 'banana', 'bluescrunge', 'candlewithholder', 'cardboardbox', 'checkeredscarf',
            'cokecan', 'dataminingbook', 'dirtyrunningshoe', 'dirtyworkgloves', 'fabricsoftenerbox', 'feltflowerrug',
            'glazedwoodpot', 'goldmedal', 'greenteabox', 'juliespot', 'largespoon', 'rapbook', 'smileyfacedoll',
            'spritecan', 'stripednotebook', 'translucentbowl', 'wd40can', 'woodrollingpin']
positive_clzs = ['apple', 'banana', 'checkeredscarf', 'cokecan', 'dataminingbook', 'goldmedal',
                 'largespoon', 'rapbook', 'smileyfacedoll', 'spritecan', 'translucentbowl', 'wd40can']

dataset_mean = torch.tensor([120.8379, 118.1655, 107.9966,   2.6262,   2.2394,   2.3817,  -2.7394,
                             -2.3569,  -3.9129,   0.4166,   0.4347,   0.3775,  -3.7723,  -3.2548,
                             -3.4267,   0.5095,   0.3329,   0.3227,  -3.8466,  -2.8435,  -2.8118,
                             0.3854,   0.4561,   0.3311,  -1.3757,  -1.0629,  -1.8387,   0.5073,
                             0.3384,   0.3398])

dataset_std = torch.tensor([52.4034, 49.3811, 51.9998,  2.0022,  1.8713,  1.9941, 58.1256, 54.4538,
                            53.9269,  1.7914,  1.5520,  1.4908, 56.0164, 52.8565, 52.5657,  1.6778,
                            1.6513,  1.5286, 57.2126, 53.9396, 54.1670,  1.8263,  1.5622,  1.5250,
                            57.0688, 53.7696, 53.3720,  1.6834,  1.6374,  1.5166])

SIVAL_N_CLASSES = len(positive_clzs) + 1
SIVAL_N_EXPECTED_DIMS = 2  # i * f
SIVAL_D_IN = 30


def clz_to_idx(clz_name):
    try:
        idx = all_clzs.index(clz_name.lower())
    except ValueError:
        raise ValueError('Invalid class name: {:s}'.format(clz_name)) from None
    return idx


def idx_to_clz(idx):
    try:
        clz = all_clzs[idx]
    except IndexError:
        raise ValueError('Invalid class idx: {:d}'.format(idx)) from None
    return clz


def create_full_dataset():
    parsed_data = parse_data_from_file()
    bag_names, bags, targets, instance_labels = parsed_data
    dataset = SIVALDataset(bag_names, bags, targets, instance_labels)
    return dataset


def create_datasets(random_state=12):
    parsed_data = parse_data_from_file()
    bag_names, bags, original_targets, instance_labels = parsed_data

    targets, instance_labels, selected_idxs = _convert_to_pos_neg_split(original_targets, instance_labels)
    bag_names = [bag_names[i] for i in selected_idxs]
    bags = [bags[i] for i in selected_idxs]
    original_targets = [original_targets[i] for i in selected_idxs]

    splits = train_test_split(bag_names, bags, targets, instance_labels, original_targets,
                              train_size=0.8, stratify=targets, random_state=random_state)

    train_bag_names, train_bags, train_targets, train_ils, train_orig_targets = [splits[i] for i in [0, 2, 4, 6, 8]]
    test_bag_names, test_bags, test_targets, test_ils, test_orig_targets = [splits[i] for i in [1, 3, 5, 7, 9]]

    splits = train_test_split(test_bag_names, test_bags, test_targets, test_ils, test_orig_targets,
                              train_size=0.5, stratify=test_targets, random_state=random_state)

    val_bag_names, val_bags, val_targets, val_ils, val_orig_targets = [splits[i] for i in [0, 2, 4, 6, 8]]
    test_bag_names, test_bags, test_targets, test_ils, test_orig_targets = [splits[i] for i in [1, 3, 5, 7, 9]]

    train_dataset = SIVALDataset(train_bag_names, train_bags, train_targets, train_ils, train_orig_targets)
    val_dataset = SIVALDataset(val_bag_names, val_bags, val_targets, val_ils, val_orig_targets)
    test_dataset = SIVALDataset(test_bag_names, test_bags, test_targets, test_ils, test_orig_targets)

    return train_dataset, val_dataset, test_dataset


def _convert_to_pos_neg_split(targets, instance_labels):
    new_targets = []
    new_instance_labels = []
    clz_counter = {}
    selected_idxs = []
    for idx, target_idx in enumerate(targets):
        target_clz = idx_to_clz(target_idx)
        ils = instance_labels[idx]
        if target_clz not in positive_clzs:
            if target_clz not in clz_counter:
                clz_counter[target_clz] = 0
            # Limit to 30 images from each negative class
            if clz_counter[target_clz] < 30:
                new_targets.append(0)
                new_instance_labels.append(torch.zeros_like(ils))
                clz_counter[target_clz] += 1
                selected_idxs.append(idx)
        else:
            new_target = positive_clzs.index(target_clz.lower()) + 1
            new_ils = ils * new_target
            new_targets.append(new_target)
            new_instance_labels.append(new_ils)
            selected_idxs.append(idx)
    return new_targets, new_instance_labels, selected_idxs


def parse_data_from_file():
    bag_data = {}
    bag_targets = {}
    bag_instance_labels = {}
    with open(input_file) as f:
        reader = csv.reader(f)
        for line in reader:
            bag_name, features, instance_label = parse_line(line)
            if bag_name not in bag_data:
                bag_data[bag_name] = []
                bag_instance_labels[bag_name] = []
                clz_name = bag_name[bag_name.index('_')+1:bag_name.rindex('_')]
                bag_targets[bag_name] = clz_to_idx(clz_name)
            bag_data[bag_name].append(features)
            bag_instance_labels[bag_name].append(instance_label)

    bag_names = []
    bags = []
    targets = []
    instance_labels = []

    for bag_name in bag_data.keys():
        bag_names.append(bag_name)

        bag = bag_data[bag_name]
        bag = torch.stack(bag)
        bags.append(bag)

        target = bag_targets[bag_name]
        targets.append(target)

        i_labels = bag_instance_labels[bag_name]
        i_labels = torch.as_tensor(i_labels).float()
        instance_labels.append(i_labels)

    return bag_names, bags, targets, instance_labels


def parse_line(line):
    bag_name, features, instance_label = line[0], line[2:-1], float(line[-1])
    features = torch.as_tensor([float(f) for f in features])
    features = (features - dataset_mean) / dataset_std
    return bag_name, features, instance_label


class SIVALDataset(MilDataset):

    def __init__(self, bag_names, bags, targets, instance_targets, original_targets):
        super().__init__(bags, targets, instance_targets)
        self.bag_names = bag_names
        self.original_targets = original_targets

    def get_bag_from_name(self, bag_name):
        try:
            idx = self.bag_names.index(bag_name)
            bag = self.bags[idx]
            target = self.targets[idx]
            instance_targets = self.instance_targets[idx]
            return bag, target, instance_targets
        except ValueError:
            raise ValueError('Invalid bag name: {:s}'.format(bag_name)) from None

    def get_path_from_name(self, root, bag_name, extension):
        try:
            idx = self.bag_names.index(bag_name)
            target = self.original_targets[idx]
            clz_name = all_clzs[target]
            path = root + "/" + clz_name + "/" + bag_name + extension
            return path
        except ValueError:
            raise ValueError('Invalid bag name: {:s}'.format(bag_name)) from None

    def get_img_from_name(self, bag_name):
        path = self.get_path_from_name(raw_dir, bag_name, ".jpg")
        img = Image.open(path)
        return img


if __name__ == "__main__":
    parse_data_from_file()
