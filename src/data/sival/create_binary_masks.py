import csv
import os

import numpy as np
from PIL import Image

from data.sival.sival_dataset import create_full_dataset
from matplotlib import pyplot as plt

path = "data/SIVAL/bags"
outpath = "data/SIVAL/masks"


def parse_file(folder, file_name, dataset):
    bag_name = file_name[:file_name.index('.')]
    file_path = path + "/" + folder + "/" + file_name
    try:
        _, _, instance_targets = dataset.get_bag_from_name(bag_name)
    except ValueError:
        return

    output_dir = outpath + "/" + folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + "/" + bag_name + "_mask.png"

    if os.path.exists(output_file):
        return

    instance_idx = 0
    parse_blocks = False
    image_arr = np.zeros((256, 192))
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=' ')
        for line in reader:
            if parse_blocks:
                if len(line) == 0:
                    parse_blocks = False
                else:
                    colour = instance_targets[instance_idx]
                    for b in line:
                        try:
                            b = int(b)
                            x = b % 256
                            y = b // 256
                            image_arr[x, y] = colour * 255
                        except:
                            pass
                    instance_idx += 1
            if len(line) > 0 and line[0] == 'Blocks:':
                parse_blocks = True
            if parse_blocks and len(line) == 0:
                parse_blocks = False

    image_arr = image_arr.swapaxes(0, 1)
    im = Image.fromarray(np.uint8(image_arr), 'L')


    im.save(output_file)


def run():
    dataset = create_full_dataset()
    folders = os.listdir(path)
    for folder in folders:
        for file in os.listdir(path + "/" + folder):
            if file.endswith(".imbag"):
                parse_file(folder, file, dataset)


if __name__ == "__main__":
    run()
