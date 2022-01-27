import os

import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from data.crc.crc_dataset import load_crc_classes, binary_clz_names, tertiary_clz_names, orig_path, raw_path


def run():
    binary_clzs, tertiary_clzs = load_crc_classes()
    for i in range(100):
        img_id = i + 1
        binary_clz = binary_clzs[img_id]
        tertiary_clz = tertiary_clzs[img_id]
        binary_clz_name = binary_clz_names[binary_clz]
        tertiary_clz_name = tertiary_clz_names[tertiary_clz]
        sort_image(img_id, binary_clz_name)
        sort_image(img_id, tertiary_clz_name)
    extract_grid_patches()


def sort_image(img_id, clz_name):
    img_path = '{:s}/img{:d}/img{:d}.bmp'.format(orig_path, img_id, img_id)
    out_dir = '{:s}/{:s}'.format(raw_path, clz_name, img_id)
    out_path = '{:s}/img{:d}.png'.format(out_dir, img_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_path):
        with Image.open(img_path) as im:
            im.save(out_path)


def extract_grid_patches():
    patch_size = 27
    brightness_threshold = 230

    num_patches = int(500 / patch_size * 500 / patch_size)
    print('{:d} patches per image'.format(num_patches))

    all_clz_names = list(set(binary_clz_names + tertiary_clz_names))

    for clz_name in all_clz_names:
        raw_dir = '{:s}/{:s}'.format(raw_path, clz_name)
        patch_dir = 'data/CRC/patch_{:d}/{:s}'.format(patch_size, clz_name)
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)

        file_names = os.listdir(raw_dir)

        for file_name in tqdm(file_names, desc='Extract patches for {:s}'.format(clz_name)):
            name = file_name[:-4]

            im = Image.open(raw_dir + "/" + file_name)
            im_arr = np.array(im)

            n_x = int(im_arr.shape[0]/patch_size)
            n_y = int(im_arr.shape[1]/patch_size)

            for i_x in range(n_x):
                for i_y in range(n_y):
                    p_x = i_x * patch_size
                    p_y = i_y * patch_size
                    patch_arr = im_arr[p_x:p_x+patch_size, p_y:p_y+patch_size, :]
                    avg_brightness = np.mean(patch_arr)
                    if avg_brightness < brightness_threshold:
                        im = Image.fromarray(patch_arr)
                        im.save(patch_dir + "/{:s}_{:d}_{:d}.png".format(name, i_x, i_y))


if __name__ == "__main__":
    run()
