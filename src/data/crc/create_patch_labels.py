import csv

import scipy.io as sio

from data.crc.crc_dataset import orig_path, cell_types


def extract_instance_labels():
    patch_size = 27
    out_dir = "data/CRC/patch_{:d}/instance_labels".format(patch_size)
    for i in range(100):
        img_id = i + 1
        locs_dict = {}
        for cell_type in cell_types:
            locs = parse_file(img_id, cell_type)
            locs_dict[cell_type] = locs

        n_x = int(500 / patch_size)
        n_y = int(500 / patch_size)

        rows = [['x', 'y', 'labels']]
        for i_x in range(n_x):
            for i_y in range(n_y):
                p_x = i_x * patch_size
                p_y = i_y * patch_size
                found_cell_types = get_cell_types_in_patch(locs_dict, p_x, p_y, p_x+patch_size, p_y+patch_size)
                if found_cell_types:
                    rows.append([i_x, i_y, *found_cell_types])

        csv_path = '{:s}/img{:d}_instance_labels.csv'.format(out_dir, img_id)
        with open(csv_path, newline='', mode='w+') as f:
            w = csv.writer(f)
            w.writerows(rows)


def parse_file(img_id, cell_type):
    path = '{:s}/img{:d}/img{:d}_{:s}.mat'.format(orig_path, img_id, img_id, cell_type)
    mat = sio.loadmat(path)
    return mat['detection']


def get_cell_types_in_patch(locs_dict, p_x, p_y, p_x2, p_y2):
    found_cell_types = []
    for cell_type in cell_types:
        locs = locs_dict[cell_type]
        for y, x in locs:
            if p_x <= x <= p_x2 and p_y <= y <= p_y2:
                found_cell_types.append(cell_type)
    return list(set(found_cell_types))


if __name__ == "__main__":
    extract_instance_labels()
