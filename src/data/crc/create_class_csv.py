import csv
from collections import Counter

import scipy.io as sio

from data.crc.crc_dataset import cell_types, csv_path, orig_path


def run():
    img_binary_clz = {}
    img_tertiary_clz = {}
    data_out = [['Image ID', 'Binary Class', 'Tertiary Class']]
    for i in range(100):
        img_id = i + 1
        cell_counts = []
        for cell_type in cell_types:
            locs = parse_file(img_id, cell_type)
            cell_counts.append(len(locs))
        binary_clz = get_binary_clz(cell_counts)
        tertiary_clz = get_tertiary_clz(cell_counts)
        img_binary_clz[img_id] = binary_clz
        img_tertiary_clz[img_id] = tertiary_clz
        data_out.append([img_id, binary_clz, tertiary_clz])

        if binary_clz == 1:
            assert tertiary_clz == 2
        elif tertiary_clz == 1:
            assert binary_clz == 0

    binary_clz_counts = Counter(img_binary_clz.values())
    tertiary_clz_counts = Counter(img_tertiary_clz.values())

    print('Binary class counts: {:}'.format(binary_clz_counts))
    print('Tertiary class counts: {:}'.format(tertiary_clz_counts))

    with open(csv_path, 'w+', newline="") as f:
        writer = csv.writer(f)
        for line in data_out:
            writer.writerow(line)


def parse_file(img_id, cell_type):
    path = '{:s}/img{:d}/img{:d}_{:s}.mat'.format(orig_path, img_id, img_id, cell_type)
    mat = sio.loadmat(path)
    return mat['detection']


def get_binary_clz(cell_counts):
    if cell_counts[3] > 0:
        return 1
    return 0


def get_tertiary_clz(cell_counts):
    if cell_counts[3] > 0:
        return 2
    if cell_counts[2] / (cell_counts[1] + cell_counts[0]) >= 0.7:
        return 1
    return 0


if __name__ == "__main__":
    run()
