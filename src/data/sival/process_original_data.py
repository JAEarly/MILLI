"""
Original data is distributed across multiple files, with duplicate information for each instance.
This script compiles the multiple data files down to a single file.
    One row per instance.
    Structure: bag_name, instance_id, features, instance_label
"""

import csv
import os

root_dir = "data/SIVAL/data"
output_file = "data/SIVAL/processed.data"


def run():
    data = []
    print('Parsing data')
    for file in os.listdir(root_dir):
        clz_name = file[:-5]
        clz_data = parse_file(root_dir + "/" + file, clz_name)
        data.extend(clz_data)
    print('Writing to single file')
    with open(output_file, 'w+', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


def parse_file(path, clz_name):
    clz_data = []
    with open(path) as f:
        r = csv.reader(f)
        for line in r:
            name = line[0]
            if clz_name in name.lower():
                clz_data.append(line)
    return clz_data


if __name__ == "__main__":
    run()
