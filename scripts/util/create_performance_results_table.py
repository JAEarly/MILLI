import csv
from texttable import Texttable
import latextable
import numpy as np


def parse_line(line):
    clean_line = []
    for l in line:
        l = l.strip()
        if '+-' in l:
            f1, f2 = [float(s.strip()) for s in l.split('+-')]
            clean_line.append('{:.3f} $\pm$ {:.3f}'.format(f1, f2))
        else:
            clean_line.append(l)
    return clean_line


def output_table(performance_table):
    clean_lines = []
    for line in performance_table:
        clean_line = parse_line(line)
        clean_lines.append(clean_line)

    n_rows = len(clean_lines)
    n_cols = len(clean_lines[0])

    for c in range(1, n_cols):
        scores = []
        for r in range(1, n_rows):
            l = clean_lines[r][c]
            if l != 'N/A':
                score = l.split(" $\\pm$ ")[0]
                scores.append(float(score))
            else:
                scores.append(0)
        if c % 2 == 1:
            top_idxs = np.argwhere(scores == np.max(scores)).squeeze(axis=1)
        else:
            top_idxs = np.argwhere(scores == np.min(scores)).squeeze(axis=1)
        for idx in top_idxs:
            clean_lines[idx + 1][c] = "\\textbf{" + clean_lines[idx + 1][c] + "}"

    n_cols = len(clean_lines[0])

    table = Texttable()
    table.set_cols_dtype(['t']*n_cols)
    table.set_cols_align(['c']*n_cols)
    table.add_rows(clean_lines)
    table.set_max_width(0)
    print(table.draw())

    print(latextable.draw_latex(table, use_booktabs=True, drop_columns=['Train Loss', 'Val Loss', 'Test Loss']))


def parse_table(file):
    table = []
    with open(file) as f:
        reader = csv.reader(f, delimiter='|')
        for line in reader:
            if line and not line[0].startswith('+') and len(line) > 2:
                table.append(line[1:-1])
    return table


def create_results_table(performance_file):
    performance_table = parse_table(performance_file)
    output_table(performance_table)


if __name__ == "__main__":
    create_results_table('results/crc/CRC_Performance')
