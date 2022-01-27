import csv
from texttable import Texttable
import latextable
import numpy as np


def parse_line(line):
    clean_line = []
    max_sem = None
    for l in line:
        l = l.strip()
        if '+-' in l:
            mean, sem = [float(s.strip()) for s in l.split('+-')]
            # clean_line.append('{:.3f} $\pm$ {:.3f}'.format(f1, f2))
            clean_line.append('{:.3f}'.format(mean))
            if max_sem is None or sem > max_sem:
                max_sem = sem
        else:
            clean_line.append(l)
    return clean_line, max_sem


def clean_table(table):
    clean_lines = []
    max_sem = None
    for line in table:
        clean_line, line_max_sem = parse_line(line)
        if 'Weighted' in clean_line[0]:
            continue
        clean_lines.append(clean_line)
        if max_sem is None or line_max_sem > max_sem:
            max_sem = line_max_sem

    for c in range(1, len(clean_lines[0])):
        scores = np.zeros(len(clean_lines) - 1)
        for r in range(1, len(clean_lines)):
            l = clean_lines[r][c]
            if l != 'N/A':
                score = l.split(" $\\pm$ ")[0]
                scores[r - 1] = float(score)
        max_idxs = np.argwhere((scores == np.max(scores)) & (scores != 0)).squeeze(axis=1)
        for idx in max_idxs:
            clean_lines[idx + 1][c] = "\\textbf{" + clean_lines[idx + 1][c] + "}"

    return clean_lines, max_sem


def output_table(interpretability_table_1, interpretability_table_2, performance_table):
    table_1_clean_lines, table_1_max_sem = clean_table(interpretability_table_1)
    if interpretability_table_2 is not None:
        table_2_clean_lines, table_2_max_sem = clean_table(interpretability_table_2)
        clean_lines = []
        for table_1_line in table_1_clean_lines:
            if table_1_line[0] == 'Methods':
                clean_lines.append(table_1_line)
            else:
                match_found = False
                for table_2_line in table_2_clean_lines:
                    if table_1_line[0] == table_2_line[0]:
                        match_found = True
                        line = [table_1_line[0]]
                        for i in range(1, len(table_1_line)):
                            line.append("{:s} / {:s}".format(table_1_line[i], table_2_line[i]))
                        print(line)
                        clean_lines.append(line)
                        break
                if not match_found:
                    raise ValueError("No matching entry in table two for entry '{:s}' in table one"
                                     .format(table_1_line[0]))
        max_sem = max(table_1_max_sem, table_2_max_sem)
    else:
        clean_lines = table_1_clean_lines
        max_sem = table_1_max_sem

    scores = []
    if performance_table is not None:
        performance_line = []
        for r in range(1, 5):
            model_performance = performance_table[r][5]
            score = model_performance.split(" +- ")[0]
            performance_line.append(model_performance)
            scores.append(float(score))
        overall_performance = str(np.mean(scores)) + ' +- ' + str(np.std(scores) / np.sqrt(len(scores)))
        performance_line += [overall_performance]
        clean_performance_line, _ = parse_line(performance_line)
        clean_performance_line.insert(0, 'Model Acc')
        clean_lines.insert(1, clean_performance_line)

    n_cols = len(clean_lines[0])

    table = Texttable()
    table.set_cols_dtype(['t']*n_cols)
    table.set_cols_align(['c']*n_cols)
    table.add_rows(clean_lines)
    table.set_max_width(0)
    print(table.draw())

    print('\nMax Sem: {:.3f}\n'.format(max_sem))

    print(latextable.draw_latex(table, use_booktabs=True))


def parse_table(file):
    table = []
    with open(file) as f:
        reader = csv.reader(f, delimiter='|')
        for line in reader:
            if line and not line[0].startswith('+') and len(line) > 2:
                table.append(line[1:-1])
    return table


def create_results_table(interpretability_file_1, interpretability_file_2, performance_file):
    interpretability_table_1 = parse_table(interpretability_file_1)
    interpretability_table_2 = parse_table(interpretability_file_2) if interpretability_file_2 is not None else None
    performance_table = parse_table(performance_file) if performance_file is not None else None

    output_table(interpretability_table_1, interpretability_table_2, performance_table)


if __name__ == "__main__":
    create_results_table('results/crc/crc_Interpretability_NDCG.txt',
                         None,
                         'results/crc/crc_Performance.txt')
