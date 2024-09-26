import os
from collections import defaultdict
import argparse

import numpy as np
import csv
from matplotlib import pyplot as plt
import matplotlib as mpl

plt.switch_backend('agg')
plt.style.use('seaborn-paper')

# mpl.font_manager.fontManager.addfont(
#         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Helvetica-Regular.ttf'))

# plt.rcParams['font.family'] = 'Helvetica Neue LT'
plt.rcParams['legend.fontsize'] = 12.0
plt.rcParams['axes.labelsize'] = 14.0
plt.rcParams['xtick.labelsize'] = 14.0
plt.rcParams['ytick.labelsize'] = 14.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 200.0

color_pool = ['blue', 'red', 'darkorange', 'purple', 'black', 'cyan', 'lime', 'gold']
def remove_outliers(data):
    """
    Identifies and removes outliers from the data set using the IQR method.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def parse_log(path, fields):
    data = defaultdict(list)
    def _parse_line(line):
        flags = line.split(':')
        if len(flags) >= 2 and any(map(lambda x: flags[-2].endswith(x), fields)):
            k = flags[-2].split('/')[1]
            v = float(flags[-1].strip())
            data[k].append(v)
            
    with open(path) as f:
        for line in f:
            _parse_line(line)

    return data


def parse_energy_log(energy_log_path):
    fields = ['ddG']
    fields_1 = ['dG_wild']
    # fields_1 = ['dG_design']

    data = defaultdict(list)
    new_data = defaultdict(list)
    key = []
    def _parse_line(line):
        flags = line.split(':')
        if len(flags) >= 2 and any(map(lambda x: flags[-2].endswith(x), fields)):
            # print(f"flags: {flags}")
            k = (flags[-2].split('/')[-1]).split('.')[0]
            v = float(flags[-1].strip())
            data[k].append(v)
            # print(f"k: {k}; v: {v}")
            key.append(k)
        if len(flags) >= 2 and any(map(lambda x: flags[-2].endswith(x), fields_1)):
            # print(f"flags: {flags}")
            k = (flags[-2].split('/')[-1]).split('.')[0]
            v = float(flags[-1].strip())
            new_data[k].append(v)
            # print(f"k: {k}; v: {v}")
            # key.append(k)
    with open(energy_log_path) as f:
        for line in f:
            _parse_line(line)

    return data, new_data, key


def get_data(log_dir, rank=None):
    data = []
    data_1 = []
    key = []
    for root, dirs, files in os.walk(log_dir):
    # Check if 'metric.log' is in the files
        for dir in dirs: 
            path = os.path.join(root, dir)
            if 'relax.log' in os.listdir(path):
                energy_log_path = os.path.join(path, 'relax.log')
                # Parse the metric.log file and retrieve relevant information
                energy, energy_1, key_ = parse_energy_log(energy_log_path)
                if len(key) == 0:
                    key = key_
                # Add the parsed metric to the list of all metrics
                data.append(energy)
                data_1.append(energy_1)
    # print(f"data: {data}\nkey: {key}")
    data = dict(zip(key, [[b[k] for b in data] for k in key]))
    data_1 = dict(zip(key, [[b[k] for b in data_1] for k in key]))

    new_data = dict()
    new_data_1 = dict()
    # print(f"new_data: {new_data}")
    for key, value in data.items():
        flattened_list = [item for sublist in value for item in sublist]
        new_data[key] = flattened_list
    for key, value in data_1.items():
        flattened_list = [item for sublist in value for item in sublist]
        new_data_1[key] = flattened_list
    # print(f"data: {new_data}")
    return new_data, new_data_1

def plot_curve( data, data_1, figure_path, agg_factor=1):
    # y = data[data < 50.0]
    # print(f"data: {data}")
    # data = {key: remove_outliers(value) for key, value in data.items()}
    save_dict = dict()
    for key, value in data_1.items():
        # print(f"key: {key}, value: {value[0]}")
        save_dict.update(
            # {key: list(value)}
            {key: [value[0]]}
        )
    csv_file = './ref_SKEMPI_AbX.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=save_dict.keys())
        writer.writeheader()  # 写入CSV文件的表头
        for row in zip(*save_dict.values()):
            writer.writerow(dict(zip(save_dict.keys(), row)))
    proportions = {key: sum(1 for value in values if value < 0) / len(values) for key, values in data.items()}
    overall_proportion = sum(1 for values in data.values() for value in values if value < 0) / sum(len(values) for values in data.values())
    overall_energy = sum(value for values in data.values() for value in values) / sum(len(values) for values in data.values())
    print(f"proportions: {proportions}")
    print(f"overall: {overall_proportion} {overall_energy}")
    fig, ax = plt.subplots()  #figsize=(4, 4))
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.boxplot(data.values(), labels=data.keys(), vert=True, patch_artist=True)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.ylabel("ddG")
    plt.title("Test Data")
    plt.tight_layout()
    plt.show()


    ax.legend(frameon=False, loc='upper right')
    
    ax.set_ylabel('Loss')
    ax.set_xlabel('Steps')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(linestyle='--')

    plt.savefig(figure_path,
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)

def main(args):
    figure_path = os.path.join(args.output_dir, 'energy.pdf')

    energy_data, energy_data_1 = get_data(args.log_dir)
    plot_curve(energy_data, energy_data_1, figure_path )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    main(args)
