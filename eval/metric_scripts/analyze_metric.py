import numpy as np
import os
from collections import defaultdict
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import pdb
import seaborn as sns
from functools import reduce

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

def parse_log(path, fields):
    data = defaultdict(list)
    log_data = pd.read_csv(path, sep='\t')
    for field in fields:
        data[field].append(log_data[['name', field]])
    return data

def get_data(log_dir, fields, rank=None):
    data = []
    for root, dirs, files in os.walk(log_dir):
    # Check if 'metric.log' is in the files
        for dir in dirs: 
            path = os.path.join(root, dir)
            if 'metric.csv' in os.listdir(path):
                metric_path = os.path.join(path, 'metric.csv')
                # Parse the metric.log file and retrieve relevant information
                metric_data = parse_log(metric_path, fields)
                # Add the parsed metric to the list of all metrics
                data.append(metric_data)
    data = dict(zip(fields, [[b[k] for b in data] for k in fields]))

    new_data = dict()
    def merge_dfs(left, right):
        return pd.merge(left, right, on='name', how='outer', suffixes=('', '_right'))
    for key, value in data.items():
        value = [df for sublist in value for df in sublist]
        new_data[key] = reduce(merge_dfs, value)
        cols = ['name'] + [f'value_{i}' for i in range(len(data[key]))]
        new_data[key].columns = cols
       
        mean = new_data[key].mean().mean()
        if key.endswith('AAR'):
            mean *= 100
        print(f"{key}: {mean:.2f}")
    # print(f"new_data: {new_data}")
    return new_data

def plot_curve(fields, data, figure_path):
    #y = data[data < 10.0]
    fig, ax = plt.subplots()  #figsize=(4, 4))
    df = data[fields]
    melted_df = pd.melt(df, id_vars=['name'], value_vars=[f'value_{i}' for i in range(99)])
    print(f"melted_df: {melted_df}")
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="name", y="value", data=melted_df)
    plt.title(f"Violin plot of {fields} by name")
    plt.xticks(rotation=45)

    print(f"figure_path: {figure_path}")
    plt.savefig(figure_path,
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)

def main(args):
    
    fields = ['heavy_cdr1_AAR', 
            'heavy_cdr1_rmsd',
            'heavy_cdr2_AAR', 
            'heavy_cdr2_rmsd',
            'heavy_cdr3_AAR', 
            'heavy_cdr3_rmsd',
            'light_cdr1_AAR',
            'light_cdr1_rmsd',
            'light_cdr2_AAR',
            'light_cdr2_rmsd',
            'light_cdr3_AAR',
            'light_cdr3_rmsd',]

    data = get_data(args.log_dir, fields)
    for i in range(len(fields)): 
        f = fields[i] 
        print(f"Plotting {f}")
        plot_curve(f, data, os.path.join(args.output_dir, f'{f}_by_name.pdf'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    main(args)