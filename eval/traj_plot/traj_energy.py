import os
from collections import defaultdict
import argparse
from scipy.interpolate import make_interp_spline

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import pdb
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

def moving_average(data, window_size):
    """计算移动平均。

    :param data: 输入的数据列表或数组。
    :param window_size: 移动窗口的大小。
    :return: 移动平均值的数组。
    """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size



def parse_energy_log(energy_log_path):
    fields_1 = ['ddG']
    # fields_1 = ['dG_wild']
    fields = ['dG_design']

    data = defaultdict(dict)
    new_data = defaultdict(dict)
    key = []
    def _parse_line(line):
        flags = line.split(':')
        if len(flags) >= 2 and any(map(lambda x: flags[-2].endswith(x), fields)):
            # print(f"flags: {flags}")
            k = (flags[-2].split('/')[-1]).split('.pdb')[0]
            pdb_name = k.split('@')[0]
            time = k.split('@')[1]
            # print(f"time: {time} k: {k}")
            v = float(flags[-1].strip())
            data[pdb_name].update(
                {time: v}
            )
            # print(f"k: {k}; v: {v}")
            # print(f"pdb_name: {pdb_name}; time: {time}; v: {v}")
            # print(f"k: {k}; v: {v}")
            key.append(pdb_name)
        if len(flags) >= 2 and any(map(lambda x: flags[-2].endswith(x), fields_1)):
            # print(f"flags: {flags}")
            k = (flags[-2].split('/')[-1]).split('.pdb')[0]
            pdb_name = k.split('@')[0]
            time = k.split('@')[1]
            v = float(flags[-1].strip())
            data[pdb_name].update(
                {time: v}
            )
            # print(f"pdb_name: {pdb_name}; time: {time:.3f}; v: {v:.3f}")
            # print(f"k: {k}; v: {v}")
            # key.append(k)
    with open(energy_log_path) as f:
        for line in f:
            _parse_line(line)
    return data, new_data, key


def get_data(log_dir, rank=None):
    # data = []
    # data_1 = []
    # Check if 'metric.log' is in the files
    path = os.path.join(log_dir)
    if 'relax.log' in os.listdir(path):
        energy_log_path = os.path.join(path, 'relax.log')
        # Parse the metric.log file and retrieve relevant information
        energy, energy_1, key = parse_energy_log(energy_log_path)
    data = energy

    df = pd.DataFrame()
    window_size = 20
    for protein, values in data.items():
        x = [float(k) for k in values.keys()]
        y = [v for v in values.values()]
        
        # 对 x 和 y 按 x 排序（这很重要，以确保线条正确连接点）
        sorted_pairs = sorted(zip(x, y))
        sorted_x, sorted_y = zip(*sorted_pairs)

        # 生成样条插值模型
        ma_y = moving_average(sorted_y, window_size)

        # 由于移动平均减少了数据点的数量，我们需要调整 x 轴的数据点
        ma_x = sorted_x[window_size - 1:]
        protein_df = pd.DataFrame(list(values.items()), columns=['Key', protein])
        protein_df.set_index('Key', inplace=True)

        # 合并到主 DataFrame
        if df.empty:
            df = protein_df
        else:
            df = df.join(protein_df, how='outer')


        # 绘制移动平均曲线
    #     plt.plot(ma_x, ma_y, label=protein)
    #     # 绘制点和线
    #     # plt.plot(sorted_x, sorted_y, '-', label=protein)

    # # 设置图例
    # plt.legend()

    # # 设置 x 和 y 轴的标签
    # plt.xlabel("Time Steps")
    # plt.ylabel("Binding Affinity")

    # # 设置标题
    # plt.title("Trajectory Visualization of Binding Affinity")

    # # 显示图表
    # plt.savefig('./traj_energy.pdf',
    #             format='pdf',
    #             bbox_inches='tight',
    #             pad_inches=0.01)

    # csv_file_path = './traj_energy.csv'
    # df.to_csv(csv_file_path)


    # new_data = dict()
    # new_data_1 = dict()
    # print(f"new_data: {new_data}")
    # for key, value in data.items():
    #     if key != '7stf_H_L_AC':
    #         flattened_list = [item for sublist in value for item in sublist]
    #         new_data[key] = flattened_list
    # for key, value in data_1.items():
    #     if key != '7stf_H_L_AC':
    #         flattened_list = [item for sublist in value for item in sublist]
    #         new_data_1[key] = flattened_list
    # print(f"data: {new_data}")
    return df

def plot_curve( data, figure_path, agg_factor=1):
    # y = data[data < 50.0]
    # print(f"data: {data}")
    # data = {key: remove_outliers(value) for key, value in data.items()}
    proportions = {key: sum(1 for value in values if value < 0) / len(values) for key, values in data.items()}
    overall_proportion = sum(1 for values in data.values() for value in values if value < 0) / sum(len(values) for values in data.values())
    overall_energy = sum(value for values in data.values() for value in values) / sum(len(values) for values in data.values())
    # print(f"proportions: {proportions}")
    # print(f"overall: {overall_proportion} {overall_energy}")
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

    for i in range(10):
        csv_file_path = f'./traj_energy_without_esm/traj_energy_{i}.csv'
        log_dir = os.path.join(args.log_dir, str(i))
        energy_data = get_data(log_dir)
        energy_data.to_csv(csv_file_path)

    # plot_curve(energy_data,figure_path )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    main(args)
