import os
import argparse
import traceback
from collections import OrderedDict
from collections import defaultdict

from matplotlib import pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils import seq1
from Bio.PDB.PDBParser import PDBParser
import logging
from antiberty import AntiBERTyRunner
import csv
from tqdm import tqdm, trange
antiberty = AntiBERTyRunner()
import pdb
def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[3].strip()


def moving_average(data, window_size):
    """计算移动平均。

    :param data: 输入的数据列表或数组。
    :param window_size: 移动窗口的大小。
    :return: 移动平均值的数组。
    """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def make_pred_ppl(pdb_file):
    sep_pad_num = 50

    parser = PDBParser(QUIET=1)
    model = parser.get_structure('pdb', pdb_file)[0]
    pid, heavy_chain_id, light_chain_id, antigen_chain_id = pdb_file.split('/')[-1].split('.')[0].split('_')
    
    heavy_residues = list(model[heavy_chain_id].get_residues()) 
    light_residues = list(model[light_chain_id].get_residues())

    heavy_residue_single_letter_codes = [seq1(residue.get_resname()) for residue in heavy_residues]
    light_residue_single_letter_codes = [seq1(residue.get_resname()) for residue in light_residues]

    heavy_str_seq = ''.join(heavy_residue_single_letter_codes)
    light_str_seq = ''.join(light_residue_single_letter_codes)
    str_seq = [heavy_str_seq, light_str_seq]
    pll = antiberty.pseudo_log_likelihood(str_seq, batch_size=16)
    pll = torch.sum(pll).detach().cpu().item()


    return str_seq, pll

def make_one(pred_file):

    pred_str_seq, pll = make_pred_ppl(pred_file)

    return pll

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])
    reverse_steps = np.linspace(0.01, 1.0, 100)[::-1]
    data = defaultdict(dict)
    key=[]
    pll = []
    
    for j in range(10):
        df = pd.DataFrame()
        for i, n in enumerate(names):
            for step in reverse_steps:
                pred_file = os.path.join(args.pred_dir, str(j),n + f'@{step:.3f}' + '.pdb')
                if os.path.exists(pred_file):
                    rmsd_metric = make_one(pred_file)
                    data[n].update({
                        step : rmsd_metric,
                    })
        
        window_size = 10
        for protein, values in data.items():
            x = [float(k) for k in values.keys()]
            y = [v for v in values.values()]
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
        df.to_csv(f'/home/zhutian/Git_repo/AbFold2/traj_plot/traj_energy_without_esm/traj_pll_{j}.csv')

        # 绘制移动平均曲线
        # plt.plot(ma_x, ma_y, label=protein)

    # 设置图例
    # plt.legend()

    # # 设置 x 和 y 轴的标签
    # plt.xlabel("Time Steps")
    # plt.ylabel("Pesudo Likelihood")

    # # 设置标题
    # plt.title("Trajectory Visualization of Pesudo Likelihood")

    # # 显示图表
    # plt.savefig('./traj_pll.pdf',
    #             format='pdf',
    #             bbox_inches='tight',
    #             pad_inches=0.01)
    # csv_file_path = './traj_PLL.csv'
    # df.to_csv(csv_file_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--name_idx', type=str, required=True)
    parser.add_argument('-p', '--pred_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.output,'all_pll.log')
    handler_test = logging.FileHandler(log_file) # stdout to file
    handler_control = logging.StreamHandler()    # stdout to console

    selfdef_fmt = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(selfdef_fmt)
    handler_test.setFormatter(formatter)
    handler_control.setFormatter(formatter)
    logger.setLevel('DEBUG')           #设置了这个才会把debug以上的输出到控制台
    logger.addHandler(handler_test)    #添加handler
    logger.addHandler(handler_control)
    main(args)

