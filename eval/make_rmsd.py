import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

from abx.common.ab_utils import calc_ab_metrics
from abx.common import residue_constants

from Bio.SeqUtils import seq1
from Bio.PDB.PDBParser import PDBParser
import logging


def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[3].strip()

def make_pred_coords(pdb_file, heavy_len, light_len, alg_type):
    sep_pad_num = 50

    parser = PDBParser(QUIET=1)
    model = parser.get_structure('pdb', pdb_file)[0]
    pid, heavy_chain_id, light_chain_id, antigen_chain_id = pdb_file.split('/')[-1].split('.')[0].split('_')
    if alg_type in ['omegafold']:
        residues = list(model.get_residues())
        residues = residues[:heavy_len] + residues[heavy_len+sep_pad_num:heavy_len+sep_pad_num+light_len]
    elif alg_type in ['esmfold']:
        residues = list(model['A'].get_residues()) + list(model['B'].get_residues())
    else:
        residues = list(model[heavy_chain_id].get_residues()) + list(model[light_chain_id].get_residues())

    residue_single_letter_codes = [seq1(residue.get_resname()) for residue in residues]
    str_seq = ''.join(residue_single_letter_codes)
    
    coords = np.zeros((len(residues), 3))
    for i, r in enumerate(residues):
        coords[i] = r['CA'].get_coord()

    return coords, str_seq

def make_one(name, gt_npz_file, pred_file, alg_type):

    gt_fea = np.load(gt_npz_file)
    gt_coords = np.concatenate([gt_fea['heavy_coords'], gt_fea['light_coords']], axis=0)
    gt_coord_mask = np.concatenate([gt_fea['heavy_coord_mask'], gt_fea['light_coord_mask']], axis=0)
    cdr_def = np.concatenate([gt_fea['heavy_cdr_def'], gt_fea['light_cdr_def']], axis=0)
    str_heavy_seq, str_light_seq = str(gt_fea['heavy_str_seq']), str(gt_fea['light_str_seq'])
    gt_str_seq = ''.join([str_heavy_seq, str_light_seq])
    ca_mask = gt_coord_mask[:, 1]
    gt_ca = gt_coords[:,1]
    pred_ca, pred_str_seq = make_pred_coords(pred_file, len(str_heavy_seq), len(str_light_seq), alg_type)
    assert (gt_ca.shape[0] == pred_ca.shape[0] and gt_ca.shape[0] == cdr_def.shape[0])
    ab_metrics = calc_ab_metrics(gt_ca, pred_ca, ca_mask, cdr_def, gt_str_seq, pred_str_seq)

    return ab_metrics

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])
    metrics = []
    for i, n in enumerate(names):
        gt_file = os.path.join(args.gt_dir, n + '.npz')
        pred_file = os.path.join(args.pred_dir, n + '.pdb')
        relax_file = os.path.join(args.pred_dir, n + '_relaxed.pdb')
        if os.path.exists(gt_file) and os.path.exists(pred_file):
            one_metric = OrderedDict({'name' : n})
            rmsd_metric = make_one(n, gt_file, pred_file, args.alg_type)
            one_metric.update(rmsd_metric)
            metrics.append(one_metric)
        elif os.path.exists(gt_file) and os.path.exists(relax_file):
            one_metric = OrderedDict({'name' : n})
            rmsd_metric = make_one(n, gt_file, relax_file, args.alg_type)
            one_metric.update(rmsd_metric)
            metrics.append(one_metric)
    columns = metrics[0].keys()
    metrics = zip(*map(lambda x:x.values(), metrics))

    df = pd.DataFrame(dict(zip(columns,metrics)))

    df = df[df['heavy_cdr3_len'] > 2]
    df = df[df['full_len'] > 200]
    output_dir = os.path.join(args.output, 'metric.csv')
    df.to_csv(output_dir, sep='\t', index=False)
    df = df.dropna()

    # print('total', df.shape[0])
    total_AAR = 0
    total_CDR3_AAR = 0
    total_CDR_RMSD = 0
    total_CDR3_RMSD = 0
    for r in df.columns:
        if r.endswith('rmsd'):
            rmsd = df[r].values
            mean = np.mean(rmsd)
            if r.__contains__('fr'):
                continue
            if r.__contains__('cdr'):
                total_CDR_RMSD += mean
            if r.__contains__('cdr3'):
                total_CDR3_RMSD+=mean
            logger.info(f'{r:15s} {mean:6.2f}')
        if r.endswith('AAR'):
            AAR = df[r].values
            mean = np.mean(AAR)
            if r.__contains__('fr'):
                continue
            if r.__contains__('cdr'):
                total_AAR += mean
            if r.__contains__('cdr3'):
                total_CDR3_AAR+=mean
            logger.info(f'{r:15s}        {mean:6.2f}')
    total_AAR/=6
    total_CDR3_AAR/=2
    total_CDR_RMSD/=6
    total_CDR3_RMSD/=2
    logger.info("*" * 100)
    logger.info(f"total CDR AAR:\t{total_AAR:6.2f}")
    logger.info(f"total CDR3 AAR:\t{total_CDR3_AAR:6.2f}")
    logger.info(f"total CDR RMSD:\t{total_CDR_RMSD:6.2f}")
    logger.info(f"total CDR3 RMSD: {total_CDR3_RMSD:5.2f}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--name_idx', type=str, required=True)
    parser.add_argument('-t', '--alg_type', type=str, choices=['igfold', 'esmfold', 'omegafold', 'abfold'], required=True)
    parser.add_argument('-g', '--gt_dir', type=str, required=True)
    parser.add_argument('-p', '--pred_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.output,'all_metric.log')
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

