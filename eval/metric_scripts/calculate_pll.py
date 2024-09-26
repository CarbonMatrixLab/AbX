import os
import argparse
import traceback
from collections import OrderedDict

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

def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[3].strip()

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
    pll = []
    for j in range(30):
        metrics = []

        for i, n in tqdm(enumerate(names)):
            pred_file = os.path.join(args.pred_dir, str(j), n + '.pdb')
            # pred_file = os.path.join(args.pred_dir, n + '.pdb')

            print(f"pred_file: {pred_file}")
            if os.path.exists(pred_file):
                one_metric = OrderedDict({'name' : n})
                rmsd_metric = make_one(pred_file)
                one_metric.update({
                    'pll' : rmsd_metric,
                })
                metrics.append(one_metric)
        columns = metrics[0].keys()
        metrics = zip(*map(lambda x:x.values(), metrics))

        df = pd.DataFrame(dict(zip(columns,metrics)))

        # output_dir = os.path.join(args.output,  str(j), 'pll.csv')

        output_dir = os.path.join(args.output, 'pll.csv')
        df.to_csv(output_dir, sep='\t', index=False)
        df = df.dropna()
        ppl = df['pll']
        mean = np.mean(ppl)
        pll.append(mean)
        logger.info(f'pll {mean:6.2f}')
    results = sum(pll)/len(pll)
    print(f"logger.info: pll {results:6.2f}")
    
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

